"""CPU preview ownership, error timing, eligibility and Trainer boundaries."""
from dataclasses import replace

import pytest
import torch

from mahjong import contracts as c
from mahjong.app.assembly import build_runtime
from mahjong.data.mahjong import policy
from mahjong.data.mahjong.shared_policy import MahjongSharedPolicyTrainSource, collate_shared_group
from mahjong.training.lookahead import CPUBatchLookahead
from tests.unit.data.test_shared_policy import Dataset, assert_state_equal
from tests.unit.models.test_qwen3_shared import config, samples
from mahjong.app.assembly import build_model
from tests.unit.training.test_loop import Run, config as loop_config
from tests.unit.training.test_runtime import assert_equal


def source(monkeypatch, *, shuffle=True, batch_size=2):
    monkeypatch.setattr(policy, '_Dataset', Dataset)
    return MahjongSharedPolicyTrainSource(build_dir='unused', seed=7, batch_size=batch_size,
        max_sequence_length=128, max_tokens=128, shuffle=shuffle)


@pytest.mark.parametrize('shuffle', [False, True])
def test_preview_retains_committed_state_and_rng_through_empty_games_epochs(monkeypatch, shuffle):
    plain = source(monkeypatch, shuffle=shuffle)
    ahead = source(monkeypatch, shuffle=shuffle)
    initial_rng = torch.get_rng_state().clone()
    for _ in range(16):
        a, b = plain.acquire(), ahead.acquire()
        assert_equal(a.batch, b.batch)
        held = ahead.state_dict()
        preview = ahead.preview_next_batch()
        assert preview is ahead.preview_next_batch()
        assert_state_equal(held, ahead.state_dict())
        with pytest.raises(c.MahjongError, match='lease already open'):
            ahead.acquire()
        plain.commit(a.token); ahead.commit(b.token)
        assert_state_equal(plain.state_dict(), ahead.state_dict())
        assert ahead._batch_preview is not None
    assert torch.equal(initial_rng, torch.get_rng_state())
    ahead.discard_batch_preview()
    restored = source(monkeypatch, shuffle=shuffle)
    restored.load_state_dict(ahead.state_dict())
    assert_equal(ahead.acquire().batch, restored.acquire().batch)
    plain.close(); ahead.close(); restored.close()


@pytest.mark.parametrize('action', ['consume', 'discard', 'restore', 'invalid_restore', 'close'])
def test_future_read_error_is_deferred_and_can_be_discarded(monkeypatch, action):
    train = source(monkeypatch, shuffle=False, batch_size=8)
    saved = train.state_dict()
    def read(self, index, **kwargs):
        if index == 1:
            raise ValueError('next game failed')
        return samples()
    monkeypatch.setattr(Dataset, 'samples_for_game', read)
    lease = train.acquire()
    held = train.state_dict()
    with pytest.raises(ValueError, match='next game failed'):
        train.preview_next_batch()
    assert_state_equal(held, train.state_dict())
    train.commit(lease.token)  # Current update/lease can still commit.
    if action == 'consume':
        with pytest.raises(ValueError, match='next game failed'):
            train.acquire()
    elif action == 'restore':
        train.load_state_dict(saved)
        assert train._batch_preview is None
        assert train.acquire().batch.sample_count == 3
    elif action == 'invalid_restore':
        with pytest.raises(c.MahjongError, match='CHECKPOINT_INVALID'):
            train.load_state_dict({})
        assert train._batch_preview is None
    elif action == 'discard':
        train.discard_batch_preview()
        assert train._batch_preview is None
    train.close()
    assert train._batch_preview is None


def cpu_runtime_with_preview(monkeypatch):
    runtime = build_runtime(config(), training=True)
    # Exercise CPU cache mechanics without pretending the production CPU gate
    # enables previews. Real GPU eligibility/CLI is covered separately.
    monkeypatch.setattr(runtime, 'supports_cpu_lookahead', runtime.raw_model.supports_cpu_lookahead)
    return runtime


@pytest.mark.parametrize('change', ['tensor', 'metadata', 'different_batch', 'hook', 'prepare_override'])
def test_runtime_invalidates_preview_and_keeps_public_prepare_checks(monkeypatch, change):
    runtime = cpu_runtime_with_preview(monkeypatch)
    batch = collate_shared_group(samples())
    runtime.prepare_cpu_batch(batch)
    assert runtime._cpu_batch_preview is not None
    handle = None
    if change == 'tensor':
        batch.payload['target_indices'][0] = 999
        with pytest.raises(c.MahjongError, match='target index'):
            runtime.compute(batch, phase='train')
    elif change == 'metadata':
        batch = replace(batch, input_schema='invalid')
        with pytest.raises(c.MahjongError, match='mapping'):
            runtime.compute(batch, phase='train')
    elif change == 'different_batch':
        other = collate_shared_group(samples()[:1])
        result = runtime.compute(other, phase='train')
        assert result.metrics['loss'].denominator == 3
    elif change == 'hook':
        handle = runtime.raw_model.register_forward_pre_hook(lambda *args: None)
        assert not runtime.raw_model.supports_cpu_lookahead()
        runtime.compute(batch, phase='train')
    else:
        def reject(*args):
            raise ValueError('custom prepare')
        monkeypatch.setattr(runtime.raw_model, 'prepare_batch', reject)
        assert not runtime.raw_model.supports_cpu_lookahead()
        with pytest.raises(ValueError, match='custom prepare'):
            runtime.compute(batch, phase='train')
    if handle is not None: handle.remove()
    assert runtime._cpu_batch_preview is None


def test_prepare_error_occurs_on_actual_compute_and_import_eval_clear_cache(monkeypatch):
    runtime = cpu_runtime_with_preview(monkeypatch)
    invalid = collate_shared_group(samples())
    invalid.payload['target_indices'][0] = 999
    runtime.prepare_cpu_batch(invalid)
    assert isinstance(runtime._cpu_batch_preview[2], c.MahjongError)
    with pytest.raises(c.MahjongError, match='target index'):
        runtime.compute(invalid, phase='train')
    batch = collate_shared_group(samples())
    runtime.prepare_cpu_batch(batch)
    state = runtime.export_state()
    runtime.import_state(state)
    assert runtime._cpu_batch_preview is None
    runtime.prepare_cpu_batch(batch)
    with runtime.evaluation_scope(43):
        assert runtime._cpu_batch_preview is None
    assert runtime._cpu_batch_preview is None


def test_cpu_and_modified_source_paths_keep_original_execution(monkeypatch):
    train = source(monkeypatch)
    runtime = build_runtime(config(), training=True)
    assert CPUBatchLookahead.create(train, runtime) is None
    assert train.supports_batch_preview()
    monkeypatch.setattr(train, 'acquire', lambda: None)
    assert not train.supports_batch_preview()


@pytest.mark.parametrize('failure', [False, True])
def test_trainer_clears_preview_at_validation_save_and_all_exits(tmp_path, monkeypatch, failure):
    previews, clears, state = [], [], {'cached': False}
    class Preview:
        def prepare(self):
            assert run.source.has_open_lease()
            assert run.runtime.updates == run.trainer.progress.global_step
            previews.append(run.trainer.progress.global_step)
            state['cached'] = True
        def clear(self):
            state['cached'] = False
            clears.append(True)
    monkeypatch.setattr(CPUBatchLookahead, 'create', classmethod(lambda cls, source, runtime: Preview()))
    cfg = loop_config(7, val=3, save=4)
    run = Run(tmp_path, cfg)
    saved = run.store.save
    def save(*args, **kwargs):
        assert not state['cached']
        return saved(*args, **kwargs)
    monkeypatch.setattr(run.store, 'save', save)
    if failure:
        def bad_gradients():
            raise c.MahjongError('NONFINITE_GRADIENT', 'injected', 'training')
        monkeypatch.setattr(run.runtime, 'check_gradients', bad_gradients)
        with pytest.raises(c.MahjongError, match='NONFINITE_GRADIENT'):
            run.execute()
        assert previews == [0]
    else:
        result = run.execute()
        assert result.outcome == 'completed'
        assert previews == [0, 1, 4]
    assert clears and not state['cached']


def test_known_stop_finishes_current_update_without_preview(tmp_path, monkeypatch):
    class Preview:
        def prepare(self): raise AssertionError('preview after stop')
        def clear(self): pass
    monkeypatch.setattr(CPUBatchLookahead, 'create', classmethod(lambda *args: Preview()))
    run = Run(tmp_path, loop_config(3))
    backward = run.runtime.backward
    def stop_after_backward(loss):
        backward(loss)
        run.stop.request(15)
    monkeypatch.setattr(run.runtime, 'backward', stop_after_backward)
    assert run.execute().outcome == 'interrupted'
    assert run.trainer.progress.global_step == 1


@pytest.mark.parametrize('fault', ['read', 'prepare', 'stop', 'gradient'])
def test_real_trainer_defers_future_fault_and_preserves_current_error_priority(tmp_path, monkeypatch, fault):
    train = source(monkeypatch, shuffle=False, batch_size=8)
    run = Run(tmp_path, loop_config(3, enabled=False), source=train, model=build_model(config()))
    monkeypatch.setattr(run.runtime, 'supports_cpu_lookahead', run.model.supports_cpu_lookahead)
    def read(self, index, **kwargs):
        if index == 1:
            if fault == 'stop': run.stop.request(15)
            if fault != 'prepare': raise ValueError('future source failed')
            rows = samples()
            rows[0]['input_ids'][-1] = 9999
            return rows
        return samples()
    monkeypatch.setattr(Dataset, 'samples_for_game', read)
    if fault == 'gradient':
        def bad_gradient():
            raise c.MahjongError('NONFINITE_GRADIENT', 'current gradient failed', 'training')
        monkeypatch.setattr(run.runtime, 'check_gradients', bad_gradient)
    if fault == 'stop':
        result = run.execute()
        assert result.outcome == 'interrupted' and result.global_step == 1
        assert run.saved()[-1].progress.global_step == 1
    else:
        with pytest.raises(c.MahjongError) as error:
            run.execute()
        expected = {'read': 'DATA_READ_FAILED', 'prepare': 'CONTRACT_MISMATCH',
                    'gradient': 'NONFINITE_GRADIENT'}[fault]
        assert error.value.code == expected
        assert run.trainer.progress.global_step == (0 if fault == 'gradient' else 1)
        assert error.value.details['safe_to_checkpoint'] is (fault == 'read')
        assert run.trainer.progress.batches_read == (2 if fault == 'prepare' else 1)
    assert train._batch_preview is None and run.runtime._cpu_batch_preview is None
    train.close()
