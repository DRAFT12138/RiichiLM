"""Policy sampling boundaries, committed resume state and bounded validation."""
from pathlib import Path

import pytest
import torch

from mahjong.app.config import parse_new
from mahjong.contracts import MahjongError
from mahjong.data.mahjong import policy


class Dataset:
    fingerprint = 'test-policy-data'

    def __init__(self, *args, **kwargs):
        pass

    def split_indices(self, validation):
        return torch.arange(3)

    def samples_for_game(self, index):
        # Two empty games expose the transition between shuffled epochs.
        if index != 0:
            return []
        return [dict(input_ids=[1, 2], candidates=[[3], [4]],
                     target_index=0, sample_id='labelled-game')]

    def close(self):
        pass


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.setattr(policy, '_Dataset', Dataset)


def test_epoch_transition_keeps_checkpoint_state_committed_and_resumes(dataset):
    source = policy.MahjongPolicyTrainSource(build_dir='unused', seed=7, batch_size=4)
    before = source.state_dict()
    lease = source.acquire()
    pending = source.state_dict()
    for key in before:
        if isinstance(before[key], torch.Tensor):
            assert torch.equal(before[key], pending[key])
        else:
            assert before[key] == pending[key]
    assert lease.batch.sample_count == 4
    source.commit(lease.token)
    assert source.epoch >= 3
    restored = policy.MahjongPolicyTrainSource(build_dir='unused', seed=7, batch_size=4)
    restored.load_state_dict(source.state_dict())
    expected, actual = source.acquire(), restored.acquire()
    assert actual.batch.sample_ids == expected.batch.sample_ids
    source.commit(expected.token)
    restored.commit(actual.token)
    assert source.epoch == restored.epoch
    assert torch.equal(source.permutation, restored.permutation)
    assert torch.equal(source.generator.get_state(), restored.generator.get_state())


def test_empty_split_fails_without_looping_forever(dataset, monkeypatch):
    monkeypatch.setattr(Dataset, 'samples_for_game', lambda self, index: [])
    source = policy.MahjongPolicyTrainSource(build_dir='unused', seed=7)
    with pytest.raises(MahjongError, match='no labelled decisions'):
        source.acquire()
    assert not source.has_open_lease()


def test_validation_limit_keeps_partial_batch_and_changes_descriptor(dataset, monkeypatch):
    def samples(self, index):
        return [dict(input_ids=[1, 2], candidates=[[3], [4]], target_index=0,
                     sample_id=f'{index}/{n}') for n in range(3)]
    monkeypatch.setattr(Dataset, 'samples_for_game', samples)
    limited = policy.MahjongPolicyEvalSource(build_dir='unused', batch_size=2, max_samples=5)
    full = policy.MahjongPolicyEvalSource(build_dir='unused', batch_size=2)
    batches = list(limited.iter_batches())
    assert [b.sample_count for b in batches] == [2, 2, 1]
    assert [s for b in batches for s in b.sample_ids] == ['0/0', '0/1', '0/2', '1/0', '1/1']
    assert sum(b.sample_count for b in full.iter_batches()) == 9
    assert limited.descriptor().config_fingerprint != full.descriptor().config_fingerprint


@pytest.mark.parametrize('value', ['0', '-1', 'true', '1.5'])
def test_validation_limit_rejects_invalid_config(value):
    with pytest.raises(MahjongError, match='validation_max_samples'):
        parse_new([f'data.validation_max_samples={value}'], Path.cwd())


def test_pretrain_config_passes_validation_limit_to_source(dataset):
    from mahjong.app.assembly import build_eval_source
    config = parse_new(['data.build_dir=unused', 'data.validation_max_samples=5'],
                       Path.cwd(), config_name='pretrain')
    source = build_eval_source(config)
    assert getattr(source, "corpus", source).max_samples == 5
    if config["evaluation"]["nanikiru_dir"] is not None:
        assert source.nanikiru.ready == 601
    else:
        assert not hasattr(source, "nanikiru")


def test_dataset_reserves_context_for_longest_action_without_truncation(monkeypatch):
    class Reader:
        catalog = {'binding': {'storage': 'mjai-clean-text/3', 'backend': 'rust'}, 'build_id': 'fixture'}
        manifest = {'capabilities': {'legal_actions': True, 'action_labels': True}, 'shards': []}

        def __init__(self, path): pass
        def close(self): pass
        def iter_games(self):
            return iter([{'game_id': str(n), 'source_key': str(n)} for n in range(2)])
        def iter_round_refs(self, source): return iter([(source, 0)])
        def read_prefix(self, *args): return [1, 2, 3], {}
        def read_round(self, *args):
            return {'facts': {'policy_decisions': [dict(viewer=0, window_id=0,
                candidates=[[7, 8, 30], [9, 30]], target_index=1)]}}

    monkeypatch.setattr(policy, 'CleaningReader', Reader)
    def source(limit):
        return policy._Dataset('fixture', validation_fraction=.5, split_seed=7,
                               max_sequence_length=limit)
    # Target length alone would fit in 5; all legal continuations need 6.
    assert source(5).samples_for_game(0) == []
    samples = source(6).samples_for_game(0)
    assert len(samples) == 1
    assert samples[0]['input_ids'] == [1, 2, 3]
    assert samples[0]['candidates'] == [[7, 8, 30], [9, 30]]


def test_cross_round_overflow_is_an_error_and_keeps_the_source_identity(monkeypatch):
    from mahjong.policy import context
    from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
    class Reader:
        catalog = {'binding': {'storage': 'mjai-clean-text/3', 'backend': 'rust'}, 'build_id': 'fixture'}
        manifest = {'capabilities': {'legal_actions': True, 'action_labels': True}, 'shards': []}
        def __init__(self, path): pass
        def close(self): pass
        def iter_games(self):
            return iter([{'game_id': str(n), 'source_key': str(n)} for n in range(2)])
        def iter_round_refs(self, source): return iter([(source, 4)])
        def read_round(self, *args):
            return {'facts': {'policy_decisions': [dict(viewer=2, window_id=100,
                candidates=[[7, 8, 30], [9, 30]], target_index=1)]}}
    monkeypatch.setattr(policy, 'CleaningReader', Reader)
    monkeypatch.setattr(context, 'read_context_prefix', lambda *args, **kwargs: ([1]*8190, {}))
    dataset = policy._Dataset('fixture', validation_fraction=.5, split_seed=7,
                              max_sequence_length=8192, input_schema=CONTEXT_SCHEMA)
    with pytest.raises(MahjongError, match='CONTEXT_TOO_LONG') as error:
        dataset.samples_for_game(0)
    assert error.value.details == dict(source='0', round_id=4, window_id=100, viewer=2,
                                      prefix_length=8190, longest_suffix=3, context=8192)


def test_history_window_changes_context_data_identity_without_changing_single_round(monkeypatch):
    from types import SimpleNamespace
    from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, CONTEXT_BINDING, LEGACY_SCHEMA
    reader = SimpleNamespace(catalog={'build_id': 'fixture', 'binding': {}}, manifest={'shards': []})
    def fingerprint(schema):
        return policy.dataset_fingerprint(reader, split_seed=7, validation_fraction=.02,
            max_sequence_length=8192, input_schema=schema)
    legacy = fingerprint(LEGACY_SCHEMA)
    three = fingerprint(CONTEXT_SCHEMA)
    with monkeypatch.context() as patch:
        patch.setitem(CONTEXT_BINDING, 'previous_rounds', 4)
        assert fingerprint(CONTEXT_SCHEMA) != three
        assert fingerprint(LEGACY_SCHEMA) == legacy
