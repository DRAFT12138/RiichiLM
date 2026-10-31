"""Exercise real Fabric BF16 updates and exact safe-state restoration on CPU."""
from pathlib import Path

import pytest
import torch

from mahjong.app.assembly import build_runtime
from mahjong.app.config import parse_new
from mahjong.contracts import MahjongError
from mahjong.data.mahjong.policy import _collate
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
from tests.unit.training.test_runtime import assert_equal


def config():
    return parse_new(['data.build_dir=unused', 'data.shared_context=false', 'runtime.device=cpu',
        'model.num_hidden_layers=1', 'model.hidden_size=32', 'model.intermediate_size=64',
        'model.num_attention_heads=4', 'model.num_key_value_heads=2', 'model.head_dim=8'],
        Path.cwd(), config_name='pretrain')


def batch():
    return _collate([dict(input_ids=[1,4,5], candidates=[[7,30],[8,30]],
                         target_index=0, sample_id='a')], CONTEXT_SCHEMA)


def update(runtime):
    runtime.zero_grad()
    result = runtime.compute(batch(), phase='train')
    runtime.backward(result.loss)
    runtime.check_gradients()
    runtime.step()
    runtime.zero_grad()
    return result


def test_bf16_is_used_and_full_state_restores_exact_next_update(tmp_path):
    cfg = config()
    assert cfg['runtime']['precision'] == 'bf16-mixed'
    assert cfg['model']['max_position_embeddings'] == 8192
    runtime = build_runtime(cfg, training=True)
    dtypes = []
    handle = runtime.raw_model.causal_lm.lm_head.register_forward_hook(
        lambda m, args, out: dtypes.append(out.dtype))
    update(runtime)
    handle.remove()
    assert dtypes == [torch.bfloat16]
    assert {p.dtype for p in runtime.raw_model.parameters()} == {torch.float32}
    state, rng = runtime.export_state(), runtime.capture_rng()
    assert state.scaler is None
    assert {v.dtype for s in state.optimizer['state'].values() for v in s.values()
            if isinstance(v, torch.Tensor)} == {torch.float32}
    update(runtime)
    expected = runtime.export_state()
    destination = build_runtime(cfg, training=True)
    destination.import_state(state)
    destination.restore_rng(rng)
    update(destination)
    assert_equal(destination.export_state(), expected)
    assert_equal(destination.capture_rng(), runtime.capture_rng())
    before = destination.export_state()
    with destination.evaluation_scope(7):
        assert torch.isfinite(destination.compute(batch(), phase='eval').loss)
    assert_equal(destination.export_state(), before)


def test_precision_cannot_be_changed_on_resume():
    from mahjong.app.config import merge_resume
    with pytest.raises(MahjongError, match='runtime.precision'):
        merge_resume(config(), ['runtime.precision=32-true'])
