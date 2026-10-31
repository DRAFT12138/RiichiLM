"""Finite-scan fault semantics and fused Adam numerical/recovery boundaries."""
import copy

import pytest
import torch

from mahjong import contracts as c
from mahjong.app.assembly import build_runtime
from mahjong.data.mahjong.shared_policy import collate_shared_group
from mahjong.training.runtime import FabricRuntime
from tests.unit.models.test_qwen3_shared import config, samples
from tests.unit.training.test_runtime import assert_equal


def gradient_runtime(device):
    runtime = FabricRuntime(device=device)
    runtime.raw_model = torch.nn.Module()
    for name, dtype in [('single', torch.float32), ('half', torch.float16),
                        ('brain', torch.bfloat16), ('double', torch.float64),
                        ('complex', torch.complex64)]:
        p = torch.nn.Parameter(torch.zeros(2, 3, device=device, dtype=dtype))
        runtime.raw_model.register_parameter('grad_' + name, p)
        p.grad = torch.ones_like(p)
    return runtime


def check_faults(runtime):
    params = list(runtime.raw_model.parameters())
    for p in params:
        original = p.grad.clone()
        runtime.check_gradients()
        assert torch.equal(p.grad, original)
        for bad in (float('nan'), float('inf'), -float('inf')):
            p.grad[0, 0] = bad
            with pytest.raises(c.MahjongError) as error:
                runtime.check_gradients()
            assert error.value.code == 'NONFINITE_GRADIENT'
            p.grad.copy_(original)
            runtime.check_gradients()  # A previous fault must not poison scratch.
    # Non-contiguous dense gradients use the conventional check.
    p = params[0]
    p.grad = torch.ones(3, 2, device=p.device).t()
    assert not p.grad.is_contiguous()
    runtime.check_gradients()
    p.grad[1, 1] = float('nan')
    with pytest.raises(c.MahjongError, match='NONFINITE_GRADIENT'):
        runtime.check_gradients()
    p.grad = torch.ones_like(p)
    # Sparse COO fallback checks stored values rather than unsupported all().
    p.grad = torch.sparse_coo_tensor([[0], [1]], [1.], p.shape, device=p.device)
    runtime.check_gradients()
    p.grad._values()[0] = float('inf')
    with pytest.raises(c.MahjongError, match='NONFINITE_GRADIENT'):
        runtime.check_gradients()
    for p in params:
        p.grad = None
    with pytest.raises(c.MahjongError, match='CONTRACT_MISMATCH'):
        runtime.check_gradients()


def test_finite_scan_cpu_fallback():
    runtime = gradient_runtime('cpu')
    check_faults(runtime)
    assert runtime._gradient_check_scratch == {}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_fused_scan_mixed_dtypes_faults_reset_and_fallback(monkeypatch):
    torch.cuda.set_per_process_memory_fraction(.4)
    runtime = gradient_runtime('cuda:0')
    p = next(runtime.raw_model.parameters())
    p.grad.copy_(torch.tensor([[0., -0., 1e-45], [-1e-45, 1e-38, 3e38]], device='cuda'))
    bits = p.grad.view(torch.int32).clone()
    runtime.check_gradients()
    assert torch.equal(p.grad.view(torch.int32), bits)
    p.grad.fill_(1)
    check_faults(runtime)
    assert len(runtime._gradient_check_scratch) == 1
    # Builds without the private primitive retain correctness, at lower speed.
    monkeypatch.delattr(torch, '_amp_foreach_non_finite_check_and_unscale_')
    check_faults(gradient_runtime('cuda:0'))


def advance(runtime):
    runtime.zero_grad()
    result = runtime.compute(collate_shared_group(samples()), phase='train')
    runtime.backward(result.loss)
    runtime.check_gradients()
    runtime.step()
    return result


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_fused_adam_matches_legacy_updates():
    torch.cuda.set_per_process_memory_fraction(.4)
    cfg = config('cuda:0', 'bf16-mixed')
    legacy, fused = [build_runtime(cfg, training=True) for _ in range(2)]
    for group in legacy.raw_optimizer.param_groups:
        group['fused'] = False
    assert fused.raw_optimizer.param_groups[0]['fused'] is True
    for _ in range(5):
        a, b = advance(legacy), advance(fused)
        torch.testing.assert_close(a.loss, b.loss, atol=1e-6, rtol=1e-5)
        for p, q in zip(legacy.raw_model.parameters(), fused.raw_model.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    assert all(p.dtype == torch.float32 for p in fused.raw_model.parameters())
    assert all(v.dtype == torch.float32 for state in fused.raw_optimizer.state.values()
               for v in state.values() if isinstance(v, torch.Tensor))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
@pytest.mark.parametrize('saved_fused,saved_foreach', [(False, False), (False, True),
                                                     (None, None), (True, False)])
def test_adam_resume_migrates_execution_and_preserves_state(saved_fused, saved_foreach, tmp_path):
    torch.cuda.set_per_process_memory_fraction(.4)
    cfg = config('cuda:0', 'bf16-mixed')
    source = build_runtime(cfg, training=True)
    for group in source.raw_optimizer.param_groups:
        group['fused'] = saved_fused
        group['foreach'] = saved_foreach
        # Deliberately differ from constructor defaults: resume must retain all
        # mathematical hyperparameters, not only the moment tensors.
        group.update(lr=.0007, betas=(.8, .97), eps=1e-7, weight_decay=.01, amsgrad=True)
    for _ in range(3):
        advance(source)
    state = source.export_state()
    filename = tmp_path / 'runtime.pt'
    torch.save(c.encode_contract(state), filename)
    loaded = c.decode_contract(c.RuntimeState, torch.load(filename, weights_only=True, map_location='cpu'))
    target = build_runtime(cfg, training=True)
    assert target.raw_optimizer.param_groups[0]['fused'] is True
    original = copy.deepcopy(loaded)
    rng = target.capture_rng()
    target.import_state(loaded)
    assert_equal(target.capture_rng(), rng)
    assert_equal(loaded, original)  # Caller/checkpoint ownership is preserved.
    expected = copy.deepcopy(loaded)
    for group in expected.optimizer['param_groups']:
        group.update(fused=True, foreach=False)
    assert_equal(target.export_state(), expected)
    for values in target.raw_optimizer.state.values():
        assert values['step'].device.type == 'cuda'
        assert values['step'].dtype == torch.float32
        assert values['step'].item() == 3
    # The migrated update stays close to continuing the legacy implementation.
    a, b = advance(source), advance(target)
    torch.testing.assert_close(a.loss, b.loss, atol=1e-6, rtol=1e-5)
    for p, q in zip(source.raw_model.parameters(), target.raw_model.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    # Once migrated, another disk restore reproduces the exact next update.
    torch.save(c.encode_contract(target.export_state()), filename)
    migrated = c.decode_contract(c.RuntimeState, torch.load(filename, weights_only=True, map_location='cpu'))
    restored = build_runtime(cfg, training=True)
    restored.import_state(migrated)
    assert_equal(target.export_state(), restored.export_state())
    advance(target)
    advance(restored)
    assert_equal(target.export_state(), restored.export_state())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_explicit_nonfused_gpu_optimizer_preserves_saved_mode():
    """RL and other runtimes without the NTP fused policy keep their mode."""
    torch.cuda.set_per_process_memory_fraction(.4)
    cfg = config('cuda:0', 'bf16-mixed')
    source = build_runtime(cfg, training=True)
    source.raw_optimizer.defaults['fused'] = False
    for group in source.raw_optimizer.param_groups:
        group['fused'] = False
    advance(source)
    state = source.export_state()
    target = build_runtime(cfg, training=True)
    target.raw_optimizer.defaults['fused'] = False
    target.import_state(state)
    assert target.raw_optimizer.param_groups[0]['fused'] is False
    assert all(s['step'].device.type == 'cpu' for s in target.raw_optimizer.state.values())
    advance(source)
    advance(target)
    assert_equal(source.export_state(), target.export_state())


def test_cpu_adam_resume_preserves_execution_and_model_only_skips_optimizer():
    cfg = config()
    source = build_runtime(cfg, training=True)
    advance(source)
    state = source.export_state()
    target = build_runtime(cfg, training=True)
    target.import_state(state)
    assert_equal(target.export_state(), state)
    assert target.raw_optimizer.param_groups[0]['fused'] is False
    assert all(s['step'].device.type == 'cpu' for s in target.raw_optimizer.state.values())
    weights = build_runtime(cfg, training=True)
    weights.import_state(state, model_only=True)
    assert_equal(weights.export_state().model, state.model)
    assert not weights.raw_optimizer.state
