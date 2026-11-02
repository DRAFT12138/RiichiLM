"""Shared pointwise fusion policy, failover and real compiled GPU equivalence."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed, Unsupported

from mahjong.models.shared_kernels import SharedTrainingKernels, _unsupported, rms_norm, _merged_projections
from mahjong.models.qwen3_shared import MIN_COMPILE_UPDATES
from mahjong.app.assembly import build_runtime
from mahjong.contracts import MahjongError
from mahjong.data.mahjong.shared_policy import collate_shared_group
from tests.unit.models.test_qwen3_shared import config, samples
from tests.unit.training.test_runtime import assert_equal


@pytest.mark.parametrize('device,hip,arch,enabled', [
    ('cpu', 'rocm', 'gfx1100', False),
    ('cuda:0', None, '', False),
    ('cuda:0', 'rocm', 'gfx1100', True),
    ('cuda:0', 'rocm', 'gfx1100:xnack-', True),
    ('cuda:0', 'rocm', 'gfx942', False),
    ('cuda:0', 'rocm', 'gfx1200', False),
])
def test_rdna3_profile_is_local_to_validated_architecture(monkeypatch, device, hip, arch, enabled):
    monkeypatch.setattr(torch.version, 'hip', hip)
    queried = []
    def properties(device):
        queried.append(device)
        return SimpleNamespace(gcnArchName=arch)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', properties)
    options = []
    def compile(function, **kwargs):
        options.append(kwargs['options'])
        return function
    monkeypatch.setattr(torch, 'compile', compile)
    kernels = SharedTrainingKernels(device)
    value = torch.ones(1)
    assert kernels.call('layer', lambda x: x, value) is value
    assert kernels.call('norm', lambda x: x, value) is value
    assert bool(options[0].get('reorder_for_locality_in_training')) is enabled
    assert 'reorder_for_locality_in_training' not in options[1]
    assert bool(queried) is (device != 'cpu' and hip is not None)


def test_native_gqa_matches_repeated_heads_and_all_input_gradients():
    from torch.nn.functional import scaled_dot_product_attention as attention
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        q = torch.randn(1, 12, 17, 64, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 4, 17, 64, dtype=torch.float64, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        expected = attention(q, repeat_kv(k, 3), repeat_kv(v, 3), is_causal=True)
        actual = attention(q, k, v, is_causal=True, enable_gqa=True)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        adjoint = torch.randn_like(expected)
        reference = torch.autograd.grad(expected, (q, k, v), adjoint)
        gradients = torch.autograd.grad(actual, (q, k, v), adjoint)
        for a, b in zip(gradients, reference):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_gfx1100_native_gqa_compiled_visibility_and_finite_gradients():
    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1100':
        pytest.skip('gfx1100-specific execution')
    torch.cuda.set_per_process_memory_fraction(.4)
    settings = config('cuda:0', 'bf16-mixed')
    settings['model'].update(num_hidden_layers=1, hidden_size=128, intermediate_size=256,
        num_attention_heads=12, num_key_value_heads=4, head_dim=64)
    runtime = build_runtime(settings, training=True)
    runtime.configure_training_execution(100000)
    model = runtime.raw_model
    batch = collate_shared_group(samples())
    payload = model.prepare_batch(batch, runtime.device).payload
    descriptor, keys = model.descriptor(), list(model.state_dict())
    def logits(p):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model.causal_lm.lm_head(model._hidden(p, fused=True)[0, p['query_nodes']])
    base = logits(payload)
    assert model._training_kernels._gfx1100
    future = {**payload, 'input_ids': payload['input_ids'].clone()}
    future['input_ids'][0, 6] = 25
    torch.testing.assert_close(logits(future)[:5], base[:5], rtol=0, atol=0)
    other = {**payload, 'input_ids': payload['input_ids'].clone()}
    other['input_ids'][0, payload['trunk_length']+1] = 55
    torch.testing.assert_close(logits(other)[3:], base[3:], rtol=0, atol=0)
    del base
    runtime.zero_grad()
    result = runtime.compute(batch, phase='train')
    runtime.backward(result.loss)
    runtime.check_gradients()
    runtime.step()
    assert all(p.grad is not None and p.grad.dtype == torch.float32
               and p.grad.is_contiguous() and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert model.descriptor() == descriptor and list(model.state_dict()) == keys
    assert model.causal_lm.lm_head.weight is model.causal_lm.model.embed_tokens.weight
    assert model._training_kernels.fallbacks == 0
    next(model.parameters()).grad.flatten()[0] = float('nan')
    with pytest.raises(MahjongError, match='NONFINITE_GRADIENT'):
        runtime.check_gradients()


@pytest.mark.parametrize('error,expected', [
    (ImportError('compiler missing'),True),(NotImplementedError('unsupported'),True),
    (Unsupported('unsupported'),True),(RuntimeError('kernel failed'),False),
    (torch.OutOfMemoryError('budget'),False),(ValueError('invalid'),False),
])
def test_only_explicit_unavailability_can_fallback(error,expected):
    assert _unsupported(error) is expected
    wrapped=BackendCompilerFailed(lambda:None,error,None)
    assert _unsupported(wrapped) is expected
    if not expected:
        outer=Unsupported('wrapped unsupported')
        outer.__cause__=wrapped
        assert not _unsupported(outer)


def test_fallback_is_reported_once_and_never_retries_after_failure(monkeypatch):
    calls=[]
    def unavailable(*a,**kw):
        calls.append(kw);raise ImportError('backend missing')
    monkeypatch.setattr(torch,'compile',unavailable)
    kernels=SharedTrainingKernels()
    value=torch.randn(2,4);weight=torch.ones(4)
    with pytest.warns(RuntimeWarning,match='using eager'):
        actual=kernels.call('norm',rms_norm,value,weight,1e-6)
    assert torch.equal(actual,rms_norm(value,weight,1e-6))
    kernels.call('norm',rms_norm,value,weight,1e-6)
    assert len(calls)==1 and kernels.fallbacks==1 and kernels.disabled


def test_backend_runtime_or_oom_propagates(monkeypatch):
    for error in (RuntimeError('kernel failed'),torch.OutOfMemoryError('budget')):
        wrapped=BackendCompilerFailed(lambda:None,error,None)
        def fail(*a,**kw):raise wrapped
        monkeypatch.setattr(torch,'compile',fail)
        kernels=SharedTrainingKernels()
        with pytest.raises(BackendCompilerFailed):
            kernels.call('probe',lambda x:x,torch.ones(1))
        assert not kernels.disabled and kernels.fallbacks==0


def test_cpu_and_short_budget_keep_eager_and_hooks_are_preserved(monkeypatch):
    runtime=build_runtime(config(),training=True)
    model=runtime.raw_model
    runtime.configure_training_execution(MIN_COMPILE_UPDATES-1)
    assert not model._compile_training
    runtime.configure_training_execution(MIN_COMPILE_UPDATES)
    assert model._compile_training
    before=model.descriptor();keys=list(model.state_dict())
    runtime.compute(collate_shared_group(samples()),phase='train')
    assert model._training_kernels is None
    assert model.descriptor()==before and list(model.state_dict())==keys
    norm=model.causal_lm.model.layers[0].input_layernorm
    seen=[]
    hook=norm.register_forward_hook(lambda *a:seen.append(True))
    kernels=SharedTrainingKernels()
    kernels.norm(norm,torch.zeros(1,3,32))
    hook.remove()
    assert seen==[True] and not kernels.compiled
    seen.clear()
    hook=torch.nn.modules.module.register_module_forward_hook(lambda *a:seen.append(True))
    try:
        kernels.norm(norm,torch.zeros(1,3,32))
    finally:
        hook.remove()
    assert seen and not kernels.compiled


def gpu_runtime():
    torch.cuda.set_per_process_memory_fraction(.4)
    runtime=build_runtime(config('cuda:0','bf16-mixed'),training=True)
    runtime.configure_training_execution(100000)
    return runtime


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU unavailable')
def test_actual_compiled_hidden_isolation_gradients_identity_and_gates():
    runtime=gpu_runtime();model=runtime.raw_model
    batch=collate_shared_group(samples())
    prepared=model.prepare_batch(batch,torch.device('cuda:0')).payload
    keys=list(model.state_dict());descriptor=model.descriptor()
    def logits(payload,compiled):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model.causal_lm.lm_head(model._hidden(payload,fused=compiled)[0,payload['query_nodes']])
    eager=logits(prepared,False)
    base=logits(prepared,True)
    torch.testing.assert_close(base,eager,atol=5e-4,rtol=5e-3)
    future={**prepared,'input_ids':prepared['input_ids'].clone()}
    future['input_ids'][0,6]=25
    actual=logits(future,True)
    torch.testing.assert_close(actual[:5],base[:5],rtol=0,atol=0)
    other={**prepared,'input_ids':prepared['input_ids'].clone()}
    other['input_ids'][0,prepared['trunk_length']+1]=55
    actual=logits(other,True)
    torch.testing.assert_close(actual[3:],base[3:],rtol=0,atol=0)
    assert list(model.state_dict())==keys and model.descriptor()==descriptor
    assert model.causal_lm.lm_head.weight is model.causal_lm.model.embed_tokens.weight
    runtime.zero_grad();result=runtime.compute(batch,phase='train')
    assert result.metrics['ntp_loss'].denominator==9
    runtime.backward(result.loss);runtime.check_gradients();runtime.step()
    assert model._training_kernels and not model._training_kernels.disabled
    assert set(model._training_kernels.compiled)=={'layer'}
    parameter=next(model.parameters())
    parameter.grad.flatten()[0]=float('nan')
    with pytest.raises(MahjongError,match='NONFINITE_GRADIENT'):
        runtime.check_gradients()
    runtime.zero_grad()
    for invalid in (float('nan'),float('inf')):
        hook=model.causal_lm.lm_head.register_forward_hook(
            lambda module,args,out:torch.full_like(out,invalid))
        try:
            with pytest.raises(MahjongError,match='NONFINITE_LOSS'):
                runtime.compute(batch,phase='train')
        finally:hook.remove()
    # Phase, model mode, autocast precision and grad gating all remain eager.
    calls=[];original=model._hidden
    def traced(payload,*,fused=False):
        calls.append(fused);return original(payload,fused=fused)
    model._hidden=traced
    runtime.compute(batch,phase='eval');assert calls[-1] is False
    model.eval();runtime.compute(batch,phase='train');assert calls[-1] is False
    model.train()
    # Model-level gate: a training Runtime correctly rejects a no-grad loss.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        model(batch,phase='train')
    assert calls[-1] is False
    # No GPU FP32 training is needed to test the FP32 gate: autocast disabled,
    # stop before the actual hidden computation.
    calls.clear()
    class GateReached(Exception):pass
    def stop(payload,*,fused=False):
        calls.append(fused);raise GateReached
    model._hidden=stop
    with pytest.raises(GateReached):model(batch,phase='train')
    assert calls==[False]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU unavailable')
def test_compiled_regions_accept_decision_counts_and_new_lengths():
    runtime=gpu_runtime()
    for cut in (6,7):
        row=copy.deepcopy(samples()[-1])
        row['cut_length']=cut  # Same physical length, a different trunk boundary.
        runtime.zero_grad()
        result=runtime.compute(collate_shared_group([row]),phase='train')
        runtime.backward(result.loss);runtime.check_gradients();runtime.step()
    for size in (1,2,3,8):
        group=[]
        for index in range(size):
            row=copy.deepcopy(samples()[-1])
            row['sample_id']=f'count-{size}/{index}'
            # Vary physical alignment without changing legal action targets.
            row['input_ids']+= [50]*(index%2)
            group.append(row)
        runtime.zero_grad()
        result=runtime.compute(collate_shared_group(group),phase='train')
        assert result.metrics['ntp_loss'].denominator==4*size
        runtime.backward(result.loss);runtime.check_gradients();runtime.step()
    assert runtime.raw_model._training_kernels.fallbacks==0


def test_layer_eligibility_rejects_custom_forward_subclasses_and_dynamic_hooks(monkeypatch):
    runtime=build_runtime(config(),training=True)
    model=runtime.raw_model; layer=model.causal_lm.model.layers[0]
    kernels=SharedTrainingKernels()
    assert kernels.supports_layer(layer)
    projection=layer.self_attn.q_proj
    class CustomLinear(torch.nn.Linear):pass
    projection.__class__=CustomLinear
    assert not kernels.supports_layer(layer)
    projection.__class__=torch.nn.Linear
    projection.forward=copy.deepcopy(projection).forward
    assert not kernels.supports_layer(layer)  # Same function, different bound weights.
    del projection.forward
    original=projection.forward; seen=[]
    projection.forward=lambda value:(seen.append('custom'),original(value))[1]
    assert not kernels.supports_layer(layer)
    # The fallback still invokes the actual replacement once.
    monkeypatch.setattr(torch,'compile',lambda function,**kwargs:function)
    prepared=model.prepare_batch(collate_shared_group(samples()),runtime.device).payload
    model._hidden(prepared,fused=True)
    assert seen==['custom']
    del projection.forward
    assert kernels.supports_layer(layer)
    hook=projection.register_forward_hook(lambda *args:seen.append('hook'))
    assert not kernels.supports_layer(layer)
    model._hidden(prepared,fused=True)
    hook.remove()
    assert seen==['custom','hook'] and kernels.supports_layer(layer)
    original_class_forward=torch.nn.Linear.forward
    monkeypatch.setattr(torch.nn.Linear,'forward',lambda self,x:original_class_forward(self,x))
    assert not kernels.supports_layer(layer)


def test_regional_fallback_preserves_replaced_norm_and_mlp_forward(monkeypatch):
    runtime=build_runtime(config(),training=True)
    layer=runtime.raw_model.causal_lm.model.layers[0]
    kernels=SharedTrainingKernels();seen=[]
    monkeypatch.setattr(torch,'compile',lambda *args,**kwargs:pytest.fail('custom forward must run directly'))
    for module,value in ((layer.input_layernorm,torch.ones(1,3,32)),
                         (layer.mlp,torch.ones(1,3,32))):
        original=module.forward
        module.forward=lambda x,original=original:(seen.append(True),original(x))[1]
        actual=(kernels.norm(module,value) if module is layer.input_layernorm
                else kernels.mlp(module,value))
        torch.testing.assert_close(actual,original(value))
    assert seen==[True,True]


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_nonfinite_shared_loss_is_rejected_before_metric_construction(invalid):
    runtime=build_runtime(config(),training=True)
    hook=runtime.raw_model.causal_lm.lm_head.register_forward_hook(
        lambda module,args,out:torch.full_like(out,invalid))
    try:
        with pytest.raises(MahjongError,match='NONFINITE_LOSS'):
            runtime.compute(collate_shared_group(samples()),phase='train')
    finally:hook.remove()


@pytest.mark.parametrize('widths',[(12,4,4),(16,16)])
@pytest.mark.parametrize('bias',['none','all','mixed'])
def test_merged_projection_fp64_outputs_all_gradients_and_parameter_identity(widths,bias):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        projections=tuple(torch.nn.Linear(8,width,bias=(bias=='all' or bias=='mixed' and i==0)).double()
                          for i,width in enumerate(widths))
        value=torch.randn(2,5,8,dtype=torch.float64,requires_grad=True)
        parameters=tuple(p for projection in projections for p in projection.parameters())
        ids=tuple(id(p) for p in parameters)
        expected=tuple(projection(value) for projection in projections)
        observed=[]
        hooks=([projection.register_forward_hook(lambda *args:observed.append(True)) for projection in projections]
               if bias!='none' else [])
        try:actual=_merged_projections(value,projections)
        finally:
            for hook in hooks:hook.remove()
        if bias!='none':assert len(observed)==len(projections)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        adjoints=tuple(torch.randn_like(output) for output in expected)
        gradients=[torch.autograd.grad(outputs,(value,*parameters),adjoints)
                   for outputs in (expected,actual)]
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        assert ids==tuple(id(p) for projection in projections for p in projection.parameters())


def test_inplace_activation_uses_original_mlp_before_merging_views(monkeypatch):
    from mahjong.models import shared_kernels
    runtime=build_runtime(config(),training=True);model=runtime.raw_model
    for layer in model.causal_lm.model.layers:layer.mlp.act_fn=torch.nn.SiLU(inplace=True)
    prepared=model.prepare_batch(collate_shared_group(samples()),runtime.device).payload
    expected=model._hidden(prepared,fused=False)
    groups=[];original=shared_kernels._merged_projections
    def tracked(value,projections):
        groups.append(len(projections));return original(value,projections)
    monkeypatch.setattr(shared_kernels,'_merged_projections',tracked)
    monkeypatch.setattr(torch,'compile',lambda function,**kwargs:function)
    actual=model._hidden(prepared,fused=True)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    assert groups==[3,3]  # QKV only; neither MLP was merged before inplace SiLU.
    actual.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
               if p is not model.causal_lm.lm_head.weight)
