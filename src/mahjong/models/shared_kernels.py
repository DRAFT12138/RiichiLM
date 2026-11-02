"""Lazy shared-layer/pointwise fusion; no parameters or persistent state."""
import warnings

import torch
from torch.nn import functional as F
from transformers.activations import SiLUActivation
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP, Qwen3RMSNorm,
    apply_rotary_pos_emb, repeat_kv,
)


_ORIGINAL_FORWARDS = {kind: kind.forward for kind in (
    Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP, Qwen3RMSNorm,
    torch.nn.Linear, torch.nn.SiLU, SiLUActivation,
)}


def _merged_projections(value, projections):
    """Same input/precision and original Parameters; caller owns hook checks."""
    if any(projection.bias is not None for projection in projections):
        return tuple(projection(value) for projection in projections)
    weights = tuple(projection.weight for projection in projections)
    return F.linear(value, torch.cat(weights, dim=0)).split(
        tuple(weight.shape[0] for weight in weights), dim=-1)


def shared_layer(layer, hidden, cos, sin, trunk_length, allowed, native_gqa=False):
    """One differentiable layer, retaining the original segmented SDPA graph."""
    residual = hidden
    normalized = layer.input_layernorm(hidden)
    attention = layer.self_attn
    shape = (*normalized.shape[:-1], -1, attention.head_dim)
    q, k, v = _merged_projections(normalized,
        (attention.q_proj, attention.k_proj, attention.v_proj))
    q, k = q.view(shape), k.view(shape)
    q = attention.q_norm(q).transpose(1, 2)
    k = attention.k_norm(k).transpose(1, 2)
    v = v.view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if not native_gqa:
        k, v = (repeat_kv(x, attention.num_key_value_groups) for x in (k, v))
    common = F.scaled_dot_product_attention(q[:, :, :trunk_length],
        k[:, :, :trunk_length], v[:, :, :trunk_length], is_causal=True,
        dropout_p=0., scale=attention.scaling, enable_gqa=native_gqa)
    if native_gqa:
        # Flash consumes the original four KV heads. Only the masked branch
        # backend needs repeated heads; its backward still reaches original K/V.
        k, v = (repeat_kv(x, attention.num_key_value_groups) for x in (k, v))
    branches = F.scaled_dot_product_attention(q[:, :, trunk_length:], k, v,
        attn_mask=allowed[None, None], is_causal=False, dropout_p=0.,
        scale=attention.scaling)
    combined = torch.cat((common, branches), dim=2).transpose(1, 2).contiguous()
    combined = combined.reshape(*hidden.shape[:-1], -1)
    hidden = residual + attention.o_proj(combined)
    normalized = layer.post_attention_layernorm(hidden)
    mlp = layer.mlp
    if getattr(mlp.act_fn, 'inplace', False):
        # Inplace activation cannot modify one of several split views.
        return hidden + mlp(normalized)
    gate, up = _merged_projections(normalized, (mlp.gate_proj, mlp.up_proj))
    return hidden + mlp.down_proj(mlp.act_fn(gate) * up)


def rms_norm(value, weight, epsilon):
    dtype = value.dtype
    value = value.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    return weight * normalized.to(dtype)


def silu_mul(gate, up):
    return F.silu(gate) * up


def _unsupported(error):
    """Only explicit compiler unavailability is safe to retry before backward."""
    from torch._dynamo.exc import BackendCompilerFailed, Unsupported
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BackendCompilerFailed):
            pending.append(current.inner_exception)
        elif not isinstance(current, (Unsupported, ImportError, NotImplementedError)):
            return False
        # Compiler wrappers must not hide an underlying tensor/OOM failure.
        pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
    return True


def _has_hooks(module):
    global_hooks = torch.nn.modules.module
    return bool(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or module._backward_pre_hooks or global_hooks._global_forward_hooks
                or global_hooks._global_forward_pre_hooks or global_hooks._global_backward_hooks
                or global_hooks._global_backward_pre_hooks)


def _original_forward(module):
    original = _ORIGINAL_FORWARDS.get(type(module))
    forward = module.forward
    return (original is not None and getattr(forward, '__func__', None) is original
            and getattr(forward, '__self__', None) is module)


class SharedTrainingKernels:
    """Ordinary Python object: compiled callables never enter model.state_dict."""
    def __init__(self, device=None):
        self.compiled = {}
        self.disabled = False
        self.fallbacks = 0
        # Diagnostic only; never a frozen config or checkpoint field.
        self.layer_enabled = True
        self._gfx1100 = (device is not None and torch.device(device).type == 'cuda'
            and bool(torch.version.hip)
            and torch.cuda.get_device_properties(device).gcnArchName.split(':')[0] == 'gfx1100')

    def supports_layer(self, layer):
        if self.disabled or not self.layer_enabled:
            return False
        for module in layer.modules():
            if _has_hooks(module) or not _original_forward(module):
                return False
        return True

    def layer(self, layer, hidden, cos, sin, trunk_length, allowed):
        attention = layer.self_attn
        native_gqa = (self._gfx1100 and attention.head_dim == 64
            and attention.q_proj.out_features == 768
            and attention.k_proj.out_features == attention.v_proj.out_features == 256)
        return self.call('layer', shared_layer, layer, hidden, cos, sin,
                         trunk_length, allowed, native_gqa)

    def call(self, name, function, *args):
        if self.disabled:
            return function(*args)
        try:
            if name not in self.compiled:
                options = {'compile_threads': 2, 'emulate_precision_casts': True}
                if name == 'layer' and self._gfx1100:
                    options['reorder_for_locality_in_training'] = True
                self.compiled[name] = torch.compile(function, backend='inductor',
                    dynamic=True, fullgraph=True,
                    options=options)
            return self.compiled[name](*args)
        except Exception as error:
            if not _unsupported(error):
                raise
            self.disabled = True
            self.fallbacks += 1
            warnings.warn(f'Shared NTP compilation unavailable; using eager: {error}',
                          RuntimeWarning, stacklevel=2)
            return function(*args)

    def norm(self, module, value):
        if _has_hooks(module) or not _original_forward(module):
            return module(value)
        return self.call('norm', rms_norm, value, module.weight, module.variance_epsilon)

    def rotary(self, q, k, cos, sin):
        return self.call('rotary', apply_rotary_pos_emb, q, k, cos, sin)

    def mlp(self, module, value):
        if (_has_hooks(module) or _has_hooks(module.act_fn)
                or not _original_forward(module) or not _original_forward(module.act_fn)
                or not isinstance(module.act_fn, (torch.nn.SiLU, SiLUActivation))):
            return module(value)
        gate, up = module.gate_proj(value), module.up_proj(value)
        return module.down_proj(self.call('silu_mul', silu_mul, gate, up))
