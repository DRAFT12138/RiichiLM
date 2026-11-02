"""Portable inference weights, independent of training run metadata."""

import json
from pathlib import Path
import re

import torch

from mahjong.contracts import MahjongError
from mahjong.models.qwen3 import SETTINGS_FIELDS, Qwen3PolicyAdapter, build_qwen3
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA


def read_weights(path):
    """Accept a training Snapshot, an inference export, or a plain state_dict."""
    path = Path(path)
    value = torch.load(path, map_location='cpu', weights_only=True)
    config = {}
    if isinstance(value, dict) and 'runtime' in value:
        config = value.get('config', {}).get('model', {})
        state = value['runtime']['model']
    elif isinstance(value, dict) and 'state_dict' in value:
        config = value.get('model_config', value.get('config', {}))
        state = value['state_dict']
    else:
        state = value
    if not isinstance(state, dict) or not state:
        raise ValueError('expected a model state_dict')
    normalized = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError('state_dict must contain named tensors')
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if not key.startswith('causal_lm.'):
            key = 'causal_lm.' + key
        normalized[key] = tensor
    # Plain state_dicts carry no architecture. Infer dimensions from Qwen tensors;
    # a same-stem JSON sidecar can supply context length, RoPE, or input schema.
    prefix = 'causal_lm.model.'
    embed = normalized[prefix + 'embed_tokens.weight']
    head = normalized.get('causal_lm.lm_head.weight')
    head_dim = normalized[prefix + 'layers.0.self_attn.q_norm.weight'].numel()
    layers = [int(m[1]) for key in normalized
              if (m := re.match(r'causal_lm\.model\.layers\.(\d+)\.', key))]
    settings = dict(
        name='qwen3', input_schema=CONTEXT_SCHEMA,
        num_hidden_layers=max(layers) + 1, hidden_size=embed.shape[1],
        intermediate_size=normalized[prefix + 'layers.0.mlp.gate_proj.weight'].shape[0],
        num_attention_heads=normalized[prefix + 'layers.0.self_attn.q_proj.weight'].shape[0] // head_dim,
        num_key_value_heads=normalized[prefix + 'layers.0.self_attn.k_proj.weight'].shape[0] // head_dim,
        head_dim=head_dim, vocab_size=embed.shape[0], max_position_embeddings=8192,
        attn_implementation='sdpa',
        tie_word_embeddings=head is None or torch.equal(embed, head), rope_theta=1000000.0,
        use_cache=False, bos_token_id=None, eos_token_id=None, pad_token_id=None,
    )
    if isinstance(config, dict):
        config = config.get('model', config)
        settings.update({k: v for k, v in config.items() if k in SETTINGS_FIELDS})
    sidecar = path.with_suffix('.json')
    if sidecar.is_file():
        overrides = json.loads(sidecar.read_text())
        overrides = overrides.get('model', overrides)
        settings.update({k: v for k, v in overrides.items() if k in SETTINGS_FIELDS})
    if settings['tie_word_embeddings'] and 'causal_lm.lm_head.weight' not in normalized:
        normalized['causal_lm.lm_head.weight'] = embed
    return settings, normalized


def load_policy_weights(path, *, device='cpu'):
    """Load usable Qwen weights without provenance, hash or dataset checks."""
    if device not in ('cpu', 'cuda:0') or (device == 'cuda:0' and not torch.cuda.is_available()):
        raise MahjongError('DEVICE_UNAVAILABLE', f'inference device unavailable: {device}')
    try:
        settings, state = read_weights(path)
        model = Qwen3PolicyAdapter(build_qwen3(settings, init_seed=0),
                                   input_schema=settings['input_schema'])
        # PyTorch reports missing/incompatible tensors; never leave random layers
        # in a supposedly loaded policy. Training descriptors are not consulted.
        model.load_state_dict(state)
        source = dict(checkpoint_id=Path(path).name, kind='weights',
                      training_rules_status='custom weights; training provenance not required')
        return model.to(device).eval(), source
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise MahjongError('CHECKPOINT_INVALID', f'Cannot load custom weights: {exc}') from exc


def export_policy_weights(path, output):
    """Strip optimizer, RNG, data cursor and logs without changing tensor precision."""
    settings, state = read_weights(path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(dict(model_config=settings, state_dict=state), stream)
    return output
