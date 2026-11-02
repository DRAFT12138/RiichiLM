"""Portable weights survive missing training provenance and preserve predictions."""

import pytest
import torch

from mahjong.app.assembly import build_model
from mahjong.app.config import parse_model
from mahjong.contracts import MahjongError
from mahjong.inference.policy import PolicyInference
from mahjong.inference.weights import export_policy_weights
from mahjong.policy.execution import policy_execution, policy_precision


@pytest.fixture
def weights(tmp_path):
    torch.set_num_threads(1)
    config = parse_model([
        'model.input_schema=majang-policy-ntp/v2',
        'model.num_hidden_layers=1', 'model.hidden_size=16',
        'model.intermediate_size=32', 'model.num_attention_heads=2',
        'model.num_key_value_heads=1', 'model.head_dim=8',
        'model.max_position_embeddings=8192',
    ])
    model = build_model(config)
    path = tmp_path / 'server.pt'
    torch.save(dict(config=config, runtime=dict(model=dict(model.state_dict()),
               optimizer={'unused': torch.ones(100000)}, auxiliary={}),
               descriptors=[{'config_fingerprint': 'incompatible-server-version'}],
               rng={}, progress={}, identity={}), path)
    return path, model, config


def test_export_and_load_without_training_metadata_preserves_weights_and_logits(weights, tmp_path):
    original, model, _ = weights
    output = export_policy_weights(original, tmp_path / 'slim.pt')
    payload = torch.load(output, weights_only=True)
    assert set(payload) == {'model_config', 'state_dict'}
    assert output.stat().st_size < original.stat().st_size / 2
    a = PolicyInference.from_weights(original)
    b = PolicyInference.from_weights(output)
    for key, expected in model.state_dict().items():
        assert torch.equal(expected, a.model.state_dict()[key])
        assert torch.equal(expected, b.model.state_dict()[key])
    ids = torch.tensor([[1, 3, 5, 8]])
    with torch.no_grad():
        assert torch.equal(a.lm(ids).logits, b.lm(ids).logits)
    with pytest.raises(FileExistsError):
        export_policy_weights(original, output)


def test_plain_hf_state_dict_and_optional_config(weights, tmp_path):
    import json
    _, model, _ = weights
    path = tmp_path / 'bare.pth'
    state = {k.removeprefix('causal_lm.'): v for k, v in model.state_dict().items()}
    state.pop('lm_head.weight')
    torch.save(state, path)
    path.with_suffix('.json').write_text(json.dumps({'max_position_embeddings': 256}))
    inference = PolicyInference.from_weights(path)
    assert inference.lm.config.max_position_embeddings == 256
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, inference.model.state_dict()[key])


def test_missing_model_layers_are_not_silently_randomized(weights, tmp_path):
    path, _, _ = weights
    payload = torch.load(path, weights_only=True)
    payload['runtime']['model'].pop('causal_lm.model.layers.0.mlp.down_proj.weight')
    bad = tmp_path / 'bad.pt'
    torch.save(payload, bad)
    with pytest.raises(MahjongError, match='Missing key'):
        PolicyInference.from_weights(bad)


def test_explicit_fp32_preserves_default_binding_and_parameters(weights):
    path, model, _ = weights
    default = PolicyInference.from_weights(path)
    explicit = PolicyInference.from_weights(path, precision='32-true')
    assert default.binding_digest == explicit.binding_digest
    assert policy_precision(explicit.model) == '32-true'
    for key, expected in model.state_dict().items():
        assert torch.equal(expected, explicit.model.state_dict()[key])


@pytest.mark.parametrize('precision,code', [('bf16-mixed', 'DEVICE_UNAVAILABLE'),
                                          ('16-mixed', 'CONTRACT_MISMATCH')])
def test_unsupported_execution_precision_is_explicit(weights, precision, code):
    with pytest.raises(MahjongError) as exc:
        PolicyInference.from_weights(weights[0], precision=precision)
    assert exc.value.code == code


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_bf16_long_gqa_prefill_uses_flash_and_releases_cache():
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 GPU unavailable')
    from torch.profiler import profile, ProfilerActivity

    # Production head layout: FP32 + native GQA falls back to quadratic math
    # attention. One layer is sufficient to exercise that dispatch at 8k.
    config = parse_model([
        'model.num_hidden_layers=1', 'model.hidden_size=768',
        'model.intermediate_size=2048', 'model.num_attention_heads=12',
        'model.num_key_value_heads=4', 'model.head_dim=64',
        'model.max_position_embeddings=8192',
    ])
    policy = PolicyInference(build_model(config).to('cuda:0'), precision='bf16-mixed')
    assert all(p.dtype == torch.float32 for p in policy.model.parameters())

    def prefill_and_decode(length):
        ids = torch.zeros((1, length), dtype=torch.int64, device='cuda:0')
        with torch.inference_mode(), policy_execution(policy.model):
            out = policy.lm(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=True, logits_to_keep=1)
            assert out.logits.dtype == torch.bfloat16
            assert torch.isfinite(out.logits).all()
            assert out.past_key_values.get_seq_length() == length
            decoded = policy.lm(input_ids=ids[:, :1], past_key_values=out.past_key_values,
                                use_cache=True, logits_to_keep=1)
            assert torch.isfinite(decoded.logits).all()
            assert decoded.past_key_values.get_seq_length() == length + 1

    # Initialize the GPU BLAS workspace before measuring retained tensors.
    prefill_and_decode(256)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        for length in (256, 2048, 8191, 256, 8191):
            prefill_and_decode(length)
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() - baseline < 8 * 2**20
    kernels = {event.key for event in trace.key_averages()}
    assert 'aten::_scaled_dot_product_flash_attention' in kernels
    assert 'aten::_scaled_dot_product_attention_math' not in kernels
    assert torch.cuda.max_memory_allocated() - baseline < 512 * 2**20
