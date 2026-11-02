"""Portable weights survive missing training provenance and preserve predictions."""

import pytest
import torch

from mahjong.app.assembly import build_model
from mahjong.app.config import parse_model
from mahjong.contracts import MahjongError
from mahjong.inference.policy import PolicyInference
from mahjong.inference.weights import export_policy_weights


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
