"""Behavioral Qwen3 checks use tiny configurations and offline local files."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3ForCausalLM

from mahjong.app.assembly import build_model, source_fingerprint
from mahjong.app.config import parse_model, parse_new
from mahjong.evaluation.aggregation import MetricAccumulator
from mahjong.contracts import Batch, MahjongError, Model, derive_seed, validate_step_result
from mahjong.models.qwen3 import INPUT_SCHEMA, Qwen3Adapter, build_qwen3, load_qwen3_local

SMALL = ['model.num_hidden_layers=2', 'model.hidden_size=32',
         'model.intermediate_size=64', 'model.num_attention_heads=4',
         'model.num_key_value_heads=2', 'model.head_dim=8',
         'model.vocab_size=32', 'model.max_position_embeddings=32']


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return parse_model(SMALL)


def batch(ids, labels=None, mask=None):
    payload = {'input_ids': ids, 'labels': ids.clone() if labels is None else labels}
    if mask is not None:
        payload['attention_mask'] = mask
    return Batch(payload, tuple(f's{i}' for i in range(len(ids))), len(ids), INPUT_SCHEMA)


def tokens():
    return torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(7))


def test_seed_rng_fp32_and_offline_native_interface(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('random construction must not load pretrained weights')
    monkeypatch.setattr(Qwen3ForCausalLM, 'from_pretrained', forbidden)
    before = torch.get_rng_state().clone()
    first = build_model(config)
    second = build_model(config)
    assert torch.equal(before, torch.get_rng_state())
    assert isinstance(first, Model) and isinstance(first.causal_lm, Qwen3ForCausalLM)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    different = deepcopy(config); different['seed'] += 1
    assert not torch.equal(first.causal_lm.lm_head.weight, build_model(different).causal_lm.lm_head.weight)
    torch.set_default_dtype(torch.float64)
    try:
        third = build_model(config)
        assert torch.get_default_dtype() == torch.float64
        assert {p.dtype for p in third.parameters()} == {torch.float32}
    finally:
        torch.set_default_dtype(torch.float32)
    native = first.causal_lm
    assert native.config.use_cache is False
    assert native.config.rope_parameters['rope_theta'] == 1000000.
    assert native.lm_head.weight is native.model.embed_tokens.weight
    assert native(tokens()).loss is None
    output = native(tokens(), labels=tokens())
    assert output.logits.shape == (2, 6, 32)
    output.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters())
    assert derive_seed(config['seed'], 'model/init/v1') != config['seed']


@pytest.mark.parametrize('override,field', [
    ('model.hiden_size=32', 'hiden_size'), ('model.hidden_size=true', 'hidden_size'),
    ('model.head_dim=7', 'head_dim'), ('model.num_key_value_heads=3', 'num_key_value_heads'),
    ('model.num_hidden_layers=0', 'num_hidden_layers'), ('model.pad_token_id=32', 'pad_token_id'),
    ('model.rope_theta=0', 'rope_theta'), ('model.tie_word_embeddings="false"', 'tie_word_embeddings'),
    ('model.attn_implementation=flash_attention_2', 'attn_implementation'),
    ('seed=true', 'seed'), ('+model.foo=1', 'foo'), ('seed=1,2', 'seed'),
])
def test_invalid_config(config, override, field):
    with pytest.raises(MahjongError, match=field):
        parse_model([item for item in SMALL if item.split('=')[0] != override.split('=')[0]] + [override])


def test_causality_mask_and_manual_shift_loss(config):
    native = build_model(config).causal_lm.eval()
    ids = tokens()
    changed = ids.clone(); changed[:, 4:] = (changed[:, 4:] + 1) % 32
    with torch.no_grad():
        baseline = native(ids).logits
        torch.testing.assert_close(native(changed).logits[:, :4], baseline[:, :4], rtol=0, atol=1e-6)
        mask = torch.ones_like(ids); mask[:, 1] = 0
        changed = ids.clone(); changed[:, 1] = (changed[:, 1] + 1) % 32
        reference = native(ids, attention_mask=mask).logits
        torch.testing.assert_close(native(changed, attention_mask=mask).logits[:, 2:],
                                   reference[:, 2:], rtol=0, atol=1e-6)
        labels = ids.clone(); labels[0, 3:] = -100; labels[1, 1] = -100
        output = native(ids, attention_mask=mask, labels=labels)
        manual = F.cross_entropy(output.logits[:, :-1].reshape(-1, 32),
                                 labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(output.loss, manual, rtol=1e-6, atol=1e-7)
        # Attention masking does not automatically remove a target from loss.
        assert int((labels[:, 1:] != -100).sum()) == 6


def test_adapter_weighted_tokens_empty_and_state(config):
    model = build_model(config)
    ids = tokens(); labels = ids.clone(); labels[0, 3:] = -100
    mask = torch.ones_like(ids); mask[0, 3:] = 0
    source = batch(ids, labels, mask)
    prepared = model.prepare_batch(source, torch.device('cpu'))
    assert (prepared.sample_ids, prepared.sample_count, prepared.input_schema) == (
        source.sample_ids, source.sample_count, source.input_schema)
    results = [model(batch(ids[i:i+1], labels[i:i+1], mask[i:i+1]), phase='train') for i in range(2)]
    joined = model(prepared, phase='train')
    assert joined.metrics['loss'].denominator == 7
    accumulator = MetricAccumulator()
    for result in results:
        accumulator.add(result.metrics)
    combined = accumulator.result()['loss']
    assert combined.denominator == 7
    assert combined.value == pytest.approx(joined.loss.item(), rel=1e-6)
    numerator = sum(result.metrics['loss'].numerator for result in results)
    assert joined.metrics['loss'].numerator == pytest.approx(numerator, rel=1e-6)
    for phase in ('train', 'eval'):
        empty = model(batch(ids, torch.full_like(ids, -100)), phase=phase)
        validate_step_result(empty, phase=phase)
        assert empty.loss is None and empty.metrics['loss'].denominator == 0
        assert (empty.skip_reason is not None) == (phase == 'train')
    # Only position zero is labeled; no shifted target is present.
    labels.fill_(-100); labels[:, 0] = ids[:, 0]
    assert model(batch(ids, labels), phase='train').skip_reason
    assert model.auxiliary_state_dict() == {}
    model.reset_auxiliary_rng(10); model.load_auxiliary_state_dict({})
    with pytest.raises(MahjongError):
        model.load_auxiliary_state_dict({'unexpected': 1})


@pytest.mark.parametrize('kind', ['schema', 'missing', 'dtype', 'range', 'shape', 'mask', 'labels'])
def test_payload_errors(config, kind):
    model = build_model(config); ids = tokens(); b = batch(ids)
    payload = dict(b.payload)
    schema = INPUT_SCHEMA
    if kind == 'schema': schema = 'synthetic-regression/v1'
    if kind == 'missing': del payload['labels']
    if kind == 'dtype': payload['input_ids'] = ids.float()
    if kind == 'range': payload['input_ids'] = torch.full_like(ids, 32)
    if kind == 'shape': payload['labels'] = ids[:, :-1]
    if kind == 'mask': payload['attention_mask'] = torch.full_like(ids, 2)
    if kind == 'labels': payload['labels'] = torch.full_like(ids, -2)
    with pytest.raises(MahjongError, match='Qwen3 payload'):
        model.prepare_batch(Batch(payload, b.sample_ids, b.sample_count, schema), torch.device('cpu'))


def test_local_hf_roundtrip_and_semantic_fingerprint(config, tmp_path):
    model = build_model(config); model.eval()
    fingerprint = model.descriptor().config_fingerprint
    expected = model.causal_lm(tokens()).logits.detach()
    model.causal_lm.save_pretrained(tmp_path)
    loaded = Qwen3Adapter(load_qwen3_local(tmp_path)).eval()
    assert loaded.descriptor().config_fingerprint == fingerprint
    assert loaded.causal_lm.lm_head.weight is loaded.causal_lm.model.embed_tokens.weight
    torch.testing.assert_close(loaded.causal_lm(tokens()).logits, expected, rtol=0, atol=0)
    loaded.causal_lm.config.rope_parameters['rope_theta'] *= 2
    assert loaded.descriptor().config_fingerprint != fingerprint
    with pytest.raises(MahjongError, match='local directory'):
        load_qwen3_local('Qwen/remote-not-a-local-directory')


def test_phase1_entry_and_fingerprint(tmp_path):
    assert parse_new([], tmp_path)['model']['name'] == 'mock_mlp'
    with pytest.raises(MahjongError):
        parse_new(['model=qwen3'], tmp_path)
    _, paths = source_fingerprint()
    assert {'requirements-rocm.txt', 'requirements-cuda.txt', 'src/mahjong/models/qwen3.py',
            'configs/model/qwen3.yaml', 'configs/model_only.yaml'} <= paths.keys()


@pytest.mark.parametrize('preimport_hf', [False, True])
def test_forced_hf_offline_overrides_environment_and_preimport(preimport_hf):
    import os
    from pathlib import Path
    import subprocess
    import sys
    code = '''
import os, sys
if sys.argv[1] == 'preimport':
    import transformers
    from huggingface_hub.utils import get_session
    get_session()  # Existing client must also obey the changed offline policy.
from mahjong.models.qwen3 import build_qwen3
from huggingface_hub import constants, is_offline_mode
from huggingface_hub.errors import OfflineModeIsEnabled
from huggingface_hub.utils import get_session
import httpx2
assert is_offline_mode()
assert constants.HF_HUB_DISABLE_TELEMETRY
assert all(os.environ[key] == '1' for key in (
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'))
# Any attempt to reach the transport is a test failure, not an actual request.
def forbidden(*args, **kwargs):
    raise AssertionError('HF reached HTTP transport despite forced offline mode')
httpx2.HTTPTransport.handle_request = forbidden
try:
    get_session().get('https://huggingface.co/never-requested')
except OfflineModeIsEnabled:
    pass
else:
    raise AssertionError('HF request did not fail offline')
print('HF request blocked before HTTP transport')
'''
    environment = dict(os.environ, HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0',
                       HF_HUB_DISABLE_TELEMETRY='0')
    result = subprocess.run([sys.executable, '-c', code, 'preimport' if preimport_hf else 'first'],
                            cwd=Path(__file__).resolve().parents[3], env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'blocked before HTTP transport' in result.stdout
