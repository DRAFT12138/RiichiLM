from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mahjong.contracts import MahjongError
from mahjong.inference.policy import DecodeOptions
from mahjong.inference.prefix_cache import PrefixCache, _aligned_sdpa
from mahjong.tokenizer import Atom, HandMeldState
from tests.unit.inference.test_policy import inference, view


def test_append_only_prefix_preserves_the_entire_old_cache(monkeypatch):
    from transformers.cache_utils import DynamicCache
    cache = PrefixCache()
    monkeypatch.setattr(cache, 'identity', lambda policy, request: 'same-seat')
    kv = DynamicCache()
    kv.update(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), 0)
    cache.put(None, SimpleNamespace(prefix_ids=(1, 2, 3, 4)), kv)
    retained, common = cache.take(None, SimpleNamespace(prefix_ids=(1, 2, 3, 4, 5, 6)))
    assert retained is kv and common == 4
    assert retained.get_seq_length() == 4


def test_seat_prefix_reuse_matches_fresh_actions_probabilities_and_rng(inference):
    cache = PrefixCache()
    a, b = [torch.Generator().manual_seed(42) for _ in range(2)]
    forwarded = []
    original = inference.lm.forward

    def spy(**kwargs):
        forwarded.append(kwargs['input_ids'].shape[1])
        return original(**kwargs)

    inference.lm.forward = spy
    for revision in range(8):
        v = view(revision=revision)
        seq = replace(v.sequence, hand_state=HandMeldState(
            [Atom('TILE_1m')] * 12 + [Atom('TILE_2m' if revision % 2 else 'TILE_3m')], []))
        v = type(v)(v.key, seq, v.legal_moves, v.rules_digest, True)
        request = inference.prepare(v)
        options = DecodeOptions(strategy='categorical')
        reference = inference.generate(request, options=options, generator=a)
        forwarded.clear()
        actual = inference.generate(request, options=options, generator=b, prefix_cache=cache)
        assert actual.suffix_ids == reference.suffix_ids
        assert actual.token_logprobs == pytest.approx(reference.token_logprobs, abs=2e-6)
        assert torch.equal(a.get_state(), b.get_state())
        if revision:
            assert forwarded[0] < len(request.prefix_ids)
        assert cache.cache.get_seq_length() == len(request.prefix_ids)
    assert cache.hits == 7 and cache.reused_tokens > 0


@pytest.mark.parametrize('change', ['viewer', 'round', 'game', 'env', 'weights'])
def test_cache_invalidates_across_ownership_and_weight_changes(inference, change):
    cache = PrefixCache()
    v = view()
    inference.generate(inference.prepare(v), prefix_cache=cache)
    if change == 'viewer':
        key = replace(v.key, viewer=1)
    else:
        field = {'round': 'round_id', 'game': 'game_id', 'env': 'env_instance_id'}.get(change)
        if field:
            key = replace(v.key, window=replace(v.key.window, **{field: 1 if change == 'round' else 'other'}))
        else:
            with torch.no_grad():
                next(inference.model.parameters()).add_(0.01)
            key = v.key
    v = type(v)(key, v.sequence, v.legal_moves, v.rules_digest, True)
    inference.generate(inference.prepare(v), prefix_cache=cache)
    assert cache.hits == 0


def test_failed_and_cancelled_requests_discard_persistent_cache(inference, monkeypatch):
    cache = PrefixCache()
    request = inference.prepare(view())
    inference.generate(request, prefix_cache=cache)
    with pytest.raises(MahjongError, match='CANCELLED'):
        inference.generate(request, prefix_cache=cache, cancel=lambda: True)
    assert cache.cache is None
    inference.generate(request, prefix_cache=cache)
    monkeypatch.setattr(inference.lm, 'forward', lambda **kw: (_ for _ in ()).throw(RuntimeError('failure')))
    with pytest.raises(MahjongError, match='MODEL_EXECUTION_FAILED'):
        inference.generate(request, prefix_cache=cache)
    assert cache.cache is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_gpu_chunked_prefill_keeps_all_tokens_and_absolute_positions(inference, monkeypatch):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    from mahjong.inference.policy import PolicyInference
    from mahjong.inference import prefix_cache
    inference = PolicyInference(inference.model.to('cuda:0'), precision='bf16-mixed')
    request = inference.prepare(view())
    reference = inference.generate(request)
    monkeypatch.setattr(prefix_cache, 'GPU_PREFILL_CHUNK_TOKENS', 32)
    positions = []
    original = inference.lm.forward

    def spy(**kwargs):
        assert kwargs['input_ids'].shape[1] <= 32
        positions.extend(kwargs['position_ids'][0].tolist())
        return original(**kwargs)

    monkeypatch.setattr(inference.lm, 'forward', spy)
    cache = PrefixCache()
    actual = inference.generate(request, prefix_cache=cache)
    assert actual.suffix_ids == reference.suffix_ids
    assert actual.token_logprobs == pytest.approx(reference.token_logprobs, abs=.02)
    assert positions[:len(request.prefix_ids)] == list(range(len(request.prefix_ids)))
    assert cache.cache.get_seq_length() == len(request.prefix_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_cached_gpu_attention_is_lower_right_causal_and_finite():
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(42)
    q = torch.randn(1, 12, 5, 64, device='cuda')
    k = torch.randn(1, 4, 13, 64, device='cuda')
    v = torch.randn_like(k)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = _aligned_sdpa(SimpleNamespace(), q, k, v, None)[0]
        altered = v.clone()
        altered[:, :, -1] = 100
        changed = _aligned_sdpa(SimpleNamespace(), q, k, altered, None)[0]
    assert torch.equal(out[:, 0], changed[:, 0])
    assert torch.isfinite(out).all()
    mask = torch.arange(13)[None, :] <= torch.arange(5)[:, None] + 8
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.bfloat16().float().cpu(), k.bfloat16().float().cpu(), v.bfloat16().float().cpu(),
        attn_mask=mask, enable_gqa=True).transpose(1, 2)
    torch.testing.assert_close(out.float().cpu(), reference, atol=.015, rtol=.015)
