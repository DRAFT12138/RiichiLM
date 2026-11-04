"""Seat-owned causal prefix KV; never saved in a model or training snapshot."""

from contextlib import contextmanager
from itertools import count

import torch

_CLOCK = count()
GPU_PREFILL_CHUNK_TOKENS = 2048


class PrefixCache:
    def __init__(self):
        self.cache = None
        self.tokens = ()
        self.owner = None
        self.last_used = next(_CLOCK)
        self.hits = self.reused_tokens = self.prefill_tokens = self.evictions = 0

    def clear(self, *, evicted=False):
        if evicted and self.cache is not None:
            self.evictions += 1
        self.cache = None
        self.tokens = ()
        self.owner = None

    @staticmethod
    def identity(policy, request):
        from mahjong.policy.execution import policy_precision
        window = request.key.window
        return (id(policy.model), policy.binding_digest, window.env_instance_id,
                window.game_id, window.round_id, request.key.viewer,
                policy_precision(policy.model), policy.lm.config._attn_implementation,
                policy.model.training, getattr(policy.lm.forward, '__func__', policy.lm.forward),
                tuple((id(p), p._version, p.device, p.dtype) for p in policy.model.parameters()))

    def take(self, policy, request):
        owner = self.identity(policy, request)
        cache, tokens, previous_owner = self.cache, self.tokens, self.owner
        self.clear()
        common = 0
        if cache is not None and previous_owner == owner:
            # Recompute the last token even for an identical request: its logits
            # are required and must reflect this call's legal candidate set.
            for old, new in zip(tokens, request.prefix_ids[:-1]):
                if old != new:
                    break
                common += 1
        if not common:
            cache = None
        else:
            remove = cache.get_seq_length() - common
            if remove:
                cache.crop(-remove)
            self.hits += 1
            self.reused_tokens += common
        self.prefill_tokens += len(request.prefix_ids) - common
        return cache, common

    def put(self, policy, request, cache):
        if cache is None:
            return
        remove = cache.get_seq_length() - len(request.prefix_ids)
        if remove < 0:
            return
        if remove:
            cache.crop(-remove)
        self.cache = cache
        self.tokens = request.prefix_ids
        self.owner = self.identity(policy, request)
        self.last_used = next(_CLOCK)

    def snapshot(self):
        return dict(hits=self.hits, reused_tokens=self.reused_tokens,
                    prefill_tokens=self.prefill_tokens, evictions=self.evictions)


def _aligned_sdpa(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
    """Public lower-right causal bias, with casting before bias dispatch."""
    from torch.nn.attention.bias import causal_lower_right
    dtype = torch.get_autocast_dtype('cuda')
    query, key, value = (x.to(dtype) for x in (query, key, value))
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=causal_lower_right(query.shape[-2], key.shape[-2]),
        dropout_p=dropout, scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def incremental_attention(lm, enabled):
    if not enabled:
        yield
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    # Register a separate interface; never replace a global PyTorch/HF function.
    name = 'riichilm_prefix_sdpa'
    if name not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(name, _aligned_sdpa)
    config = lm.config
    original = config._attn_implementation
    try:
        config._attn_implementation = name
        yield
    finally:
        config._attn_implementation = original
