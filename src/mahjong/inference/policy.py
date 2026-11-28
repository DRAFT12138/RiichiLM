"""B=1 constrained autoregression with FP32 parameters and policy precision."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
import threading
import time

import torch

from mahjong.contracts import MahjongError
from mahjong.policy.constraints import build_trie, deterministic_tail
from mahjong.tokenizer import Action, MahjongTokenizer, ProtocolError
from mahjong.tokenizer.codec import BINDING
from mahjong.tokenizer.policy_context import (
    CONTEXT_SCHEMA, LEGACY_SCHEMA, PolicyContextInput, PolicyContextCodec, codec_for_schema,
)
from .sampling import DEFAULT_INFERENCE_TEMPERATURE, DEFAULT_INFERENCE_TOP_P, validate_sampling_params

EVENT_ADAPTER = 'riichi-online-ir/1'
ACTION_ADAPTER = 'riichi-policy-actions/1'

_SHARED_LEGACY_CODEC = None
_SHARED_CONTEXT_CODEC = None
_SHARED_LIST_TOKEN_ID = None


def _get_shared_codecs():
    global _SHARED_LEGACY_CODEC, _SHARED_CONTEXT_CODEC, _SHARED_LIST_TOKEN_ID
    if _SHARED_LEGACY_CODEC is None:
        _SHARED_LEGACY_CODEC = MahjongTokenizer()
        _SHARED_CONTEXT_CODEC = PolicyContextCodec()
        _SHARED_LIST_TOKEN_ID = _SHARED_LEGACY_CODEC.token_id('LIST')
    return _SHARED_LEGACY_CODEC, _SHARED_CONTEXT_CODEC, _SHARED_LIST_TOKEN_ID


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fail(code, message, **details):
    raise MahjongError(code, message, 'inference', details)


@dataclass(frozen=True)
class WindowKey:
    env_instance_id: str
    game_id: str
    round_id: int
    window_id: int
    state_revision: int


@dataclass(frozen=True)
class DecisionKey:
    window: WindowKey
    viewer: int


@dataclass(frozen=True)
class LegalMove:
    action: Action
    execution_ref: str


@dataclass(frozen=True, init=False)
class DecisionView:
    """Store encoded immutable content; sequence reads return fresh codec values."""
    key: DecisionKey
    prefix_ids: tuple[int, ...]
    legal_moves: tuple[LegalMove, ...]
    rules_digest: str
    legal_set_complete: bool

    def __init__(self, key, sequence, legal_moves, rules_digest, legal_set_complete):
        legacy_codec, context_codec, _ = _get_shared_codecs()
        codec = context_codec if isinstance(sequence, PolicyContextInput) else legacy_codec
        object.__setattr__(self, 'key', key)
        object.__setattr__(self, 'prefix_ids', tuple(codec.encode_policy_prefix(sequence)))
        moves = tuple(sorted((LegalMove(codec.decode_action_suffix(codec._encode_action_suffix_tuple(m.action)),
                                m.execution_ref) for m in legal_moves),
                             key=lambda m: codec._encode_action_suffix_tuple(m.action)))
        object.__setattr__(self, 'legal_moves', moves)
        object.__setattr__(self, 'rules_digest', rules_digest)
        object.__setattr__(self, 'legal_set_complete', legal_set_complete)

    @classmethod
    def _from_encoded(cls, key, prefix_ids, legal_moves, rules_digest, legal_set_complete):
        obj = object.__new__(cls)
        object.__setattr__(obj, 'key', key)
        object.__setattr__(obj, 'prefix_ids', prefix_ids)
        object.__setattr__(obj, 'legal_moves', legal_moves)
        object.__setattr__(obj, 'rules_digest', rules_digest)
        object.__setattr__(obj, 'legal_set_complete', legal_set_complete)
        return obj

    @property
    def sequence(self):
        legacy_codec, context_codec, list_token_id = _get_shared_codecs()
        codec = context_codec if len(self.prefix_ids) > 3 and self.prefix_ids[3] == list_token_id else legacy_codec
        return codec.decode_policy_prefix(self.prefix_ids)


@dataclass(frozen=True)
class PreparedDecision:
    key: DecisionKey
    prefix_ids: tuple[int, ...]
    legal_suffixes: tuple[tuple[int, ...], ...]
    binding_digest: str
    legal_set_digest: str


@dataclass(frozen=True)
class DecodeOptions:
    strategy: str = 'greedy'
    use_kv_cache: bool = True
    skip_deterministic_tail: bool = True
    temperature: float = 1.0
    top_p: float = 1.0


def validate_decode_options(options: DecodeOptions, generator=None) -> None:
    """Validate DecodeOptions fields and categorical RNG requirements."""
    if (
        not isinstance(options, DecodeOptions)
        or options.strategy not in ('greedy', 'categorical')
        or type(options.use_kv_cache) is not bool
        or type(options.skip_deterministic_tail) is not bool
    ):
        fail('CONTRACT_MISMATCH', 'unsupported decode options')
    try:
        validate_sampling_params(options.temperature, options.top_p, phase='inference')
    except MahjongError as exc:
        fail('CONTRACT_MISMATCH', exc.message)
    if options.strategy == 'categorical' and (generator is None or generator.device.type != 'cpu'):
        fail('CONTRACT_MISMATCH', 'categorical sampling requires an explicit CPU generator')


def select_legal_token(
    allowed_logits: torch.Tensor,
    options: DecodeOptions,
    generator=None,
) -> tuple[int, float, float]:
    """Select one legal token index and return (pick, model_logprob, behavior_logprob)."""
    raw_log_probs = torch.log_softmax(allowed_logits.float(), dim=-1)
    if options.strategy == 'greedy' or float(options.temperature) == 0.0:
        pick = int(allowed_logits.argmax().item())
        return pick, float(raw_log_probs[pick].item()), 0.0

    temperature = float(options.temperature)
    top_p = float(options.top_p)
    if temperature == 1.0 and top_p == 1.0:
        probabilities = raw_log_probs.exp().cpu()
        pick = int(torch.multinomial(probabilities, 1, generator=generator).item())
        model_lp = float(raw_log_probs[pick].item())
        return pick, model_lp, model_lp

    scaled_logits = allowed_logits.float() / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1).cpu()
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumulative - sorted_probs) < top_p
        keep[0] = True
        filtered = torch.zeros_like(probabilities)
        filtered[sorted_indices[keep]] = sorted_probs[keep]
        total = float(filtered.sum().item())
        if total > 0.0 and math.isfinite(total):
            probabilities = filtered / total
        else:
            probabilities = torch.zeros_like(probabilities)
            probabilities[sorted_indices[0]] = 1.0
    pick = int(torch.multinomial(probabilities, 1, generator=generator).item())
    model_lp = float(raw_log_probs[pick].item())
    behavior_lp = float(math.log(max(float(probabilities[pick].item()), 1e-30)))
    return pick, model_lp, behavior_lp


@dataclass(frozen=True)
class GeneratedDecision:
    key: DecisionKey
    legal_set_digest: str
    candidate_index: int
    suffix_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    model_action_logprob: float
    finish_reason: str
    model_binding_digest: str
    sampling: str = 'greedy'
    behavior_action_logprob: float = 0.0
    actor_ref: dict | None = None


@dataclass(frozen=True)
class EnvDecision:
    key: DecisionKey
    action: Action
    execution_ref: str
    generation: GeneratedDecision


def legal_digest(view, suffixes):
    w = view.key.window
    key_dict = {
        'window': {
            'env_instance_id': w.env_instance_id,
            'game_id': w.game_id,
            'round_id': w.round_id,
            'window_id': w.window_id,
            'state_revision': w.state_revision,
        },
        'viewer': view.key.viewer,
    }
    return digest(dict(domain='majang-policy-inference-legal-set/v1', key=key_dict,
        rules_digest=view.rules_digest, event_adapter=EVENT_ADAPTER,
        action_adapter=ACTION_ADAPTER, suffixes=suffixes))


class PolicyInference:
    def __init__(self, model, *, source=None, precision=None):
        from mahjong.policy.execution import policy_precision, set_policy_precision
        self.model = model.eval()
        self.lm = model.causal_lm
        self.codec = MahjongTokenizer()
        self.input_schema = getattr(model, 'input_schema', LEGACY_SCHEMA)
        self.sequence_codec = codec_for_schema(self.input_schema)
        self.source = source
        self.device = next(model.parameters()).device
        if any(p.dtype != torch.float32 for p in model.parameters()):
            fail('CONTRACT_MISMATCH', 'inference requires FP32 parameters')
        if precision is not None:
            if precision == 'bf16-mixed':
                if self.device.type != 'cuda' or not torch.cuda.is_bf16_supported():
                    fail('DEVICE_UNAVAILABLE', 'BF16 policy inference requires a supported GPU')
                # An eager config from custom weights must not bypass the fused
                # attention enforced by policy_execution on long GPU contexts.
                self.lm.set_attn_implementation('sdpa')
            set_policy_precision(model, precision)
        self.binding_digest = digest(dict(domain='majang-policy-inference-binding/v1',
            codec={**BINDING, 'codec_spec_sha256': self.codec.codec_spec_sha256},
            descriptor=asdict(model.descriptor()), precision=policy_precision(model),
            backend=self.lm.config._attn_implementation,
            context=self.lm.config.max_position_embeddings))
        self._prepared = {}
        self._views = {}
        self._lock = threading.RLock()
        self._validated_requests = OrderedDict()
        self._validated_tries = OrderedDict()
        self._validated_heads = []
        self._end_history_id = self.codec.token_id('END_HISTORY')
        self._decision_id = self.codec.token_id('DECISION')

    @classmethod
    def from_checkpoint(cls, path, *, device='cpu', precision=None):
        from .loading import load_policy_checkpoint
        model, source = load_policy_checkpoint(path, device=device)
        return cls(model, source=source, precision=precision)

    @classmethod
    def from_weights(cls, path, *, device='cpu', precision=None):
        from .weights import load_policy_weights
        model, source = load_policy_weights(path, device=device)
        return cls(model, source=source, precision=precision)

    @classmethod
    def from_actor_snapshot(cls, path, *, device='cpu'):
        from .actor_loading import load_actor
        return load_actor(path, device=device)

    def new_bridge(self, profile):
        """Verify the frozen training build's profile before opening an environment.

        This establishes rule identity, not completeness of legacy candidates
        or model strength. Missing source evidence is explicit and not defaulted.
        """
        if self.source and 'actor_ref' in self.source:
            from .actor_loading import new_actor_bridge
            if digest(profile) != digest(self.source['compatibility']['profile']):
                fail('UNSUPPORTED_PROFILE', 'profile differs from Actor evidence')
            return new_actor_bridge([self] * 4)
        self.verify_training_profile(profile)
        from mahjong.envs.policy_bridge import RiichiDecisionBridge
        if self.input_schema == CONTEXT_SCHEMA:
            from .context_bridge import ContextDecisionBridge
            bridge = ContextDecisionBridge(profile, model_binding_digest=self.binding_digest)
        else:
            bridge = RiichiDecisionBridge(profile, model_binding_digest=self.binding_digest)
        self.source['training_rules_status'] = 'verified frozen processing-assumption profile and dataset fingerprint'
        self.source['action_normalization_status'] = 'legacy candidate compatibility requires separate differential verification'
        return bridge

    def verify_training_profile(self, profile):
        """Verify source identity independently of the selected Web match length."""
        from mahjong.data.mahjong.corpus import open_policy_reader as CleaningReader
        if not self.source or not self.source.get('training_data'):
            fail('UNSUPPORTED_PROFILE', 'checkpoint training source evidence is missing')
        data = self.source['training_data']
        try:
            with CleaningReader(data['build_dir']) as reader:
                from mahjong.data.mahjong.policy import dataset_fingerprint
                fingerprint = dataset_fingerprint(reader, split_seed=data['split_seed'],
                    validation_fraction=data['validation_fraction'],
                    max_sequence_length=self.lm.config.max_position_embeddings, input_schema=self.input_schema)
                if fingerprint != self.source['training_data_fingerprint']:
                    fail('UNSUPPORTED_PROFILE', 'frozen training build differs from checkpoint data fingerprint')
                entry = reader.catalog['binding']['rules'].get(profile['ruleset_id'])
                if entry is None or digest(entry['profile']) != digest(profile):
                    fail('UNSUPPORTED_PROFILE', 'online profile differs from frozen training rules')
        except (KeyError, OSError, ValueError) as exc:
            fail('UNSUPPORTED_PROFILE', f'cannot verify frozen training rule evidence: {exc}')
        self.source['training_rules_status'] = 'verified frozen processing-assumption profile and dataset fingerprint'

    def _check_context_budget(self, prefix_len, longest):
        max_pos = self.lm.config.max_position_embeddings
        if prefix_len + longest > max_pos:
            fail('CONTEXT_TOO_LONG', 'prefix plus longest legal suffix exceeds context',
                 prefix_length=prefix_len, longest_suffix=longest,
                 context=max_pos)

    def _validate(self, prefix, suffixes):
        is_imm_tuple = type(prefix) is tuple and type(suffixes) is tuple
        if is_imm_tuple:
            cache_key = (id(prefix), id(suffixes))
            with self._lock:
                entry = self._validated_tries.get(cache_key)
                if entry is not None and entry[0] is prefix and entry[1] is suffixes:
                    self._validated_tries.move_to_end(cache_key)
                    trie, longest = entry[2], entry[3]
                    self._check_context_budget(len(prefix), longest)
                    return trie, longest

        prefix_t = prefix if type(prefix) is tuple else (tuple(prefix) if type(prefix) is list else None)
        matched_head = None
        if prefix_t is not None:
            p_len = len(prefix_t)
            with self._lock:
                for idx, (h_end, h_tail, head) in enumerate(self._validated_heads):
                    if p_len > h_end and prefix_t[h_end - len(h_tail) : h_end] == h_tail and prefix_t[:h_end] == head:
                        matched_head = h_end
                        if idx > 0:
                            self._validated_heads.insert(0, self._validated_heads.pop(idx))
                        break

        new_h_end = -1
        try:
            if matched_head is not None:
                tail = prefix_t[matched_head:]
                events, hand_state = self.sequence_codec.decode_policy_prefix_tail(tail)
                reencoded = []
                for ev in events:
                    reencoded.extend(self.sequence_codec.encode_event(ev))
                reencoded.append(self._end_history_id)
                enc_hs = self.sequence_codec.encode_hand_state(hand_state)
                reencoded.extend(enc_hs)
                reencoded.append(self._decision_id)
                if tuple(reencoded) != tail:
                    fail('CONTRACT_MISMATCH', 'noncanonical prefix')
                new_h_end = len(prefix_t) - len(enc_hs) - 2
            else:
                seq = self.sequence_codec.decode_policy_prefix(prefix)
                if self.input_schema == CONTEXT_SCHEMA:
                    from mahjong.policy.context import validate_context
                    validate_context(seq)
                if tuple(self.sequence_codec.encode_policy_prefix(seq)) != tuple(prefix):
                    fail('CONTRACT_MISMATCH', 'noncanonical prefix')
                if prefix_t is not None:
                    enc_hs = self.sequence_codec.encode_hand_state(seq.hand_state)
                    new_h_end = len(prefix_t) - len(enc_hs) - 2

            if not suffixes:
                fail('LEGAL_SET_EMPTY', 'empty legal set')
            for suffix in suffixes:
                action = self.codec.decode_action_suffix(suffix)
                if self.codec._encode_action_suffix_tuple(action) != tuple(suffix):
                    fail('ACTION_UNREPRESENTABLE', 'noncanonical action suffix')
        except ProtocolError as exc:
            fail('ACTION_UNREPRESENTABLE', str(exc))

        trie = build_trie(suffixes)
        longest = max(map(len, suffixes))
        self._check_context_budget(len(prefix), longest)

        if prefix_t is not None and new_h_end > 0 and prefix_t[new_h_end] == self._end_history_id:
            head_slice = prefix_t[:new_h_end]
            tail_sig = head_slice[max(0, new_h_end - 8) : new_h_end]
            with self._lock:
                self._validated_heads = [h for h in self._validated_heads if h[0] != new_h_end or h[2] != head_slice]
                self._validated_heads.insert(0, (new_h_end, tail_sig, head_slice))
                if len(self._validated_heads) > 16:
                    self._validated_heads.pop()

        if is_imm_tuple and all(type(s) is tuple for s in suffixes):
            cache_key = (id(prefix), id(suffixes))
            with self._lock:
                self._validated_tries[cache_key] = (prefix, suffixes, trie, longest)
                self._validated_tries.move_to_end(cache_key)
                while len(self._validated_tries) > 64:
                    self._validated_tries.popitem(last=False)

        return trie, longest

    def prepare(self, view):
        if not view.legal_set_complete:
            fail('LEGAL_SET_INCOMPLETE', 'candidate completeness is unknown')
        suffixes = tuple(sorted(self.codec._encode_action_suffix_tuple(m.action) for m in view.legal_moves))
        trie, longest = self._validate(view.prefix_ids, suffixes)
        request = PreparedDecision(view.key, view.prefix_ids, suffixes,
                                   self.binding_digest, legal_digest(view, suffixes))
        previous = self._prepared.get(view.key)
        if previous is not None and (previous != request or self._views[view.key] != view):
            fail('STALE_DECISION', 'same key cannot bind different content')
        # Only prepared requests in the most recent window are retained here.
        self._prepared = {k: v for k, v in self._prepared.items() if k.window == view.key.window}
        self._prepared[view.key] = request
        self._views = {k: v for k, v in self._views.items() if k.window == view.key.window}
        self._views[view.key] = view
        req_key = (request.key, request.legal_set_digest)
        with self._lock:
            stale = [k for k in self._validated_requests if k[0].window != view.key.window]
            for k in stale:
                self._validated_requests.pop(k, None)
            self._validated_requests[req_key] = (request.prefix_ids, request.legal_suffixes, trie, longest)
            self._validated_requests.move_to_end(req_key)
            while len(self._validated_requests) > 64:
                self._validated_requests.popitem(last=False)
        return request

    def get_validated_trie(self, request):
        if (not isinstance(request, PreparedDecision)
                or self._prepared.get(request.key) != request
                or request.binding_digest != self.binding_digest):
            fail('STALE_DECISION', 'request differs from frozen preparation')
        req_key = (request.key, request.legal_set_digest)
        with self._lock:
            entry = self._validated_requests.get(req_key)
            if entry is not None:
                self._validated_requests.move_to_end(req_key)
        if (entry is not None
                and type(request.prefix_ids) is tuple
                and type(request.legal_suffixes) is tuple):
            cached_prefix, cached_suffixes, trie, longest = entry
            if (request.prefix_ids is cached_prefix and request.legal_suffixes is cached_suffixes) or (
                all(type(s) is tuple for s in request.legal_suffixes)
                and request.prefix_ids == cached_prefix
                and request.legal_suffixes == cached_suffixes
            ):
                self._check_context_budget(len(request.prefix_ids), longest)
                return trie, longest
        return self._validate(request.prefix_ids, request.legal_suffixes)

    def generate(self, request, *, options=DecodeOptions(), cancel=None, deadline_monotonic=None, generator=None,
                 prefix_cache=None):
        from mahjong.policy.execution import policy_execution
        from .prefix_cache import incremental_attention, GPU_PREFILL_CHUNK_TOKENS
        def check():
            if cancel is not None and cancel():
                if prefix_cache is not None:
                    prefix_cache.clear()
                fail('CANCELLED', 'decision cancelled')
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                if prefix_cache is not None:
                    prefix_cache.clear()
                fail('DEADLINE_EXCEEDED', 'decision deadline exceeded')
        check()
        validate_decode_options(options, generator)
        if self._prepared.get(request.key) != request or request.binding_digest != self.binding_digest:
            fail('STALE_DECISION', 'request differs from frozen preparation')
        node, limit = self._validate(request.prefix_ids, request.legal_suffixes)
        prefix = request.prefix_ids
        generated, logprobs, behavior_logprobs = [], [], []
        cache = out = kwargs = None
        success = False
        try:
            with torch.inference_mode(), policy_execution(self.model):
                for offset in range(limit):
                    check()
                    allowed = sorted(k for k in node if k is not None)
                    if options.skip_deterministic_tail and deterministic_tail(node):
                        # No later branch needs this token's K/V. Keep the CPU
                        # singleton draw so subsequent categorical RNG is exact.
                        token = allowed[0]
                        if options.strategy == 'categorical':
                            torch.multinomial(torch.ones(1), 1, generator=generator)
                        logprobs.append(0.)
                        behavior_logprobs.append(0.)
                    else:
                        reused = 0
                        if offset == 0 and prefix_cache is not None and options.use_kv_cache:
                            cache, reused = prefix_cache.take(self, request)
                        ids = (prefix + tuple(generated) if not options.use_kv_cache or offset == 0
                               else (generated[-1],))
                        if reused:
                            ids = ids[reused:]
                        length = len(prefix) + offset
                        gpu_prefix = bool(prefix_cache is not None and options.use_kv_cache
                                          and self.device.type == 'cuda' and torch.is_autocast_enabled('cuda'))
                        # Bound transient prefill allocations without removing
                        # history. Every chunk keeps its absolute RoPE positions.
                        chunk_size = GPU_PREFILL_CHUNK_TOKENS if gpu_prefix else len(ids)
                        for start in range(0, len(ids), chunk_size):
                            check()
                            chunk = ids[start:start + chunk_size]
                            end = length - len(ids) + start + len(chunk)
                            kwargs = dict(input_ids=torch.tensor([chunk], device=self.device),
                                attention_mask=torch.ones((1, end), dtype=torch.int64, device=self.device),
                                position_ids=torch.arange(end-len(chunk), end, device=self.device)[None],
                                use_cache=options.use_kv_cache, logits_to_keep=1, return_dict=True)
                            if cache is not None and options.use_kv_cache:
                                kwargs['past_key_values'] = cache
                            aligned = bool(gpu_prefix and cache is not None and len(chunk) > 1)
                            if aligned:
                                kwargs['attention_mask'] = {'full_attention': None}
                            with incremental_attention(self.lm, aligned):
                                out = self.lm(**kwargs)
                            cache = out.past_key_values if options.use_kv_cache else None
                            check()
                        logits = out.logits[0, -1].float()
                        allowed_logits = logits[allowed] if allowed else logits.new_empty((0,))
                        if not allowed or not bool(torch.isfinite(allowed_logits).all()):
                            fail('NONFINITE_LOGITS', 'empty successors or nonfinite legal logits')
                        pick, model_lp, behavior_lp = select_legal_token(allowed_logits, options, generator)
                        token = allowed[pick]
                        logprobs.append(model_lp)
                        behavior_logprobs.append(behavior_lp)
                    generated.append(token)
                    node = node[token]
                    if token == self.codec.token_id('END_DECISION'):
                        if set(node) != {None}:
                            fail('INCOMPLETE_GENERATION', 'END_DECISION is not a terminal leaf')
                        index, suffix = node[None], tuple(generated)
                        if suffix != request.legal_suffixes[index]:
                            fail('INCOMPLETE_GENERATION', 'generated action is not in original legal set')
                        action = self.codec.decode_action_suffix(suffix)
                        if tuple(self.codec.encode_action_suffix(action)) != suffix:
                            fail('ACTION_UNREPRESENTABLE', 'generated action failed round trip')
                        result = GeneratedDecision(request.key, request.legal_set_digest, index,
                            suffix, tuple(logprobs), math.fsum(logprobs), 'end_decision', self.binding_digest,
                            options.strategy, math.fsum(behavior_logprobs) if options.strategy == 'categorical' else 0.,
                            None if self.source is None else deepcopy(self.source.get('actor_ref')))
                        check()
                        if prefix_cache is not None and options.use_kv_cache:
                            prefix_cache.put(self, request, cache)
                        success = True
                        return result
                fail('INCOMPLETE_GENERATION', 'legal depth exceeded')
        except RuntimeError as exc:
            if isinstance(exc, MahjongError):
                raise
            fail('MODEL_EXECUTION_FAILED', f'model forward failed: {exc}',
                 decision_key=asdict(request.key), stage='forward')
        finally:
            if not success and prefix_cache is not None:
                prefix_cache.clear()
            cache = out = kwargs = None
