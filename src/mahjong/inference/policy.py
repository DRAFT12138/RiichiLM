"""B=1 constrained autoregression with FP32 parameters and policy precision."""
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
import time

import torch

from mahjong.contracts import MahjongError
from mahjong.policy.constraints import build_trie, deterministic_tail
from mahjong.tokenizer import Action, MahjongTokenizer, ProtocolError
from mahjong.tokenizer.codec import BINDING
from mahjong.tokenizer.policy_context import (
    CONTEXT_SCHEMA, LEGACY_SCHEMA, PolicyContextInput, PolicyContextCodec, codec_for_schema,
)

EVENT_ADAPTER = 'riichi-online-ir/1'
ACTION_ADAPTER = 'riichi-policy-actions/1'


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
        codec = PolicyContextCodec() if isinstance(sequence, PolicyContextInput) else MahjongTokenizer()
        object.__setattr__(self, 'key', key)
        object.__setattr__(self, 'prefix_ids', tuple(codec.encode_policy_prefix(sequence)))
        moves = tuple(sorted((LegalMove(codec.decode_action_suffix(codec.encode_action_suffix(m.action)),
                                m.execution_ref) for m in legal_moves),
                             key=lambda m: tuple(codec.encode_action_suffix(m.action))))
        object.__setattr__(self, 'legal_moves', moves)
        object.__setattr__(self, 'rules_digest', rules_digest)
        object.__setattr__(self, 'legal_set_complete', legal_set_complete)

    @property
    def sequence(self):
        codec = PolicyContextCodec() if self.prefix_ids[3] == MahjongTokenizer().token_id('LIST') else MahjongTokenizer()
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
    return digest(dict(domain='majang-policy-inference-legal-set/v1', key=asdict(view.key),
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

    def _validate(self, prefix, suffixes):
        try:
            seq = self.sequence_codec.decode_policy_prefix(prefix)
            if self.input_schema == CONTEXT_SCHEMA:
                from mahjong.policy.context import validate_context
                validate_context(seq)
            if tuple(self.sequence_codec.encode_policy_prefix(seq)) != tuple(prefix):
                fail('CONTRACT_MISMATCH', 'noncanonical prefix')
            if not suffixes:
                fail('LEGAL_SET_EMPTY', 'empty legal set')
            for suffix in suffixes:
                action = self.codec.decode_action_suffix(suffix)
                if tuple(self.codec.encode_action_suffix(action)) != tuple(suffix):
                    fail('ACTION_UNREPRESENTABLE', 'noncanonical action suffix')
        except ProtocolError as exc:
            fail('ACTION_UNREPRESENTABLE', str(exc))
        trie = build_trie(suffixes)
        longest = max(map(len, suffixes))
        if len(prefix) + longest > self.lm.config.max_position_embeddings:
            fail('CONTEXT_TOO_LONG', 'prefix plus longest legal suffix exceeds context',
                 prefix_length=len(prefix), longest_suffix=longest,
                 context=self.lm.config.max_position_embeddings)
        return trie, longest

    def prepare(self, view):
        if not view.legal_set_complete:
            fail('LEGAL_SET_INCOMPLETE', 'candidate completeness is unknown')
        suffixes = tuple(sorted(tuple(self.codec.encode_action_suffix(m.action)) for m in view.legal_moves))
        self._validate(view.prefix_ids, suffixes)
        request = PreparedDecision(view.key, view.prefix_ids, suffixes,
                                   self.binding_digest, legal_digest(view, suffixes))
        previous = self._prepared.get(view.key)
        if previous is not None and (previous != request or self._views[view.key] != view):
            fail('STALE_DECISION', 'same key cannot bind different content')
        # Only the most recent window is retained; no histories or KV survive it.
        self._prepared = {k: v for k, v in self._prepared.items() if k.window == view.key.window}
        self._prepared[view.key] = request
        self._views = {k: v for k, v in self._views.items() if k.window == view.key.window}
        self._views[view.key] = view
        return request

    def generate(self, request, *, options=DecodeOptions(), cancel=None, deadline_monotonic=None, generator=None):
        from mahjong.policy.execution import policy_execution
        def check():
            if cancel is not None and cancel():
                fail('CANCELLED', 'decision cancelled')
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                fail('DEADLINE_EXCEEDED', 'decision deadline exceeded')
        check()
        if (options.strategy not in ('greedy', 'categorical') or type(options.use_kv_cache) is not bool
                or type(options.skip_deterministic_tail) is not bool):
            fail('CONTRACT_MISMATCH', 'unsupported decode options')
        if options.strategy == 'categorical' and (generator is None or generator.device.type != 'cpu'):
            fail('CONTRACT_MISMATCH', 'categorical sampling requires an explicit CPU generator')
        if self._prepared.get(request.key) != request or request.binding_digest != self.binding_digest:
            fail('STALE_DECISION', 'request differs from frozen preparation')
        node, limit = self._validate(request.prefix_ids, request.legal_suffixes)
        prefix = request.prefix_ids
        generated, logprobs = [], []
        cache = out = kwargs = None
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
                    else:
                        ids = (prefix + tuple(generated) if not options.use_kv_cache or offset == 0
                               else (generated[-1],))
                        length = len(prefix) + offset
                        kwargs = dict(input_ids=torch.tensor([ids], device=self.device),
                            attention_mask=torch.ones((1, length), dtype=torch.int64, device=self.device),
                            position_ids=torch.arange(length-len(ids), length, device=self.device)[None],
                            use_cache=options.use_kv_cache, logits_to_keep=1, return_dict=True)
                        if cache is not None and options.use_kv_cache:
                            kwargs['past_key_values'] = cache
                        out = self.lm(**kwargs)
                        cache = out.past_key_values if options.use_kv_cache else None
                        check()
                        logits = out.logits[0, -1].float()
                        if not allowed or not bool(torch.isfinite(logits[allowed]).all()):
                            fail('NONFINITE_LOGITS', 'empty successors or nonfinite legal logits')
                        masked = torch.full_like(logits, -torch.inf)
                        masked[allowed] = logits[allowed]
                        if options.strategy == 'categorical':
                            probabilities = torch.softmax(logits[allowed], -1).cpu()
                            token = allowed[int(torch.multinomial(probabilities, 1, generator=generator).item())]
                        else:
                            token = int(masked.argmax().item())
                        logprobs.append(float(torch.log_softmax(masked, -1)[token].item()))
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
                            options.strategy, math.fsum(logprobs) if options.strategy == 'categorical' else 0.,
                            None if self.source is None else deepcopy(self.source.get('actor_ref')))
                        check()
                        return result
                fail('INCOMPLETE_GENERATION', 'legal depth exceeded')
        except RuntimeError as exc:
            if isinstance(exc, MahjongError):
                raise
            fail('MODEL_EXECUTION_FAILED', f'model forward failed: {exc}',
                 decision_key=asdict(request.key), stage='forward')
        finally:
            # Cache is local even on cancellation, numeric error or model exception.
            cache = out = kwargs = None
