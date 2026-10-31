"""Official randomly initialized Qwen3 and the project's Token CausalLM adapter."""

import hashlib
import math
import os
from pathlib import Path

import torch
from torch import nn


def _force_hf_offline():
    # Force the process policy even if Hub was imported before this module.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from huggingface_hub import constants
    constants.HF_HUB_OFFLINE = True
    constants.HF_HUB_DISABLE_TELEMETRY = True


_force_hf_offline()
from transformers import Qwen3Config, Qwen3ForCausalLM

from mahjong.contracts import (Batch, ComponentDescriptor, MahjongError, MetricStat,
                              State, StepResult, canonical_json)
from mahjong.policy.constraints import build_trie
from mahjong.tokenizer import MahjongTokenizer
from mahjong.tokenizer.codec import BINDING as TOKENIZER_BINDING
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, CONTEXT_BINDING

INPUT_SCHEMA = 'token-causallm/v1'
POLICY_INPUT_SCHEMA = 'majang-policy-ntp/v1'
POLICY_INPUT_SCHEMAS = (POLICY_INPUT_SCHEMA, CONTEXT_SCHEMA)
MIXED_EXECUTION = ('cpu/32-true', 'cuda:0/32-true', 'cpu/bf16-mixed', 'cuda:0/bf16-mixed')
SIZE_FIELDS = ('num_hidden_layers', 'hidden_size', 'intermediate_size',
               'num_attention_heads', 'num_key_value_heads', 'head_dim',
               'vocab_size', 'max_position_embeddings')
SETTINGS_FIELDS = set(SIZE_FIELDS) | {
    'name', 'input_schema', 'attn_implementation', 'tie_word_embeddings',
    'rope_theta', 'use_cache', 'bos_token_id', 'eos_token_id', 'pad_token_id',
}


def validate_settings(settings):
    """Validate our small configuration surface, leaving architecture to HF."""
    def invalid(field, reason):
        raise MahjongError('CONFIG_INVALID', f'model.{field}: {reason}')
    if type(settings) is not dict:
        invalid('config', 'expected mapping')
    for field in sorted(settings.keys() - SETTINGS_FIELDS):
        invalid(field, 'unknown field')
    for field in sorted(SETTINGS_FIELDS - settings.keys()):
        invalid(field, 'required field missing')
    if settings['name'] != 'qwen3' or settings['input_schema'] not in (INPUT_SCHEMA, *POLICY_INPUT_SCHEMAS):
        invalid('name/input_schema', 'expected qwen3 with a supported token policy schema')
    if settings['input_schema'] in POLICY_INPUT_SCHEMAS and settings['vocab_size'] != MahjongTokenizer.vocab_size:
        invalid('vocab_size', f'policy inputs require the fixed {MahjongTokenizer.vocab_size}-token Mahjong vocabulary')
    for field in SIZE_FIELDS:
        if type(settings[field]) is not int or settings[field] <= 0:
            invalid(field, 'expected positive integer')
    if settings['num_attention_heads'] % settings['num_key_value_heads']:
        invalid('num_key_value_heads', 'must divide num_attention_heads')
    if settings['head_dim'] % 2:
        invalid('head_dim', 'RoPE requires an even dimension')
    if settings['attn_implementation'] not in ('sdpa', 'eager'):
        invalid('attn_implementation', 'supported backends: sdpa, eager')
    for field in ('tie_word_embeddings', 'use_cache'):
        if type(settings[field]) is not bool:
            invalid(field, 'expected boolean')
    theta = settings['rope_theta']
    if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
        invalid('rope_theta', 'expected finite positive number')
    for field in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        value = settings[field]
        if value is not None and (type(value) is not int or not 0 <= value < settings['vocab_size']):
            invalid(field, 'expected null or ID within vocab_size')
    return settings


def build_qwen3(settings: dict, *, init_seed: int) -> Qwen3ForCausalLM:
    """Build on CPU in FP32 without network access or caller RNG consumption."""
    _force_hf_offline()
    validate_settings(settings)
    if type(init_seed) is not int or not 0 <= init_seed < 2**63:
        raise MahjongError('CONFIG_INVALID', 'model.init_seed: expected seed in [0,2**63)')
    options = {field: settings[field] for field in SIZE_FIELDS}
    options.update({field: settings[field] for field in (
        'tie_word_embeddings', 'use_cache', 'bos_token_id', 'eos_token_id', 'pad_token_id')})
    config = Qwen3Config(**options, rope_parameters={
        'rope_type': 'default', 'rope_theta': float(settings['rope_theta']),
    }, attn_implementation=settings['attn_implementation'])
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(init_seed)
        # Explicit context keeps caller dtype/device defaults from changing initialization.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            with torch.device('cpu'):
                model = Qwen3ForCausalLM(config)
        finally:
            torch.set_default_dtype(previous_dtype)
    return model


def load_qwen3_local(directory: str | Path, *, attn_implementation='sdpa') -> Qwen3ForCausalLM:
    """Import an HF export, never interpret a remote model ID as a source."""
    _force_hf_offline()
    path = Path(directory)
    if not path.is_dir():
        raise MahjongError('CONFIG_INVALID', f'model.directory: local directory required: {path}')
    if attn_implementation not in ('sdpa', 'eager'):
        raise MahjongError('CONFIG_INVALID', 'model.attn_implementation: supported backends: sdpa, eager')
    return Qwen3ForCausalLM.from_pretrained(
        path, local_files_only=True, attn_implementation=attn_implementation, dtype=torch.float32)


def _semantic_config(config):
    # Exclude HF export/display metadata; retain structure and learning semantics.
    excluded = {'transformers_version', 'architectures', 'dtype', '_name_or_path',
                'id2label', 'label2id', 'problem_type', 'output_hidden_states',
                'output_attentions', 'return_dict'}
    def normalize(value):
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        return value
    result = normalize({key: value for key, value in config.to_dict().items()
                        if key not in excluded})
    result['attn_implementation'] = config._attn_implementation
    result['input_schema'] = INPUT_SCHEMA
    return result


class Qwen3Adapter(nn.Module):
    """Batch payload: int64 IDs/labels [B,T], optional binary mask [B,T].

    Labels are unshifted; HF predicts labels[:,1:], ignoring -100. A mask
    controls attention only. The producer must independently ignore loss labels.
    Native logits/generation/export are available through ``causal_lm``.
    """
    def __init__(self, causal_lm: Qwen3ForCausalLM):
        super().__init__()
        self.causal_lm = causal_lm

    def descriptor(self) -> ComponentDescriptor:
        fingerprint = hashlib.sha256(canonical_json(
            _semantic_config(self.causal_lm.config)).encode()).hexdigest()
        return ComponentDescriptor('model', 'majang.qwen3/v1', 1, INPUT_SCHEMA,
                                   fingerprint, None, 1, 'state_dict',
                                   MIXED_EXECUTION)

    def _payload(self, batch: Batch):
        def invalid(reason):
            raise MahjongError('CONTRACT_MISMATCH', f'Qwen3 payload: {reason}', 'training')
        if batch.input_schema != INPUT_SCHEMA:
            invalid('expected token-causallm/v1 input_schema')
        payload = batch.payload
        if (type(payload) is not dict or not {'input_ids', 'labels'} <= payload.keys()
                or payload.keys() - {'input_ids', 'labels', 'attention_mask'}):
            invalid('requires input_ids, labels and optional attention_mask')
        ids = payload['input_ids']
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 2
                or ids.shape[0] != batch.sample_count or ids.shape[1] < 1):
            invalid('input_ids must be int64 [sample_count,T], T>=1')
        if ids.shape[1] > self.causal_lm.config.max_position_embeddings:
            invalid('T exceeds max_position_embeddings')
        vocab = self.causal_lm.config.vocab_size
        if bool(((ids < 0) | (ids >= vocab)).any()):
            invalid('input_ids outside vocab_size')
        labels = payload['labels']
        if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
                or labels.shape != ids.shape):
            invalid('labels must be int64 with input_ids shape')
        if bool(((labels != -100) & ((labels < 0) | (labels >= vocab))).any()):
            invalid('labels must be -100 or within vocab_size')
        mask = payload.get('attention_mask')
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != ids.shape
                                 or mask.dtype not in (torch.bool, torch.int64, torch.int32)
                                 or bool(((mask != 0) & (mask != 1)).any())):
            invalid('attention_mask must be binary bool/integer with input_ids shape')
        return payload

    def prepare_batch(self, batch: Batch, device: torch.device) -> Batch:
        payload = self._payload(batch)
        return Batch({key: value.to(device) for key, value in payload.items()},
                     batch.sample_ids, batch.sample_count, batch.input_schema)

    def forward(self, batch: Batch, *, phase: str) -> StepResult:
        if phase not in ('train', 'eval'):
            raise MahjongError('CONTRACT_MISMATCH', 'Qwen3 phase must be train/eval', 'training')
        payload = self._payload(batch)
        count = int((payload['labels'][:, 1:] != -100).sum().item())
        if count == 0:
            return StepResult(None, {'loss': MetricStat(0., 0, 'nats')},
                              'no_valid_target_tokens' if phase == 'train' else None)
        # KV cache is an inference feature, not auxiliary training state.
        loss = self.causal_lm(**payload, use_cache=False).loss
        if not bool(torch.isfinite(loss).item()):
            raise MahjongError('NONFINITE_LOSS', 'Qwen3 produced nonfinite loss',
                              'training' if phase == 'train' else 'evaluating')
        return StepResult(loss, {'loss': MetricStat(float(loss.detach().item()) * count, count, 'nats')})

    def auxiliary_state_dict(self) -> State:
        return {}

    def load_auxiliary_state_dict(self, state: State) -> None:
        if type(state) is not dict or state:
            raise MahjongError('CHECKPOINT_INVALID', 'Qwen3 auxiliary state must be empty', 'restoring')

    def reset_auxiliary_rng(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise MahjongError('CONTRACT_MISMATCH', 'Qwen3 reset seed must be in [0,2**63)', 'evaluating')


class Qwen3PolicyAdapter(nn.Module):
    """Teacher-forced action-token NTP with legal continuations before softmax.

    The official CausalLM vocabulary head predicts each next action token. A trie
    of complete legal suffixes supplies the allowed vocabulary at that position;
    the trie depends only on legality and the preceding tokens, never the target
    token. Context and padding have no loss; END_DECISION is part of the suffix.
    """
    def __init__(self, causal_lm: Qwen3ForCausalLM, *, input_schema=POLICY_INPUT_SCHEMA):
        super().__init__()
        self.causal_lm = causal_lm
        if input_schema not in POLICY_INPUT_SCHEMAS:
            raise MahjongError('CONFIG_INVALID', 'unsupported policy input schema')
        self.input_schema = input_schema

    def descriptor(self) -> ComponentDescriptor:
        binding = {**TOKENIZER_BINDING, 'codec_spec_sha256': MahjongTokenizer().codec_spec_sha256}
        if self.input_schema == CONTEXT_SCHEMA:
            binding['context'] = CONTEXT_BINDING
        fingerprint = hashlib.sha256(canonical_json({
            'model': _semantic_config(self.causal_lm.config),
            'policy_binding': binding,
            'objective': 'legal-next-token-masked-logits/token-mean/v1',
        }).encode()).hexdigest()
        return ComponentDescriptor('model', 'majang.qwen3_legal_ntp/v1', 1,
            self.input_schema, fingerprint, None, 1, 'state_dict', MIXED_EXECUTION)

    def _payload(self, batch: Batch):
        def invalid(reason):
            raise MahjongError('CONTRACT_MISMATCH', f'Qwen3 policy payload: {reason}', 'training')
        if batch.input_schema != self.input_schema or type(batch.payload) is not dict:
            invalid(f'expected {self.input_schema} mapping')
        payload = batch.payload
        expected = {'input_ids', 'attention_mask', 'candidate_ids', 'target_indices'}
        if payload.keys() != expected:
            invalid('expected input_ids, attention_mask, candidate_ids, target_indices')
        ids, mask = payload['input_ids'], payload['attention_mask']
        candidates, targets = payload['candidate_ids'], payload['target_indices']
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 2
                or ids.shape[0] != batch.sample_count or not 1 <= ids.shape[1] <= self.causal_lm.config.max_position_embeddings):
            invalid('input_ids must be int64 [B,T] within configured context')
        if (not isinstance(mask, torch.Tensor) or mask.shape != ids.shape
                or mask.dtype not in (torch.bool, torch.int64)
                or bool((mask.sum(1) < 1).any()) or bool(((mask != 0) & (mask != 1)).any())):
            invalid('attention_mask must be binary and contain at least one token per sample')
        if bool((mask[:, 1:].to(torch.bool) & ~mask[:, :-1].to(torch.bool)).any()):
            invalid('attention_mask must use right padding')
        if (not isinstance(candidates, torch.Tensor) or candidates.dtype != torch.int64
                or candidates.ndim != 3 or candidates.shape[0] != batch.sample_count
                or candidates.shape[1] < 2 or candidates.shape[2] < 1):
            invalid('candidate_ids must be int64 [B,C,A], C>=2, A>=1')
        active = candidates != 0
        candidate_counts = active.any(-1).sum(-1)
        if bool((candidate_counts < 2).any()):
            invalid('every sample needs at least two nonempty legal candidates')
        if bool((active[:, :, 1:] & ~active[:, :, :-1]).any()):
            invalid('candidate padding must be a right suffix')
        candidate_valid = active.any(-1)
        if bool((candidate_valid[:, 1:] & ~candidate_valid[:, :-1]).any()):
            invalid('candidate padding must be a right suffix')
        vocab = self.causal_lm.config.vocab_size
        if bool(((ids < 0) | (ids >= vocab)).any()) or bool(((candidates < 0) | (candidates >= vocab)).any()):
            invalid('token ID outside configured vocabulary')
        if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64
                or targets.shape != (batch.sample_count,)):
            invalid('target_indices must be int64 [B]')
        if bool((targets < 0).any()) or bool((targets >= candidate_counts).any()):
            invalid('target index outside legal candidate set')
        return payload, active

    def prepare_batch(self, batch: Batch, device: torch.device) -> Batch:
        from .nanikiru import is_nanikiru, prepare_nanikiru
        if is_nanikiru(batch):
            return prepare_nanikiru(self, batch)
        payload, _ = self._payload(batch)
        return Batch({key: value.to(device) for key, value in payload.items()},
                     batch.sample_ids, batch.sample_count, batch.input_schema)

    def _teacher_forcing(self, payload, active):
        ids, mask = payload['input_ids'], payload['attention_mask']
        candidates, targets = payload['candidate_ids'], payload['target_indices']
        lengths = mask.sum(1).tolist()
        action_lengths = active.sum(-1).tolist()
        candidate_lists = candidates.tolist()
        target_indices = targets.tolist()
        groups, chosen = [], []
        for row, (padded, sizes, target) in enumerate(zip(candidate_lists, action_lengths, target_indices)):
            actions = [tokens[:size] for tokens, size in zip(padded, sizes) if size]
            if lengths[row] + max(map(len, actions)) > self.causal_lm.config.max_position_embeddings:
                raise MahjongError('CONTRACT_MISMATCH',
                    'Qwen3 policy prefix + longest legal action exceeds context', 'training')
            groups.append(actions)
            chosen.append(actions[target])
        width = max(length + len(action) for length, action in zip(lengths, chosen))
        input_ids = ids.new_zeros((len(lengths), width))
        attention_mask = mask.new_zeros((len(lengths), width))
        labels = ids.new_full((len(lengths), width), -100)
        allowed = []
        for row, (length, actions, target) in enumerate(zip(lengths, groups, chosen)):
            end = length + len(target)
            input_ids[row, :length] = ids[row, :length]
            input_ids[row, length:end] = ids.new_tensor(target)
            attention_mask[row, :end] = 1
            labels[row, length:end] = ids.new_tensor(target)
            # None marks a completed suffix. Prefix-free suffixes are required:
            # real Mahjong suffixes carry an explicit END_DECISION token.
            trie = build_trie(actions)
            node = trie
            for token in target:
                allowed.append(tuple(node))
                node = node[token]
        legal_mask = torch.zeros((len(allowed), self.causal_lm.config.vocab_size),
                                 dtype=torch.bool, device=ids.device)
        for row, tokens in enumerate(allowed):
            legal_mask[row, list(tokens)] = True
        return input_ids, attention_mask, labels, legal_mask

    def forward(self, batch: Batch, *, phase: str) -> StepResult:
        from .nanikiru import evaluate_nanikiru, is_nanikiru
        if is_nanikiru(batch):
            return evaluate_nanikiru(self, batch, phase=phase)
        if phase not in ('train', 'eval'):
            raise MahjongError('CONTRACT_MISMATCH', 'Qwen3 policy phase must be train/eval', 'training')
        payload, active = self._payload(batch)
        ids, mask, labels, legal_mask = self._teacher_forcing(payload, active)
        output = self.causal_lm(input_ids=ids, attention_mask=mask, use_cache=False,
                               return_dict=True)
        # Position t predicts token t+1. Select action labels only, so neither
        # context nor padding contributes a row to the vocabulary softmax/loss.
        supervised = labels[:, 1:] != -100
        targets = labels[:, 1:][supervised]
        logits = output.logits[:, :-1][supervised].float()
        logits = logits.masked_fill(~legal_mask, -torch.inf)
        # cross_entropy consumes logits and combines log_softmax + NLL stably.
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise MahjongError('NONFINITE_LOSS', 'Qwen3 policy produced nonfinite loss', 'training')
        correct = int((logits.argmax(-1) == targets).sum().item())
        count = targets.numel()
        return StepResult(loss, {
            'loss': MetricStat(float(loss.detach().item()) * count, count, 'nats'),
            'ntp_loss': MetricStat(float(loss.detach().item()) * count, count, 'nats'),
            'token_accuracy': MetricStat(float(correct), count, 'ratio'),
        })

    def auxiliary_state_dict(self) -> State: return {}

    def load_auxiliary_state_dict(self, state: State) -> None:
        if type(state) is not dict or state:
            raise MahjongError('CHECKPOINT_INVALID', 'Qwen3 policy auxiliary state must be empty', 'restoring')

    def reset_auxiliary_rng(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise MahjongError('CONTRACT_MISMATCH', 'Qwen3 policy seed must be in [0,2**63)', 'evaluating')
