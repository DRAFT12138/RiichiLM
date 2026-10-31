"""Evaluation-only Nanikiru questions and the policy validation suite."""
import hashlib
import json
from pathlib import Path

from mahjong.contracts import Batch, ComponentDescriptor, MahjongError, encode_json_contract
from mahjong.policy.constraints import build_trie
from mahjong.policy.context import validate_context
from mahjong.policy.shared_context import NANIKIRU_LAYOUT
from mahjong.tokenizer import MahjongTokenizer
from mahjong.tokenizer.codec import BINDING
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, PolicyContextCodec
from .policy import SUPPORTED_EXECUTION, _fingerprint

class NanikiruEvalSource:
    """Load a checksummed snapshot containing only evaluable questions.

    Each question is its own batch, independent of the corpus validation limit
    and batch size. This keeps generation unpadded and cancellation bounded.
    """
    def __init__(self, directory, *, max_sequence_length):
        directory = Path(directory)
        try:
            manifest_bytes = (directory / 'manifest.json').read_bytes()
            manifest = json.loads(manifest_bytes)
            questions_bytes = (directory / 'questions.jsonl').read_bytes()
            self._load(manifest, questions_bytes, max_sequence_length)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MahjongError('CONFIG_INVALID', f'Invalid Nanikiru dataset: {exc}',
                               'preparing_data') from exc
        fingerprint = _fingerprint({'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'questions_sha256': manifest['questions_sha256'], 'layout': NANIKIRU_LAYOUT,
            'max_sequence_length': max_sequence_length})
        self._descriptor = ComponentDescriptor('data', 'mahjong.nanikiru_eval/v1', 1,
            CONTEXT_SCHEMA, fingerprint, fingerprint, 1, 'committed_batch', SUPPORTED_EXECUTION)

    def _load(self, manifest, questions_bytes, limit):
        def require(condition, message):
            if not condition:
                raise ValueError(message)
        codec, actions = PolicyContextCodec(), MahjongTokenizer()
        require(manifest['schema'] == 'mahjong-nanikiru/1' and
                manifest['split'] == 'evaluation_only' and
                manifest['policy_input_schema'] == CONTEXT_SCHEMA, 'unsupported schema/split')
        require(manifest['tokenizer_binding'] == {
            'name': BINDING['name'], 'vocab_sha256': BINDING['vocab_sha256'],
            'codec_spec_sha256': actions.codec_spec_sha256}, 'tokenizer binding mismatch')
        require(hashlib.sha256(questions_bytes).hexdigest() == manifest['questions_sha256'],
                'questions checksum mismatch')
        self.questions = []
        seen = set()
        for line in questions_bytes.decode('utf-8').splitlines():
            record = json.loads(line)
            require(record['schema'] == manifest['schema'], 'record schema mismatch')
            identity = record['id']
            require(type(identity) is str and identity and identity not in seen, 'duplicate/invalid question id')
            seen.add(identity)
            require(_fingerprint({k: v for k, v in record.items() if k != 'record_sha256'}) ==
                    record['record_sha256'], f'{identity}: record checksum mismatch')
            require(record['status'] == 'ready' and not record['issues'], f'{identity}: invalid status/issues')
            policy = record['policy']
            require(policy['input_schema'] == CONTEXT_SCHEMA, f'{identity}: input schema mismatch')
            prefix, candidates, targets = policy['input_ids'], policy['candidates'], policy['target_indices']
            require(type(prefix) is list and prefix and all(type(t) is int for t in prefix),
                    f'{identity}: invalid prefix')
            sequence = codec.decode_policy_prefix(prefix)
            validate_context(sequence)
            require(codec.encode_policy_prefix(sequence) == prefix, f'{identity}: noncanonical prefix')
            require(type(candidates) is list and len(candidates) >= 2, f'{identity}: invalid candidates')
            for suffix in candidates:
                require(type(suffix) is list and suffix and all(type(t) is int for t in suffix),
                        f'{identity}: invalid suffix')
                require(actions.encode_action_suffix(actions.decode_action_suffix(suffix)) == suffix,
                        f'{identity}: noncanonical suffix')
            build_trie(candidates)
            require(type(targets) is list and targets and len(set(targets)) == len(targets) and
                    all(type(t) is int and 0 <= t < len(candidates) for t in targets),
                    f'{identity}: invalid target indices')
            require(record['target']['kind'] in ('discard', 'riichi_discard'), f'{identity}: unsupported answer kind')
            require(len(prefix) + max(map(len, candidates)) <= limit,
                    f'{identity}: prefix + longest legal action exceeds context')
            self.questions.append({'input_ids': tuple(prefix),
                'candidates': tuple(tuple(a) for a in candidates), 'target_indices': tuple(targets),
                'answer_kind': record['target']['kind'], 'sample_id': identity})
        require(len(seen) == manifest['records'] and len(self.questions) == manifest['ready'] and
                manifest['excluded'] == [], 'manifest counts/exclusions mismatch')
        require(bool(self.questions), 'no ready questions')
        self.total, self.ready = len(seen), len(self.questions)

    def descriptor(self):
        return self._descriptor

    def iter_batches(self):
        for index, question in enumerate(self.questions):
            yield Batch({'layout': NANIKIRU_LAYOUT, **question,
                         'coverage': (self.ready, self.total) if index == 0 else None},
                        (question['sample_id'],), 1, CONTEXT_SCHEMA)


class PolicyEvalSuite:
    """Run the existing held-out corpus followed by the complete benchmark."""
    def __init__(self, corpus, nanikiru):
        self.corpus, self.nanikiru = corpus, nanikiru
        descriptors = [encode_json_contract(s.descriptor()) for s in (corpus, nanikiru)]
        self._descriptor = ComponentDescriptor('data', 'mahjong.policy_eval_suite/v1', 1,
            CONTEXT_SCHEMA, _fingerprint(descriptors),
            _fingerprint([d['data_fingerprint'] for d in descriptors]),
            1, 'committed_batch', SUPPORTED_EXECUTION)

    def descriptor(self):
        return self._descriptor

    def iter_batches(self):
        yield from self.corpus.iter_batches()
        yield from self.nanikiru.iter_batches()

    def close(self):
        self.corpus.close()
