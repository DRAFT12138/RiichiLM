"""Fixed v2 sequence codec. See docs/tokenizer/implementation-v2.md for scope.

No rules engine, state reconstruction, model dependency, or network access.
"""
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import csv
import hashlib
from importlib.resources import files
import io
import json
from pathlib import Path


class ProtocolError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def fail(code, detail):
    raise ProtocolError(code, detail)


@dataclass(frozen=True)
class Atom:
    """Explicit tile, seat, enum, yaku, UNKNOWN or HIDDEN (without angle brackets)."""
    name: str


@dataclass(frozen=True)
class Extension:
    namespace: str
    version: int
    payload: dict


@dataclass(frozen=True)
class Action:
    kind: str
    args: tuple = ()


@dataclass(frozen=True)
class Event:
    kind: str
    event_id: int
    payload: tuple = ()


@dataclass(frozen=True)
class ResponseBlock:
    payload: dict


@dataclass(frozen=True)
class HandMeldState:
    hands: list[Atom]
    melds: list[dict]


@dataclass(frozen=True)
class PolicySequenceInput:
    header: dict
    visible_history: list[Event]
    hand_state: HandMeldState


@dataclass(frozen=True)
class PolicySequence:
    input: PolicySequenceInput
    action: Action | None = None
    eos: bool = False


# Technical report §8.1; consumed tiles are positional, never a LIST.
ACTION_SHAPES = {
    'A_PASS': (), 'A_DISCARD': ('tile', 'discard'),
    'A_RIICHI_DISCARD': ('tile', 'discard'), 'A_TSUMO_WIN': (),
    'A_RON': ('seat', 'tile'), 'A_CHI': ('seat', 'tile', 'tile', 'tile'),
    'A_PON': ('seat', 'tile', 'tile', 'tile'),
    'A_DAIMINKAN': ('seat', 'tile', 'tile', 'tile', 'tile'),
    'A_ANKAN': ('tile', 'tile', 'tile', 'tile'),
    'A_KAKAN': ('natural', 'tile'), 'A_ABORT_KYUUSHU': (),
}
EVENT_FIELDS = {
    'EV_GAME_START': ('header',), 'EV_ROUND_START': ('state',),
    'EV_DRAW': ('actor', 'tile', 'draw_kind'),
    'EV_DISCARD': ('actor', 'tile', 'discard_kind', 'riichi_tile'),
    'EV_CHI': ('actor', 'target', 'tile', 'consumed', 'meld_id', 'trigger_id'),
    'EV_PON': ('actor', 'target', 'tile', 'consumed', 'meld_id', 'trigger_id'),
    'EV_DAIMINKAN': ('actor', 'target', 'tile', 'consumed', 'meld_id', 'trigger_id'),
    'EV_ANKAN_ATTEMPT': ('actor', 'base_tile', 'consumed', 'meld_id', 'window_id'),
    'EV_KAKAN_ATTEMPT': ('actor', 'tile', 'meld_id', 'window_id'),
    'EV_KAN_COMMIT': ('actor', 'meld_id'),
    'EV_KAN_ROBBED': ('actor', 'meld_id', 'window_id'),
    'EV_DORA_REVEAL': ('tile',), 'EV_RIICHI_DECLARE': ('actor',),
    'EV_RIICHI_ACCEPT': ('actor',), 'EV_RESPONSE_OPEN': ('window',),
    'EV_RESPONSE_CLOSE': ('window_id', 'resolution', 'result'),
    'EV_WIN': ('win_group',), 'EV_DRAW_END': ('result',),
    'EV_ABORT_END': ('reason', 'actor', 'result'), 'EV_NAGASHI': ('result',),
    'EV_SETTLEMENT': ('settlement',), 'EV_REVEAL': ('reveal',),
    'EV_ROUND_END': ('next_state',), 'EV_GAME_END': ('final_result',),
    'EV_PROTOCOL_NOOP': ('metadata',), 'EV_EXTENSION': ('extension',),
}

# Frozen on-disk identity; package renames must not change codec bindings.
BINDING = {
    'name': 'majang-4p-v2',
    'vocab_sha256': '675c528d06fa8c9c36f49205f8380e006541abe0d468c191e39f321f2956fb0f',
    'sequence_schema_version': 'majang-sequence/1.3',
    'ir_schema_version': 'majang-ir/2', 'task': 'policy-latest-hand/2',
}


class MahjongTokenizer:
    vocab_size = 1024
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def __init__(self, *, _resource=None):
        resource = _resource or files('mahjong.tokenizer').joinpath('resources')
        self._resource = resource
        raw = resource.joinpath('vocab-v2.tsv').read_bytes()
        manifest = json.loads(resource.joinpath('vocab-v2.manifest.json').read_text())
        self.validate_binding(manifest)
        if hashlib.sha256(raw).hexdigest() != BINDING['vocab_sha256']:
            fail('NONCANONICAL', 'vocabulary digest mismatch')
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8')), delimiter='\t'))
        if ([int(r['id']) for r in rows] != list(range(1024))
                or len({r['token'] for r in rows}) != 1024
                or dict(Counter(r['category'] for r in rows)) != manifest['categories']
                or sum(r['status'] == 'active' for r in rows) != 832):
            fail('NONCANONICAL', 'vocabulary mapping/count mismatch')
        if (manifest.get('reserved_count') != 192
                or [manifest.get(k) for k in ('vocab_size', 'pad_token_id', 'bos_token_id', 'eos_token_id')] != [1024, 0, 1, 2]):
            fail('NONCANONICAL', 'vocabulary metadata mismatch')
        self._rows = rows
        self._ids = {r['token'][1:-1]: int(r['id']) for r in rows if r['status'] == 'active'}
        self._fields = {name[2:].lower(): name for name in self._ids if name.startswith('F_')}
        self.vocab_sha256 = BINDING['vocab_sha256']

    @staticmethod
    def validate_binding(binding):
        """Lexical/task identity only; not the full training/checkpoint binding."""
        if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in BINDING.items()):
            fail('NONCANONICAL', 'incompatible vocabulary/schema/IR/task binding')

    def token_id(self, name):
        if not isinstance(name, str) or name not in self._ids:
            fail('TYPE_MISMATCH', f'unknown or reserved token: {name!r}')
        return self._ids[name]

    def token_name(self, token_id):
        if type(token_id) is not int or not 0 <= token_id < self.vocab_size:
            fail('TYPE_MISMATCH', 'invalid token ID')
        row = self._rows[token_id]
        if row['status'] != 'active':
            fail('TYPE_MISMATCH', 'reserved token ID')
        return row['token'][1:-1]

    def _atom(self, value):
        return self.token_id(value.name)

    def tile_key(self, value):
        name = value.name.removeprefix('TILE_')
        if name == 'HIDDEN':
            return (34, 0)
        if name[-1] == 'r':
            return ('mps'.index(name[1]) * 9 + 4, 1)
        if name[-1] in 'mps':
            return ('mps'.index(name[1]) * 9 + int(name[0]) - 1, 0)
        return (27 + 'ESWNPFC'.index(name), 0)

    def _key(self, key):
        if type(key) is not str:
            fail('TYPE_MISMATCH', 'object key must be string')
        if key in self._fields:
            return (0, self.token_id(self._fields[key]))
        try:
            return (1, key.encode('utf-8'))
        except UnicodeError:
            fail('TYPE_MISMATCH', 'invalid Unicode key')

    def encode_value(self, value):
        """Encode already prepared values; this does not validate business input."""
        return self._encode(value, 0)

    def _encode(self, value, depth):
        if depth > 128:
            fail('TYPE_MISMATCH', 'container nesting exceeds 128')
        t = self.token_id
        enc = lambda v: self._encode(v, depth + 1)
        if value is None:
            return [t('NULL')]
        if type(value) is bool:
            return [t('TRUE' if value else 'FALSE')]
        if type(value) is int:
            if 0 <= value <= 127:
                return [t(f'U{value}')]
            return [t('INT'), t('NEG' if value < 0 else 'POS'),
                    *(t(f'U{digit}') for digit in str(abs(value))), t('END_INT')]
        if isinstance(value, Atom):
            return [self._atom(value)]
        if isinstance(value, Decimal):
            if not value.is_finite():
                fail('TYPE_MISMATCH', 'decimal must be finite')
            sign, digits, exponent = value.as_tuple()
            coefficient = 0
            for digit in digits:
                coefficient = coefficient * 10 + digit
            coefficient *= -1 if sign else 1
            if exponent > 0:
                coefficient *= 10 ** exponent
            scale = max(0, -exponent)
            if coefficient == 0:
                scale = 0
            while scale and coefficient % 10 == 0:
                coefficient //= 10
                scale -= 1
            return [t('DECIMAL'), *enc(coefficient), *enc(scale)]
        if type(value) in (str, bytes):
            is_text = type(value) is str
            try:
                data = value.encode('utf-8') if is_text else value
            except UnicodeError:
                fail('TYPE_MISMATCH', 'invalid Unicode string')
            tag = 'UTF8' if is_text else 'BYTES'
            return [t(tag), *(t(f'BYTE_{b:02X}') for b in data), t('END_' + tag)]
        if type(value) is list:
            return [t('LIST'), *(i for v in value for i in enc(v)), t('END_LIST')]
        if type(value) is dict:
            result = [t('OBJ')]
            for key in sorted(value, key=self._key):
                result.extend([t(self._fields[key])] if key in self._fields else enc(key))
                result.extend(enc(value[key]))
            return [*result, t('END_OBJ')]
        if isinstance(value, Extension):
            if type(value.namespace) is not str or type(value.version) is not int or type(value.payload) is not dict:
                fail('TYPE_MISMATCH', 'invalid extension')
            return [t('EXT'), *enc(value.namespace), *enc(value.version), *enc(value.payload), t('END_EXT')]
        if isinstance(value, Action):
            return self.encode_action(value)
        if isinstance(value, Event):
            return self._encode_event(value, depth + 1)
        if isinstance(value, ResponseBlock):
            return [t('RESPONSE'), *enc(value.payload), t('END_RESPONSE')]
        fail('TYPE_MISMATCH', f'unsupported value type {type(value).__name__}')

    def decode_value(self, ids):
        reader = _Reader(self, ids)
        value = reader.value()
        reader.finish()
        return value

    def _action(self, action):
        # Sorting is canonical representation, not a legality/schema check.
        args = list(action.args)
        if action.kind in ('A_CHI', 'A_PON', 'A_DAIMINKAN'):
            args[2:] = sorted(args[2:], key=self.tile_key)
        elif action.kind == 'A_ANKAN':
            args.sort(key=self.tile_key)
        return Action(action.kind, tuple(args))

    def encode_action(self, action):
        """Standalone ACT archive wrapper; never append this to a policy prefix."""
        return [self.token_id('ACT'), *self.encode_action_suffix(action)[:-1], self.token_id('END_ACT')]

    def encode_action_suffix(self, action):
        """Payload + END_DECISION, to append after the prefix's DECISION."""
        action = self._action(action)
        return [self.token_id(action.kind), *(i for arg in action.args for i in self.encode_value(arg)), self.token_id('END_DECISION')]

    def decode_action(self, ids):
        reader = _Reader(self, ids)
        reader.expect('ACT')
        action = reader.action('END_ACT')
        reader.finish()
        return action

    def decode_action_suffix(self, ids):
        reader = _Reader(self, ids)
        action = reader.action('END_DECISION')
        reader.finish()
        return action

    def encode_header(self, header):
        return [self.token_id('HEADER'), *self.encode_value(header), self.token_id('END_HEADER')]

    def decode_header(self, ids):
        reader = _Reader(self, ids)
        reader.expect('HEADER')
        header = reader.value()
        reader.expect('END_HEADER')
        reader.finish()
        return header

    def _encode_event(self, event, depth=0):
        fields = EVENT_FIELDS[event.kind]
        payload = list(event.payload)
        if 'consumed' in fields:
            index = fields.index('consumed')
            payload[index] = sorted(payload[index], key=self.tile_key)
        t = self.token_id
        return [t('EV'), t(event.kind), *self._encode(event.event_id, depth),
                *(i for value in payload for i in self._encode(value, depth)), t('END_EV')]

    def encode_event(self, event):
        """Position-encode an already prepared event, preserving its supplied ID."""
        return self._encode_event(event)

    def decode_event(self, ids):
        reader = _Reader(self, ids)
        reader.expect('EV')
        event = reader.event()
        reader.finish()
        return event

    def encode_hand_state(self, state):
        t = self.token_id
        result = [t('HAND_STATE')]
        for tile in sorted(state.hands, key=self.tile_key):
            result.extend(self.encode_value(tile))
        result.append(t('HAND_MELDS_SEP'))
        for meld in sorted(state.melds, key=lambda item: item['created_event']):
            # Copy before canonicalization; never mutate the upstream state.
            value = {**meld, 'tiles': sorted(meld['tiles'], key=self.tile_key)}
            result.extend(self.encode_value(value))
        return [*result, t('END_HAND_STATE')]

    def decode_hand_state(self, ids):
        reader = _Reader(self, ids)
        state = reader.hand_state()
        reader.finish()
        return state

    def encode_policy_prefix(self, sequence):
        """Rebuild from preprocessed history and current state; no input checks.

        Caller supplies the already filtered/projected history. No event is
        added, removed, renumbered or resolved against a reference table here.
        """
        t = self.token_id
        result = [t('BOS'), t('FORMAT_V1'), t('TASK_POLICY')]
        result.extend(self.encode_header(sequence.header))
        result.extend([t('HISTORY'), t('FULL_HISTORY')])
        for event in sequence.visible_history:
            result.extend(self.encode_event(event))
        result.append(t('END_HISTORY'))
        result.extend(self.encode_hand_state(sequence.hand_state))
        return [*result, t('DECISION')]

    def decode_policy_prefix(self, ids):
        reader = _Reader(self, ids)
        sequence = reader.policy_input()
        reader.finish()
        return sequence

    def encode_policy_sequence(self, sequence, action, *, eos=False):
        result = [*self.encode_policy_prefix(sequence), *self.encode_action_suffix(action)]
        if eos:
            result.append(self.eos_token_id)
        return result

    def decode_policy_sequence(self, ids):
        """Parse a prefix or a complete offline example (optional terminal EOS)."""
        reader = _Reader(self, ids)
        sequence = reader.policy_input()
        action = None
        eos = False
        if reader.pos < len(reader.names):
            action = reader.action('END_DECISION')
        if reader.pos < len(reader.names):
            reader.expect('EOS')
            eos = True
        reader.finish()
        return PolicySequence(sequence, action, eos)

    def save(self, directory):
        """Export codec resources only, without rules registry/model/training state."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('vocab-v2.tsv', 'vocab-v2.manifest.json', 'codec-spec-v2.json'):
            (directory / name).write_bytes(self._resource.joinpath(name).read_bytes())
        binding = {**BINDING, 'codec_spec_sha256': self.codec_spec_sha256}
        (directory / 'binding.json').write_text(
            json.dumps(binding, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')

    @property
    def codec_spec_sha256(self):
        return hashlib.sha256(self._resource.joinpath('codec-spec-v2.json').read_bytes()).hexdigest()

    @classmethod
    def load_local(cls, directory):
        """Load a local codec package compatible with this implementation."""
        directory = Path(directory)
        binding = json.loads((directory / 'binding.json').read_text(encoding='utf-8'))
        cls.validate_binding(binding)
        spec = (directory / 'codec-spec-v2.json').read_bytes()
        bundled = files('mahjong.tokenizer').joinpath('resources/codec-spec-v2.json').read_bytes()
        if spec != bundled or hashlib.sha256(spec).hexdigest() != binding.get('codec_spec_sha256'):
            fail('NONCANONICAL', 'incompatible codec resource semantics')
        return cls(_resource=directory)


class _Reader:
    def __init__(self, codec, ids):
        if type(ids) not in (list, tuple):
            fail('TYPE_MISMATCH', 'expected token ID sequence')
        self.codec = codec
        self.names = [codec.token_name(i) for i in ids]
        self.pos = 0

    def peek(self):
        if self.pos >= len(self.names):
            fail('TYPE_MISMATCH', 'incomplete value or container')
        return self.names[self.pos]

    def pop(self):
        name = self.peek()
        self.pos += 1
        return name

    def expect(self, name):
        if self.pop() != name:
            fail('TYPE_MISMATCH', f'expected {name}')

    def finish(self):
        if self.pos != len(self.names):
            fail('TYPE_MISMATCH', 'trailing tokens')

    def action(self, end, depth=0):
        kind = self.pop()
        if kind not in ACTION_SHAPES:
            fail('TYPE_MISMATCH', 'expected action kind')
        args = tuple(self.value(depth + 1) for _ in ACTION_SHAPES[kind])
        self.expect(end)
        return Action(kind, args)

    def event(self, depth=0):
        kind = self.pop()
        try:
            fields = EVENT_FIELDS[kind]
        except KeyError:
            fail('TYPE_MISMATCH', 'expected event kind')
        event_id = self.value(depth + 1)
        payload = tuple(self.value(depth + 1) for _ in fields)
        self.expect('END_EV')
        return Event(kind, event_id, payload)

    def hand_state(self):
        self.expect('HAND_STATE')
        hands = []
        while self.peek() != 'HAND_MELDS_SEP':
            hands.append(self.value())
        self.pop()
        melds = []
        while self.peek() != 'END_HAND_STATE':
            melds.append(self.value())
        self.pop()
        return HandMeldState(hands, melds)

    def policy_input(self):
        for token in ('BOS', 'FORMAT_V1', 'TASK_POLICY', 'HEADER'):
            self.expect(token)
        header = self.value()
        for token in ('END_HEADER', 'HISTORY', 'FULL_HISTORY'):
            self.expect(token)
        events = []
        while self.peek() != 'END_HISTORY':
            self.expect('EV')
            events.append(self.event())
        self.pop()
        hand_state = self.hand_state()
        self.expect('DECISION')
        return PolicySequenceInput(header, events, hand_state)

    def value(self, depth=0):
        if depth > 128:
            fail('TYPE_MISMATCH', 'container nesting exceeds 128')
        name = self.pop()
        sub = lambda: self.value(depth + 1)
        if name in ('NULL', 'TRUE', 'FALSE'):
            return {'NULL': None, 'TRUE': True, 'FALSE': False}[name]
        if name.startswith('U') and name[1:].isdigit():
            return int(name[1:])
        if name == 'INT':
            sign = self.pop()
            if sign not in ('POS', 'NEG'):
                fail('TYPE_MISMATCH', 'expected integer sign')
            digits = []
            while self.peek() != 'END_INT':
                digit = self.pop()
                if digit not in tuple(f'U{i}' for i in range(10)):
                    fail('TYPE_MISMATCH', 'expected single decimal digit')
                digits.append(digit[1:])
            self.pop()
            if not digits:
                fail('TYPE_MISMATCH', 'empty integer')
            result = int(''.join(digits)) * (-1 if sign == 'NEG' else 1)
            return result
        if name == 'DECIMAL':
            coefficient, scale = sub(), sub()
            if type(coefficient) is not int or type(scale) is not int or scale < 0:
                fail('TYPE_MISMATCH', 'invalid decimal parts')
            return Decimal((int(coefficient < 0), tuple(int(x) for x in str(abs(coefficient))), -scale))
        if name in ('UTF8', 'BYTES'):
            data = bytearray()
            while self.peek() != 'END_' + name:
                byte = self.pop()
                if not byte.startswith('BYTE_'):
                    fail('TYPE_MISMATCH', 'expected byte token')
                data.append(int(byte[5:], 16))
            self.pop()
            try:
                return data.decode('utf-8') if name == 'UTF8' else bytes(data)
            except UnicodeError:
                fail('TYPE_MISMATCH', 'invalid UTF-8')
        if name == 'LIST':
            result = []
            while self.peek() != 'END_LIST':
                result.append(sub())
            self.pop()
            return result
        if name == 'OBJ':
            result = {}
            while self.peek() != 'END_OBJ':
                key = self.pop()[2:].lower() if self.peek().startswith('F_') else sub()
                if key in result:
                    fail('TYPE_MISMATCH', 'duplicate key cannot be represented by dict')
                result[key] = sub()
            self.pop()
            return result
        if name == 'EXT':
            namespace, version, payload = sub(), sub(), sub()
            self.expect('END_EXT')
            if type(namespace) is not str or type(version) is not int or type(payload) is not dict:
                fail('TYPE_MISMATCH', 'invalid extension')
            return Extension(namespace, version, payload)
        if name == 'ACT':
            return self.action('END_ACT', depth)
        if name == 'EV':
            return self.event(depth)
        if name == 'RESPONSE':
            payload = sub()
            self.expect('END_RESPONSE')
            return ResponseBlock(payload)
        category = self.codec._rows[self.codec.token_id(name)]['category']
        if category in ('tile', 'seat', 'enum', 'yaku') or name in ('UNKNOWN', 'HIDDEN'):
            return Atom(name)
        fail('TYPE_MISMATCH', f'expected value, got {name}')
