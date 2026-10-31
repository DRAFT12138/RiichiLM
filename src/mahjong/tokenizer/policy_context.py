"""Cross-round policy container; reuses the frozen vocabulary and value codec.

Business validation and visibility projection belong to policy.context. This
module only serializes already prepared values, without changing text/3.
"""
from dataclasses import dataclass

from .codec import MahjongTokenizer, PolicySequenceInput, _Reader

# Serialized schema identities are independent of the Python import name.
LEGACY_SCHEMA = 'majang-policy-ntp/v1'
CONTEXT_SCHEMA = 'majang-policy-ntp/v2'
PREVIOUS_ROUNDS = 3
CONTEXT_BINDING = {
    'schema': 'policy-round-context/1',
    'previous_rounds': PREVIOUS_ROUNDS,
    'order': 'oldest-first',
    'missing': 'NULL',
    'overflow': 'error',
    'history': 'published-visible-events/1',
}


@dataclass(frozen=True)
class RoundContext:
    header: dict
    visible_history: list
    complete: bool


@dataclass(frozen=True)
class PolicyContextInput(PolicySequenceInput):
    previous_rounds: tuple[RoundContext | None, ...] = ()


class PolicyContextCodec(MahjongTokenizer):
    def encode_previous_rounds(self, previous_rounds):
        t = self.token_id
        previous = [t('LIST')]
        for round_ in previous_rounds:
            if round_ is None:
                previous.append(t('NULL'))
                continue
            previous.append(t('TRUE' if round_.complete else 'FALSE'))
            previous.extend(self.encode_header(round_.header))
            previous.extend((t('HISTORY'), t('FULL_HISTORY')))
            for event in round_.visible_history:
                previous.extend(self.encode_event(event))
            previous.append(t('END_HISTORY'))
        previous.append(t('END_LIST'))
        return previous

    def encode_policy_prefix(self, sequence):
        current = super().encode_policy_prefix(sequence)
        return [*current[:3], *self.encode_previous_rounds(sequence.previous_rounds), *current[3:]]

    def decode_policy_prefix(self, ids):
        reader = _Reader(self, ids)
        for name in ('BOS', 'FORMAT_V1', 'TASK_POLICY', 'LIST'):
            reader.expect(name)
        previous = []
        while reader.peek() != 'END_LIST':
            if reader.peek() == 'NULL':
                reader.pop()
                previous.append(None)
                continue
            complete = reader.peek() == 'TRUE'
            reader.expect('TRUE' if complete else 'FALSE')
            reader.expect('HEADER')
            header = reader.value()
            for name in ('END_HEADER', 'HISTORY', 'FULL_HISTORY'):
                reader.expect(name)
            events = []
            while reader.peek() != 'END_HISTORY':
                reader.expect('EV')
                events.append(reader.event())
            reader.pop()
            previous.append(RoundContext(header, events, complete))
        reader.pop()
        current = super().decode_policy_prefix([*ids[:3], *ids[reader.pos:]])
        return PolicyContextInput(current.header, current.visible_history,
                                  current.hand_state, tuple(previous))


def codec_for_schema(schema):
    if schema == CONTEXT_SCHEMA:
        return PolicyContextCodec()
    if schema == LEGACY_SCHEMA:
        return MahjongTokenizer()
    raise ValueError(f'unsupported policy input schema: {schema}')
