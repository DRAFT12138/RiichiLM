"""Prepare causal, viewer-specific cross-round inputs outside the cleaner.

Only published, validated view prefixes are consumed. Missing rounds retain
their position; truncated rounds carry an explicit incomplete marker. The
side-channel outcome facts are not policy history.
"""
from mahjong.contracts import MahjongError
from functools import lru_cache
from mahjong.tokenizer import MahjongTokenizer
from mahjong.tokenizer.policy_context import (
    PREVIOUS_ROUNDS, PolicyContextInput, PolicyContextCodec, RoundContext,
)

def stored_round_context(record, viewer, codec=None):
    codec = codec or MahjongTokenizer()
    view = next((v for v in record['views'] if v['viewer'] == viewer), None)
    if view is None:
        return None
    arrays = view['arrays']
    offsets = arrays['history_offsets']
    events = [codec.decode_event(arrays['history'][lo:hi])
              for lo, hi in zip(offsets, offsets[1:])]
    complete = view['invalid_reason'] is None
    return RoundContext(codec.decode_header(arrays['header']), events, complete)


def validate_previous_rounds(previous_rounds):
    if len(previous_rounds) > PREVIOUS_ROUNDS:
        raise MahjongError('CONTRACT_MISMATCH',
                          f'policy context contains more than {PREVIOUS_ROUNDS} previous rounds')
    for round_ in previous_rounds:
        if round_ is None:
            continue
        ids = [e.event_id for e in round_.visible_history]
        if any(a >= b for a, b in zip(ids, ids[1:])):
            raise MahjongError('CONTRACT_MISMATCH', 'past-round events must be in strictly increasing order')


def validate_context(sequence):
    validate_previous_rounds(sequence.previous_rounds)


@lru_cache(maxsize=1)
def _context_codec():
    return PolicyContextCodec()


def read_context_prefix(reader, source, round_id, viewer, window_id, *, round_ids=None, context_cache=None):
    codec = reader.codec
    current, identity = reader.read_prefix(source, round_id, viewer, window_id)
    key = (source, round_id, viewer)
    if context_cache is not None and key in context_cache:
        return [*current[:3], *context_cache[key], *current[3:]], identity
    available = set(round_ids if round_ids is not None else
                    (r for s, r in reader.iter_round_refs(source)))
    previous = tuple(stored_round_context(reader.read_round(source, r), viewer, codec)
                     if r in available else None
                     for r in range(max(0, round_id - PREVIOUS_ROUNDS), round_id))
    validate_previous_rounds(previous)
    encoded = _context_codec().encode_previous_rounds(previous)
    if context_cache is not None:
        context_cache[key] = encoded
    return [*current[:3], *encoded, *current[3:]], identity


def split_policy_prefix(reader, source, round_id, viewer, window_id, *, round_ids=None, context_cache=None):
    """Split at the current history boundary using the stored hand anchor.

    Past rounds contain END_HISTORY too; never search token values for the cut.
    Reuse the existing causal reader as the reference for the virtual sequence.
    """
    prefix, identity = read_context_prefix(reader, source, round_id, viewer, window_id,
        round_ids=round_ids, context_cache=context_cache)
    arrays = reader.read_view(source, round_id, viewer).arrays
    for offset in range(0, len(arrays['anchors']), 5):
        window, _, _, _, block = arrays['anchors'][offset:offset + 5]
        if window != window_id:
            continue
        lo, hi = arrays['hand_offsets'][block:block + 2]
        suffix = [reader.codec.token_id('END_HISTORY'), *arrays['hand_blocks'][lo:hi],
                  reader.codec.token_id('DECISION')]
        cut = len(prefix) - len(suffix)
        if cut < 1 or prefix[cut:] != suffix:
            raise MahjongError('CONTRACT_MISMATCH', 'stored hand boundary differs from policy prefix',
                               'preparing_data')
        return prefix[:cut], suffix, identity
    raise MahjongError('CONTRACT_MISMATCH', 'policy decision has no hand anchor', 'preparing_data')
