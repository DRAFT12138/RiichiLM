"""Compact view encoding and joint index/IR boundary checks."""
from bisect import bisect_right
from dataclasses import dataclass
import struct
import zlib

from mahjong.tokenizer import MahjongTokenizer
from .common import canonical, require
from .models import VISIBLE
from .validation import check_event, check_event_references, check_hand, check_header, check_ir, natural

ARRAY_WIDTHS = {'header': 2, 'history': 2, 'history_offsets': 8,
                'hand_blocks': 2, 'hand_offsets': 8, 'state_event_ids': 8,
                'anchors': 8}


def pack(values, width):
    require(width in (2, 8), 'STORAGE_CORRUPT', 'array width')
    require(all(type(i) is int and 0 <= i < 2**(8 * width) for i in values),
            'STORAGE_CORRUPT', 'array integer range')
    raw = struct.pack(f'<{len(values)}' + ('H' if width == 2 else 'Q'), *values)
    return zlib.compress(raw, 1), len(raw), len(values)


def unpack(blob, byte_count, count, width, *, max_bytes=64 * 1024 * 1024):
    require(type(byte_count) is int and 0 <= byte_count <= max_bytes and
            type(count) is int and count >= 0 and byte_count == count * width,
            'STORAGE_CORRUPT', 'array metadata')
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(blob, byte_count + 1)
    except zlib.error as error:
        require(False, 'STORAGE_CORRUPT', str(error))
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
            len(raw) == byte_count, 'STORAGE_CORRUPT', 'compressed array length')
    return list(struct.unpack(f'<{count}' + ('H' if width == 2 else 'Q'), raw))


@dataclass
class EncodedView:
    viewer: int
    valid_through_event_id: int
    invalid_reason: dict | None
    arrays: dict[str, list[int]]

    @property
    def valid_event_count(self):
        return len(self.arrays['history_offsets']) - 1

    @property
    def anchor_count(self):
        return len(self.arrays['anchors']) // 5


def check_layout(view, event_ids, *, codec=None):
    """Only persisted integrity; no rules engine or frozen-state reconstruction."""
    a = view.arrays
    require(a.keys() == ARRAY_WIDTHS.keys(), 'STORAGE_CORRUPT', 'view arrays')
    natural(view.valid_through_event_id, 'valid_through_event_id')
    require(all(type(i) is int and i >= 0 for i in event_ids) and
            all(x < y for x, y in zip(event_ids, event_ids[1:])), 'STORAGE_CORRUPT', 'shared IDs')
    v = bisect_right(event_ids, view.valid_through_event_id)
    require(len(a['history_offsets']) == v + 1, 'STORAGE_CORRUPT', 'effective history boundary')
    for offsets, tokens in (('history_offsets', 'history'), ('hand_offsets', 'hand_blocks')):
        values = a[offsets]
        require(values and values[0] == 0 and values[-1] == len(a[tokens]) and
                all(x < y for x, y in zip(values, values[1:])), 'STORAGE_CORRUPT', offsets)
    m = len(a['hand_offsets']) - 1
    require(len(a['state_event_ids']) == m and
            all(x < y for x, y in zip(a['state_event_ids'], a['state_event_ids'][1:])),
            'STORAGE_CORRUPT', 'hand block positions')
    require(len(a['anchors']) % 5 == 0, 'STORAGE_CORRUPT', 'anchor arity')
    windows, first_use = set(), {}
    previous_cutoff = -1
    for start in range(0, len(a['anchors']), 5):
        window, trigger, cutoff, count, index = a['anchors'][start:start + 5]
        require(window not in windows and trigger <= cutoff <= view.valid_through_event_id and
                cutoff >= previous_cutoff and count == bisect_right(event_ids, cutoff) and
                count <= v and index < m and a['state_event_ids'][index] <= cutoff,
                'STORAGE_CORRUPT', 'anchor boundary/reference')
        windows.add(window)
        previous_cutoff = cutoff
        first_use.setdefault(index, cutoff)
    require(set(first_use) == set(range(m)) and
            [first_use[i] for i in range(m)] == a['state_event_ids'],
            'STORAGE_CORRUPT', 'unused block/incorrect first use')
    if codec is not None:
        for name in ('header', 'history', 'hand_blocks'):
            for token in a[name]:
                codec.token_name(token)
        codec.decode_header(a['header'])
        for i in range(v):
            lo, hi = a['history_offsets'][i:i + 2]
            require(codec.decode_event(a['history'][lo:hi]).event_id == event_ids[i],
                    'STORAGE_CORRUPT', 'history event ID')
        for i in range(m):
            lo, hi = a['hand_offsets'][i:i + 2]
            codec.decode_hand_state(a['hand_blocks'][lo:hi])


def encode_view(view, round_record, *, codec=None):
    codec = codec or MahjongTokenizer()
    require(type(view.viewer) is int and 0 <= view.viewer < 4, 'TYPE_MISMATCH', 'viewer')
    check_header(view.header)
    check_ir(round_record)
    ledger = {r['event_id']: r for r in round_record.ledger}
    require(len(ledger) == len(round_record.ledger) and
            list(ledger) == list(range(len(ledger))) and ledger[0]['kind'] == 'EV_ROUND_START',
            'INVALID_REFERENCE', 'complete IR IDs')
    require(round_record.event_ids == [i for i, r in ledger.items() if r['kind'] in VISIBLE],
            'INVALID_REFERENCE', 'shared visible IDs')
    require(view.valid_through_event_id in ledger, 'INVALID_REFERENCE', 'valid IR boundary')
    require([e.event_id for e in view.history] ==
            round_record.event_ids[:bisect_right(round_record.event_ids, view.valid_through_event_id)],
            'INVALID_REFERENCE', 'history must be contiguous shared prefix')
    require(len(view.history) >= 1 and view.history[0].kind == 'EV_DORA_REVEAL' and
            view.history[0].event_id == 1, 'NONINITIAL_VALUE', 'initial dora')
    history, offsets = [], [0]
    for event in view.history:
        payload=check_event(event, viewer=view.viewer)
        check_event_references(event,payload,viewer=view.viewer,ledger=ledger)
        require(ledger[event.event_id]['kind'] == event.kind, 'INVALID_REFERENCE', 'IR/history kind')
        history.extend(codec.encode_event(event))
        offsets.append(len(history))
    arrays = dict(header=codec.encode_header(view.header), history=history,
                  history_offsets=offsets, hand_blocks=[], hand_offsets=[0],
                  state_event_ids=[], anchors=[])
    windows = {w['window_id']: w for w in round_record.windows}
    previous = None
    for anchor in view.anchors:
        natural(anchor.cutoff_event_id, 'cutoff')
        window = windows.get(anchor.window_id)
        require(window is not None and window['trigger_id'] == anchor.trigger_id and
                window['cutoff_event_id'] == anchor.cutoff_event_id and
                view.viewer in window['viewers'] and anchor.cutoff_event_id in ledger,
                'INVALID_REFERENCE', 'anchor/frozen window')
        require(anchor.cutoff_event_id <= view.valid_through_event_id,
                'INVALID_REFERENCE', 'anchor outside effective boundary')
        check_hand(anchor.state, viewer=view.viewer, cutoff=anchor.cutoff_event_id, ledger=ledger,
                   aka_counts=round_record.aka_counts)
        block = codec.encode_hand_state(anchor.state)
        if block != previous:
            arrays['hand_blocks'].extend(block)
            arrays['hand_offsets'].append(len(arrays['hand_blocks']))
            arrays['state_event_ids'].append(anchor.cutoff_event_id)
            previous = block
        index = len(arrays['state_event_ids']) - 1
        arrays['anchors'].extend((anchor.window_id, anchor.trigger_id, anchor.cutoff_event_id,
                                 bisect_right(round_record.event_ids, anchor.cutoff_event_id), index))
    encoded = EncodedView(view.viewer, view.valid_through_event_id, view.invalid_reason, arrays)
    check_layout(encoded, round_record.event_ids, codec=codec)
    check_frozen_blocks(encoded, view.anchors, codec)
    return encoded


def check_frozen_blocks(encoded, anchors, codec):
    """Writer-only semantic proof. The reader cannot reconstruct frozen state."""
    arrays=encoded.arrays
    require(encoded.anchor_count==len(anchors),'INVALID_REFERENCE','frozen anchor count')
    for i,anchor in enumerate(anchors):
        window,trigger,cutoff,count,index=arrays['anchors'][5*i:5*i+5]
        require((window,trigger,cutoff)==(anchor.window_id,anchor.trigger_id,anchor.cutoff_event_id),
                'INVALID_REFERENCE','frozen cutoff identity')
        lo,hi=arrays['hand_offsets'][index:index+2]
        require(arrays['hand_blocks'][lo:hi]==codec.encode_hand_state(anchor.state),
                'INVALID_REFERENCE','stale frozen hand block')


def assemble_prefix(view, window_id, codec):
    a = view.arrays
    for i in range(0, len(a['anchors']), 5):
        window, trigger, cutoff, count, block = a['anchors'][i:i + 5]
        if window == window_id:
            lo, hi = a['hand_offsets'][block:block + 2]
            t = codec.token_id
            ids = [t('BOS'), t('FORMAT_V1'), t('TASK_POLICY'), *a['header'],
                   t('HISTORY'), t('FULL_HISTORY'),
                   *a['history'][:a['history_offsets'][count]], t('END_HISTORY'),
                   *a['hand_blocks'][lo:hi], t('DECISION')]
            return ids, dict(window_id=window, trigger_id=trigger, cutoff_event_id=cutoff)
    raise KeyError(window_id)
