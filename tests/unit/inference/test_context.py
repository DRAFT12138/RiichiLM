"""Past-round boundaries, visibility, causal parity and strict task identity."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from mahjong.contracts import MahjongError
from mahjong.data.mahjong.cleaning.encoding import encode_view
from mahjong.inference.context_bridge import ContextDecisionBridge
from mahjong.policy.context import read_context_prefix, stored_round_context, validate_context
from mahjong.tokenizer import Atom, Event, MahjongTokenizer, ProtocolError
from mahjong.tokenizer.policy_context import PolicyContextCodec, PolicyContextInput, RoundContext
from tests.unit.inference.test_bridge import select

PROFILE = json.loads(Path('configs/inference-profile.json').read_text())


def snapshot_reader(b, records):
    class Reader:
        codec = MahjongTokenizer()
        def read_round(self, source, round_id): return records[round_id]
        def iter_round_refs(self, source): return iter((source, r) for r in sorted(records))
        def read_prefix(self, source, round_id, viewer, window):
            from mahjong.data.mahjong.cleaning.encoding import assemble_prefix, EncodedView
            view = records[round_id]['views'][viewer]
            return assemble_prefix(EncodedView(**view), window, self.codec)
    return Reader()


def store_current(b):
    record = deepcopy(b._builder.prepared())
    # Native transitions are zero-based; file readers use one-based raw lines.
    for entry in record.ledger:
        for ref in entry['source_refs']:
            ref['raw_line'] += 1
    for fact in record.facts.get('furiten_transitions', []):
        fact['source_refs'] = [line + 1 for line in fact['source_refs']]
    return {'views': [asdict(encode_view(view, record)) for view in record.views],
            'facts': {'outcomes': []}}


def test_context_codec_is_distinct_and_preserves_one_current_hand():
    b = ContextDecisionBridge(PROFILE, model_binding_digest='test')
    b.reset(seed=42)
    current = b.snapshot(0).sequence
    past = RoundContext(current.header, current.visible_history, True)
    seq = replace(current, previous_rounds=(None, replace(past, complete=False), past))
    codec = PolicyContextCodec()
    ids = codec.encode_policy_prefix(seq)
    assert codec.decode_policy_prefix(ids) == seq
    assert ids.count(codec.token_id('HAND_STATE')) == 1
    assert ids.count(codec.token_id('DECISION')) == 1
    with pytest.raises(ProtocolError): MahjongTokenizer().decode_policy_prefix(ids)
    with pytest.raises(ProtocolError): codec.decode_policy_prefix(MahjongTokenizer().encode_policy_prefix(current))
    with pytest.raises(MahjongError): validate_context(replace(current, previous_rounds=(past,)*4))
    with pytest.raises(MahjongError):
        validate_context(replace(current, previous_rounds=(replace(past,
            visible_history=[past.visible_history[0],past.visible_history[0]]),)))


def test_online_and_stored_inputs_match_across_five_boundaries_and_reset():
    b = ContextDecisionBridge(PROFILE, model_binding_digest='test')
    b.reset(seed=42)
    records = {}
    compared = 0
    for round_index in range(6):
        assert b._round == round_index
        records[round_index] = store_current(b)
        reader = snapshot_reader(b, records)
        for viewer in range(4):
            view = b.snapshot(viewer)
            if view is None: continue
            encoded, _ = read_context_prefix(reader, 'game', round_index, viewer,
                                             view.key.window.window_id)
            assert tuple(encoded) == view.prefix_ids
            seq = view.sequence
            assert len(seq.previous_rounds) == min(round_index, 3)
            assert seq.previous_rounds == tuple(stored_round_context(records[r], viewer)
                for r in range(max(0, round_index - 3), round_index))
            compared += 1
        # Fixed legal policy advances the environment without invoking a model.
        for _ in range(250):
            views = [v for p in range(4) if (v := b.snapshot(p)) is not None]
            if not views: break
            from riichienv import Phase
            if b._native.phase == Phase.WaitResponse:
                result = b.apply_responses(views[0].key.window, {v.key.viewer: select(b, v,
                    next(m for m in v.legal_moves if m.action.kind == 'A_PASS')) for v in views})
            else:
                view = views[0]
                move = next((m for m in view.legal_moves if m.action.kind == 'A_TSUMO_WIN'), None)
                move = move or next(m for m in view.legal_moves if m.action.kind == 'A_DISCARD')
                result = b.apply(select(b, view, move))
            if result.status != 'advanced': break
        assert b._status == 'round_ended'
        records[round_index] = store_current(b)
        b.next_round()
    assert compared >= 6
    b.reset(seed=42)
    assert b.snapshot(0).sequence.previous_rounds == ()


def test_missing_round_slots_and_truncated_history_cannot_be_filled_from_other_views():
    b = ContextDecisionBridge(PROFILE, model_binding_digest='test')
    b.reset(seed=42)
    record = store_current(b)
    # Different actors' draws remain hidden even in a past round. Only this
    # viewer's persisted projected history is consumed, never the four-view IR.
    r = stored_round_context(record, 1)
    draw = next(e for e in r.visible_history if e.kind == 'EV_DRAW')
    assert draw.payload[1] == Atom('TILE_HIDDEN')
    record['views'][1]['invalid_reason'] = {'code': 'truncated'}
    record['facts']['outcomes'] = [{'kind': 'EV_WIN', 'payload': 'future-private-result'}]
    assert stored_round_context(record, 1).complete is False
    records = {0: record, 4: record}
    reader = snapshot_reader(b, records)
    window = b.snapshot(0).key.window.window_id
    encoded, _ = read_context_prefix(reader, 'game', 4, 0, window)
    seq = PolicyContextCodec().decode_policy_prefix(encoded)
    assert seq.previous_rounds == (None,)*3  # The fourth previous round is outside the window.
    before = list(encoded)
    records[5] = deepcopy(record)
    assert read_context_prefix(reader, 'game', 4, 0, window)[0] == before
