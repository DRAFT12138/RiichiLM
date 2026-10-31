"""Independent framing/state fixtures, not a rule replay oracle."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mahjong.tokenizer import (
    Action, Atom, Event, Extension, HandMeldState, MahjongTokenizer,
    PolicySequenceInput, ProtocolError, ResponseBlock,
)
from mahjong.tokenizer.codec import ACTION_SHAPES, EVENT_FIELDS

A = Atom


@pytest.fixture
def codec():
    return MahjongTokenizer()


def ids(codec, text):
    return [codec.token_id(x) for x in text.split()]


def make_header():
    return dict(mode=A('MODE_HANCHAN'), round_wind=A('ROUND_E'), round_number=1,
                honba=0, kyotaku=0, dealer=A('SELF'), scores=[25000] * 4,
                seat_winds=[A('TILE_' + x) for x in 'ESWN'])


def sample():
    return PolicySequenceInput(make_header(), [Event('EV_DORA_REVEAL', 1, (A('TILE_1m'),)),
        Event('EV_DRAW', 2, (A('SELF'), A('TILE_5mr'), A('DRAW_LIVE')))],
        HandMeldState([A('TILE_' + x) for x in
            ['1m','2m','3m','4m','5m','6m','1p','2p','3p','7s','8s','9s','E','5mr']], []))


def meld(kind='PON', *, created=8, updated=None, meld_id=0):
    ankan = kind == 'ANKAN'
    return dict(name=A('MELD_' + kind), actor=A('SELF'), meld_id=meld_id,
        tiles=[A('TILE_5m')] * (2 if kind in ('PON', 'CHI') else 3) + [A('TILE_5mr')],
        base_tile=A('TILE_5m'), **{'from': None if ankan else A('KAMICHA')},
        called_tile=None if ankan else A('TILE_5m'), trigger_id=None if ankan else 3,
        created_event=created, updated_event=updated)


# Exact positional payload strings are transcribed from report §7.1.
EVENT_CASES = [
    ('EV_DRAW', (A('SHIMOCHA'), A('TILE_HIDDEN'), A('DRAW_LIVE')), 'SHIMOCHA TILE_HIDDEN DRAW_LIVE'),
    ('EV_DISCARD', (A('SELF'), A('TILE_5mr'), A('TEDASHI'), True), 'SELF TILE_5mr TEDASHI TRUE'),
    ('EV_CHI', (A('SELF'), A('KAMICHA'), A('TILE_4m'), [A('TILE_6m'), A('TILE_5mr')], 2, 3), 'SELF KAMICHA TILE_4m LIST TILE_5mr TILE_6m END_LIST U2 U3'),
    ('EV_PON', (A('SELF'), A('TOIMEN'), A('TILE_5m'), [A('TILE_5mr'), A('TILE_5m')], 2, 3), 'SELF TOIMEN TILE_5m LIST TILE_5m TILE_5mr END_LIST U2 U3'),
    ('EV_DAIMINKAN', (A('SELF'), A('TOIMEN'), A('TILE_5m'), [A('TILE_5mr'), A('TILE_5m'), A('TILE_5m')], 2, 3), 'SELF TOIMEN TILE_5m LIST TILE_5m TILE_5m TILE_5mr END_LIST U2 U3'),
    ('EV_ANKAN_ATTEMPT', (A('SHIMOCHA'), A('TILE_5m'), [A('TILE_HIDDEN')] * 4, 2, 83), 'SHIMOCHA TILE_5m LIST TILE_HIDDEN TILE_HIDDEN TILE_HIDDEN TILE_HIDDEN END_LIST U2 U83'),
    ('EV_KAKAN_ATTEMPT', (A('SELF'), A('TILE_5mr'), 2, 83), 'SELF TILE_5mr U2 U83'),
    ('EV_KAN_COMMIT', (A('SELF'), 2), 'SELF U2'),
    ('EV_KAN_ROBBED', (A('SELF'), 2, 83), 'SELF U2 U83'),
    ('EV_DORA_REVEAL', (A('TILE_1m'),), 'TILE_1m'),
    ('EV_RIICHI_DECLARE', (A('SELF'),), 'SELF'),
    ('EV_RIICHI_ACCEPT', (A('SELF'),), 'SELF'),
    ('EV_RESPONSE_CLOSE', (14, A('RESOLVE_NO_CALL'), {}), 'U14 RESOLVE_NO_CALL OBJ END_OBJ'),
    ('EV_ABORT_END', (A('ABORT_KYUUSHU'), A('SELF'), {}), 'ABORT_KYUUSHU SELF OBJ END_OBJ'),
    ('EV_EXTENSION', (Extension('x', 1, {}),), 'EXT UTF8 BYTE_78 END_UTF8 U1 OBJ END_OBJ END_EXT'),
]
# Object internals are covered by base codecs; {} tests the event envelope only.
EVENT_CASES += [(kind, ({},), 'OBJ END_OBJ') for kind in (
    'EV_GAME_START','EV_ROUND_START','EV_RESPONSE_OPEN','EV_WIN','EV_DRAW_END',
    'EV_NAGASHI','EV_SETTLEMENT','EV_REVEAL','EV_ROUND_END','EV_GAME_END','EV_PROTOCOL_NOOP')]


@pytest.mark.parametrize('kind,payload,expected', EVENT_CASES)
def test_every_event_position_grammar(codec, kind, payload, expected):
    event = Event(kind, 20, payload)
    original = deepcopy(event)
    want = ids(codec, f'EV {kind} U20 {expected} END_EV')
    assert codec.encode_event(event) == want
    parsed = codec.decode_event(want)
    assert parsed.kind == kind and parsed.event_id == 20
    assert codec.encode_event(parsed) == want
    assert codec.decode_value(want) == parsed
    assert codec.encode_value(event) == want
    assert event == original


def test_nested_response(codec):
    block = ResponseBlock({'window_id': 14, 'responses': [
        {'actor': A('SELF'), 'action': Action('A_PASS'), 'status': A('ACCEPTED'),
         'knowledge': A('PROVENANCE_RECORDED')}], 'resolution': A('RESOLVE_NO_CALL')})
    raw = codec.encode_value(block)
    assert raw[0] == 43 and raw[-1] == 44
    assert codec.decode_value(raw) == block


def test_complete_independent_prefix_ids(codec):
    expected = [1,40,37,23,4,358,98,360,101,361,513,362,512,363,512,364,86,365,6]
    expected += [8,10,514,517,512,512,512,9] * 4
    expected += [7,366,6,75,76,77,78,7,5,24,27,42]
    expected += [35,235,513,48,36,35,226,514,86,82,121,36,28]
    expected += [640,48,49,50,51,52,82,53,57,58,59,72,73,74,75,641,642,29]
    sequence = sample()
    before = deepcopy(sequence)
    assert codec.encode_policy_prefix(sequence) == expected
    parsed = codec.decode_policy_prefix(expected)
    assert parsed.hand_state.hands[5] == A('TILE_5mr')
    assert parsed.visible_history == sequence.visible_history
    assert sequence == before
    assert codec.encode_policy_prefix(parsed) == expected


def string_tokens(value):
    return 'UTF8 ' + ' '.join(f'BYTE_{b:02X}' for b in value.encode()) + ' END_UTF8'


def test_meld_complete_independent_token_vector(codec):
    value = meld()
    expected = ('HAND_STATE TILE_5m TILE_5mr HAND_MELDS_SEP OBJ '
        'F_ACTOR SELF F_TILES LIST TILE_5m TILE_5m TILE_5mr END_LIST '
        'F_MELD_ID U0 F_FROM KAMICHA F_CALLED_TILE TILE_5m F_TRIGGER_ID U3 F_NAME MELD_PON '
        + string_tokens('base_tile') + ' TILE_5m '
        + string_tokens('created_event') + ' U8 '
        + string_tokens('updated_event') + ' NULL END_OBJ END_HAND_STATE')
    state = HandMeldState([A('TILE_5mr'), A('TILE_5m')], [value])
    before = deepcopy(state)
    assert codec.encode_hand_state(state) == ids(codec, expected)
    assert codec.decode_hand_state(ids(codec, expected)).melds == [value]
    assert state == before


@pytest.mark.parametrize('eos', [False, True])
@pytest.mark.parametrize('action', [Action('A_PASS'), Action('A_KAKAN', (2, A('TILE_5pr')))])
def test_complete_example_and_optional_eos(codec, action, eos):
    sequence = sample()
    raw = codec.encode_policy_sequence(sequence, action, eos=eos)
    assert raw.count(29) == 1 and raw.count(30) == 1
    assert raw[-1] == (2 if eos else 30)
    parsed = codec.decode_policy_sequence(raw)
    assert parsed.action == action and parsed.eos is eos
    assert codec.encode_policy_sequence(parsed.input, parsed.action, eos=parsed.eos) == raw
    prefix = codec.decode_policy_sequence(codec.encode_policy_prefix(sequence))
    assert prefix.action is None and prefix.eos is False


def test_rebuild_retains_business_history_and_no_old_state(codec):
    first = sample()
    old_bytes = codec.encode_policy_sequence(first, Action('A_DISCARD', (A('TILE_5mr'), A('TSUMOGIRI'))))
    events = first.visible_history + [Event('EV_DISCARD', 3, (A('SELF'), A('TILE_5mr'), A('TSUMOGIRI'), False)),
        Event('EV_DRAW', 6, (A('SHIMOCHA'), A('TILE_HIDDEN'), A('DRAW_LIVE'))),
        Event('EV_DISCARD', 7, (A('SHIMOCHA'), A('TILE_1p'), A('TEDASHI'), False))]
    current = PolicySequenceInput(first.header, events, HandMeldState(first.hand_state.hands[:-1], []))
    encoded = codec.encode_policy_prefix(current)
    old_bytes[:] = [0]  # There is no old-token-state input to the encoder.
    assert encoded == codec.encode_policy_prefix(current)
    decoded = codec.decode_policy_prefix(encoded)
    assert [event.event_id for event in decoded.visible_history] == [1,2,3,6,7]
    assert encoded.count(640) == encoded.count(641) == encoded.count(642) == encoded.count(29) == 1
    assert 30 not in encoded and 31 not in encoded and 25 not in encoded and 392 not in encoded
    assert [event.kind for event in decoded.visible_history].count('EV_DORA_REVEAL') == 1
    assert A('TILE_5mr') not in decoded.hand_state.hands


def test_before_after_chi_uses_delivered_state(codec):
    first = sample()
    trigger = Event('EV_DISCARD', 3, (A('KAMICHA'), A('TILE_4m'), A('TEDASHI'), False))
    response = PolicySequenceInput(first.header, [*first.visible_history, trigger], first.hand_state)
    chi = Event('EV_CHI', 6, (A('SELF'), A('KAMICHA'), A('TILE_4m'), [A('TILE_5mr'), A('TILE_6m')], 0, 3))
    state = meld('CHI', created=6)
    state.update(tiles=[A('TILE_4m'), A('TILE_5mr'), A('TILE_6m')], base_tile=A('TILE_4m'), called_tile=A('TILE_4m'))
    hands = list(first.hand_state.hands)
    hands.remove(A('TILE_5mr')); hands.remove(A('TILE_6m'))
    after = PolicySequenceInput(first.header, [*response.visible_history, chi], HandMeldState(hands, [state]))
    before_tokens = codec.encode_policy_prefix(response)
    after_tokens = codec.encode_policy_prefix(after)
    assert 228 not in before_tokens and after_tokens.count(228) == 1
    decoded = codec.decode_policy_prefix(after_tokens)
    assert len(decoded.hand_state.hands) == 12
    assert decoded.hand_state.melds[0]['created_event'] == 6
    assert codec.encode_policy_sequence(after, Action('A_DISCARD', (A('TILE_E'), A('TEDASHI'))))[-4:] == [257,75,120,30]


@pytest.mark.parametrize('kind,created,updated', [('ANKAN',23,None), ('DAIMINKAN',20,None), ('KAKAN',8,23)])
def test_kan_reference_values_and_order_are_preserved(codec, kind, created, updated):
    # IDs chosen independently per the adopted R2-TOK-02 mapping; no codec replay.
    kan = meld(kind, created=created, updated=updated)
    other = meld('PON', created=12, meld_id=1)
    state = HandMeldState([], [other, kan])
    parsed = codec.decode_hand_state(codec.encode_hand_state(state))
    assert [m['created_event'] for m in parsed.melds] == sorted([created,12])
    result = next(m for m in parsed.melds if m['meld_id'] == 0)
    assert result['created_event'] == created and result['updated_event'] == updated
    if kind == 'KAKAN':
        assert (result['from'], result['called_tile'], result['trigger_id']) == (A('KAMICHA'), A('TILE_5m'),3)
        assert parsed.melds[0]['meld_id'] == 0  # updated_event does not reorder it.


def test_attempt_commit_draw_cutpoints_and_robbed_branch(codec):
    first = sample()
    pon = meld('PON', created=8)
    attempt = Event('EV_KAKAN_ATTEMPT', 20, (A('SELF'), A('TILE_5mr'),0,83))
    commit = Event('EV_KAN_COMMIT', 23, (A('SELF'),0))
    draw = Event('EV_DRAW', 24, (A('SELF'), A('TILE_1s'), A('DRAW_RINSHAN')))
    histories = [[*first.visible_history,attempt], [*first.visible_history,attempt,commit],
                 [*first.visible_history,attempt,commit,draw]]
    for index, history in enumerate(histories):
        current = pon if index == 0 else meld('KAKAN', created=8, updated=23)
        seq = PolicySequenceInput(first.header, history, HandMeldState([], [current]))
        parsed = codec.decode_policy_prefix(codec.encode_policy_prefix(seq))
        assert parsed.visible_history == history
        assert parsed.hand_state.melds[0]['name'] == A('MELD_PON' if index == 0 else 'MELD_KAKAN')
    # A robbed result is an archive event, supplied separately, never synthesized.
    robbed = Event('EV_KAN_ROBBED', 23, (A('SELF'),0,83))
    assert codec.decode_event(codec.encode_event(robbed)) == robbed


def test_no_sequence_reference_visibility_or_header_validation(codec):
    # Intentionally invalid input proves no validator, rule engine or correction is hiding here.
    state = meld('KAKAN', created=999, updated=1000)
    sequence = PolicySequenceInput({'round_number': 99}, [
        Event('EV_DORA_REVEAL',9,(A('TILE_1m'),)), Event('EV_DORA_REVEAL',9,(A('TILE_1m'),))],
        HandMeldState([A('TILE_HIDDEN')], [state]))
    parsed = codec.decode_policy_prefix(codec.encode_policy_prefix(sequence))
    assert parsed == sequence
    assert not hasattr(codec, 'validate_sequence')
    assert not hasattr(codec, '_header')


def test_presupplied_projection_has_no_hidden_sidecar_dependency(codec):
    sequence = sample()
    sidecar = {'other_hands': ['secret'], 'future': ['win'], 'legal_actions': [Action('A_PASS')]}
    before = codec.encode_policy_prefix(sequence)
    sidecar.update(other_hands=['different'], future=['draw'], legal_actions=[])
    assert codec.encode_policy_prefix(sequence) == before
    # No sidecar parameter is accepted; public differences still change tokens.
    changed = PolicySequenceInput(sequence.header, [Event('EV_DORA_REVEAL',1,(A('TILE_2m'),))], sequence.hand_state)
    assert codec.encode_policy_prefix(changed) != before


def test_primitive_parse_does_not_enforce_canonical_input(codec):
    assert codec.decode_value(ids(codec, 'INT NEG U0 END_INT')) == 0
    assert codec.decode_value(ids(codec, 'INT POS U0 U1 END_INT')) == 1
    assert codec.decode_value(ids(codec, 'OBJ F_SCORES U1 F_MODE U2 END_OBJ')) == {'scores':1,'mode':2}


def test_codec_resource_spec_and_local_export(codec, tmp_path):
    codec.save(tmp_path)
    spec = json.loads((tmp_path / 'codec-spec-v2.json').read_text())
    assert spec['event_fields'] == {k:list(v) for k,v in EVENT_FIELDS.items()}
    assert spec['action_shapes'] == {k:list(v) for k,v in ACTION_SHAPES.items()}
    assert len(spec['event_fields']) == 26
    assert len(spec['policy_history']) == 11
    loaded = MahjongTokenizer.load_local(tmp_path)
    sequence = sample()
    raw = codec.encode_policy_sequence(sequence, Action('A_PASS'), eos=True)
    assert loaded.encode_policy_sequence(sequence, Action('A_PASS'), eos=True) == raw
    assert loaded.decode_policy_sequence(raw) == codec.decode_policy_sequence(raw)
    assert loaded.codec_spec_sha256 == codec.codec_spec_sha256


@pytest.mark.parametrize('name', ['vocab-v2.tsv','codec-spec-v2.json','binding.json'])
def test_incompatible_local_resources_rejected(codec, tmp_path, name):
    codec.save(tmp_path)
    file = tmp_path / name
    if name == 'binding.json':
        obj = json.loads(file.read_text()); obj['task'] = 'policy-current-private/1'
        file.write_text(json.dumps(obj))
    else:
        file.write_bytes(file.read_bytes() + b'\n')
    with pytest.raises(ProtocolError):
        MahjongTokenizer.load_local(tmp_path)


@pytest.mark.parametrize('seat', ['SELF','SHIMOCHA','TOIMEN','KAMICHA'])
def test_preprojected_seats_are_serialized_without_rotation(codec, seat):
    # Projection is supplied; codec must preserve all three seat occurrences.
    event = Event('EV_CHI', 9, (A(seat), A('TOIMEN'), A('TILE_4m'),
                              [A('TILE_5mr'), A('TILE_6m')], 2, 6))
    raw = codec.encode_event(event)
    assert raw == ids(codec, f'EV EV_CHI U9 {seat} TOIMEN TILE_4m LIST TILE_5mr TILE_6m END_LIST U2 U6 END_EV')
    assert codec.decode_event(raw) == event


def test_hidden_ankan_fixed_slots_and_multiple_meld_order(codec):
    # Archive/public projection uses four hidden slots; self policy input remains upstream's responsibility.
    hidden = meld('ANKAN', created=23, meld_id=2)
    hidden['tiles'] = [A('TILE_HIDDEN')] * 4
    normal = meld('PON', created=8)
    raw = codec.encode_hand_state(HandMeldState([], [hidden, normal]))
    decoded = codec.decode_hand_state(raw)
    assert [m['meld_id'] for m in decoded.melds] == [0,2]
    assert decoded.melds[1]['tiles'] == [A('TILE_HIDDEN')] * 4
    assert decoded.melds[1]['base_tile'] == A('TILE_5m')
    assert raw.count(codec.token_id('TILE_HIDDEN')) == 4


def test_unknown_response_stays_unknown_in_archive(codec):
    block = ResponseBlock({'window_id':14, 'responses':[
        {'actor':A('SELF'), 'action':A('UNKNOWN'), 'status':A('UNOBSERVED'),
         'knowledge':A('PROVENANCE_UNKNOWN')}], 'resolution':A('RESOLVE_NO_CALL')})
    raw = codec.encode_value(block)
    assert codec.token_id('A_PASS') not in raw
    assert codec.decode_value(raw) == block


def test_codec_spec_covers_the_exact_active_event_and_action_vocabulary(codec):
    event_names = {codec.token_name(i) for i in range(224,250)}
    action_names = {codec.token_name(i) for i in range(256,267)}
    assert set(EVENT_FIELDS) == event_names
    assert set(ACTION_SHAPES) == action_names
