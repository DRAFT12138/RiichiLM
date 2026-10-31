"""Public IR remains usable without private ingestion and preserves old types."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import pytest

from mahjong.envs import mjai_errors, mjai_events, mjai_ir, round_types
from mahjong.tokenizer import Atom, HandMeldState


@pytest.mark.parametrize('legacy_name,public_name,names', [
    ('common', 'mjai_errors', ('CleaningError', 'require')),
    ('source', 'mjai_events', ('SourceEvent',)),
    ('validation', 'ir_primitives', ('TILES', 'natural')),
    ('ir', 'mjai_ir', None),
    ('models', 'round_types', ('VISIBLE', 'FrozenAnchor', 'PreparedView', 'PreparedRound')),
    ('replay', 'round_ir', ('abs_seat', 'tile', 'furiten_values', 'canonical_settlement')),
])
def test_legacy_exports_are_same_objects(legacy_name, public_name, names):
    legacy = importlib.import_module('mahjong.data.mahjong.cleaning.' + legacy_name)
    public = importlib.import_module('mahjong.envs.' + public_name)
    if names is None:
        names = [name for name in vars(public) if not name.startswith('_')]
    for name in names:
        assert getattr(legacy, name) is getattr(public, name), name


def test_public_validation_is_caught_by_legacy_error_with_same_record():
    from mahjong.data.mahjong.cleaning.common import CleaningError

    with pytest.raises(CleaningError) as caught:
        mjai_errors.require(False, 'TYPE_MISMATCH', 'manual invalid event')
    assert type(caught.value) is mjai_errors.CleaningError
    assert str(caught.value) == 'TYPE_MISMATCH: manual invalid event'
    assert caught.value.record() == dict(code='TYPE_MISMATCH', reason='manual invalid event',
        stage='validation', raw_line=None, round_id=None, event_id=None)


def test_event_frozen_and_anchor_owns_nested_snapshot():
    event = mjai_events.SourceEvent(7, {'type': 'dahai', 'pai': '5mr'})
    with pytest.raises(FrozenInstanceError):
        event.raw_line = 8
    state = HandMeldState([Atom('TILE_5mr')], [{'tiles': [Atom('TILE_E')]}])
    original = deepcopy(state)
    anchor = round_types.FrozenAnchor.capture(8, 2, 2, state)
    state.hands.clear()
    state.melds[0]['tiles'].clear()
    assert anchor.state == original
    with pytest.raises(FrozenInstanceError):
        anchor.window_id = 9
    first = round_types.PreparedRound(0, [], [], [], [], True)
    second = round_types.PreparedRound(1, [], [], [], [], False)
    first.facts['result'] = 'manual'
    assert second.facts == {}


@pytest.mark.parametrize('legacy_name,instance', [
    ('source', mjai_events.SourceEvent(7, {'type': 'end_kyoku'})),
    ('models', round_types.FrozenAnchor.capture(8, 2, 2, HandMeldState([], []))),
    ('models', round_types.PreparedView(0, {}, [], [], 2)),
    ('models', round_types.PreparedRound(0, [], [], [], [], True)),
    ('ir', mjai_ir.IRProtocolFailure('invalid generated IR')),
])
def test_old_pickle_global_names_and_new_roundtrip(legacy_name, instance):
    # Protocol 0 has explicit GLOBAL module/name records. Resolve an old global
    # without storing binary fixtures or mutating the class's __module__.
    raw = pickle.dumps(instance, protocol=0)
    public = type(instance).__module__.encode() + b'\n'
    legacy = ('mahjong.data.mahjong.cleaning.' + legacy_name).encode() + b'\n'
    assert raw.count(public) == 1
    old = raw.replace(public, legacy, 1)
    for payload in (old, pickle.dumps(instance)):
        restored = pickle.loads(payload)
        assert type(restored) is type(instance)
        assert vars(restored) == vars(instance)
        if isinstance(instance, Exception):
            assert restored.args == instance.args


def test_public_ir_business_in_isolated_process_without_private_dependencies():
    source = Path(__file__).resolve().parents[3] / 'src'
    program = textwrap.dedent('''
        import importlib.abc
        import sys
        from copy import deepcopy

        # Python 3.14 pathlib imports fcntl internally. Block private storage
        # modules; stdlib's own platform support is not a project file lock.
        blocked = ('mahjong.data', 'mahjong_cleaning_native', 'torch',
                   'sqlite3', '_sqlite3', 'riichienv')
        class RejectPrivate(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
                    raise AssertionError('unexpected private dependency: ' + fullname)
        sys.meta_path.insert(0, RejectPrivate())
        from mahjong.envs import (mjai_errors, mjai_events, ir_primitives,
                                 mjai_ir, round_types, round_ir)
        from mahjong.tokenizer import Atom, Event, HandMeldState

        events = [Event('EV_DRAW', 0, (Atom('ABS_P0'), Atom('TILE_9m'), Atom('DRAW_LIVE'))),
                  Event('EV_DISCARD', 1, (Atom('ABS_P0'), Atom('TILE_9m'), Atom('TSUMOGIRI'), False))]
        window = dict(window_id=6, trigger_id=1, phase=Atom('PHASE_DISCARD_RESPONSE'),
                      actor=None, source=Atom('ABS_P0'), tile=Atom('TILE_9m'),
                      responders=[Atom('ABS_P1'), Atom('ABS_P2'), Atom('ABS_P3')])
        mjai_ir.check_window(window, events)
        mjai_ir.check_close(6, Atom('RESOLVE_NO_CALL'),
                           dict(actor=None, meld_id=None, winners=[], reason=None), {6: window})
        invalid = dict(window, responders=[Atom('ABS_P2')])
        try:
            mjai_ir.check_window(invalid, events)
        except mjai_errors.CleaningError as error:
            assert error.code == 'VISIBILITY_VIOLATION'
        else:
            raise AssertionError('invalid visibility admitted')
        settlement = dict(scores_before=[25000]*4, scores_after=[23000, 28000, 25000, 25000],
                          score_delta=[-2000, 3000, 0, 0], kyotaku=1, next_kyotaku=0,
                          payments=[dict(payer=0, payee=1, amount=2000, payment_kind='BASE'),
                                    dict(payer=None, payee=1, amount=1000, payment_kind='KYOTAKU')])
        original = deepcopy(settlement)
        result = round_ir.canonical_settlement(settlement)
        assert result['payments'] == [
            dict(payer=None, payee=Atom('ABS_P1'), amount=1000, payment_kind=Atom('PAYMENT_KYOTAKU')),
            dict(payer=Atom('ABS_P0'), payee=Atom('ABS_P1'), amount=2000, payment_kind=Atom('PAYMENT_BASE'))]
        mjai_ir.check_settlement(result)
        assert settlement == original
        result['score_delta'][1] += 1
        try:
            mjai_ir.check_settlement(result)
        except mjai_errors.CleaningError as error:
            assert error.code == 'SETTLEMENT_MISMATCH'
        else:
            raise AssertionError('invalid settlement admitted')
        try:
            ir_primitives.natural(True, 'round_id')
        except mjai_errors.CleaningError:
            pass
        else:
            raise AssertionError('bool admitted as integer')
        state = HandMeldState([Atom('TILE_5mr')], [])
        anchor = round_types.FrozenAnchor.capture(8, 2, 2, state)
        state.hands.clear()
        assert anchor.state.hands == [Atom('TILE_5mr')]
        assert mjai_events.SourceEvent(7, {'type': 'end_kyoku'}).raw_line == 7
        assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
    ''')
    result = subprocess.run([sys.executable, '-I', '-c',
        'import sys; sys.path.insert(0, ' + repr(str(source)) + ')\n' + program],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
