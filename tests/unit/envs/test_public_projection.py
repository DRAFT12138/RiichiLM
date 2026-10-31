"""Public leaf modules work without private ingestion or native extensions."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from mahjong.envs import projection, result_constants
from mahjong.tokenizer import Atom, Event


RELATIVE = (
    ('SELF', 'SHIMOCHA', 'TOIMEN', 'KAMICHA'),
    ('KAMICHA', 'SELF', 'SHIMOCHA', 'TOIMEN'),
    ('TOIMEN', 'KAMICHA', 'SELF', 'SHIMOCHA'),
    ('SHIMOCHA', 'TOIMEN', 'KAMICHA', 'SELF'),
)


@pytest.mark.parametrize('viewer', range(4))
@pytest.mark.parametrize('actor', range(4))
def test_draw_visibility_and_nested_seat_rotation(viewer, actor):
    absolute = Atom(f'ABS_P{actor}')
    relative = Atom(RELATIVE[viewer][actor])
    source = Event('EV_DRAW', 7, (absolute, Atom('TILE_5mr'), Atom('DRAW_LIVE')))
    projected = projection.project_event(source, viewer, ankan_public='base_only')
    assert projection.seat(actor, viewer) == relative
    assert projected == Event('EV_DRAW', 7, (
        relative, Atom('TILE_5mr' if viewer == actor else 'TILE_HIDDEN'), Atom('DRAW_LIVE'),
    ))
    nested = {'seat': [absolute, (absolute,)], 'unchanged': 3}
    before = deepcopy(nested)
    assert projection.rotate(nested, viewer) == {'seat': [relative, (relative,)], 'unchanged': 3}
    assert nested == before
    assert source.payload[1] == Atom('TILE_5mr')


@pytest.mark.parametrize('viewer', range(4))
@pytest.mark.parametrize('visibility', ['base_only', 'all_tiles'])
def test_concealed_kan_visibility(viewer, visibility):
    tiles = [Atom('TILE_5m')] * 3 + [Atom('TILE_5mr')]
    source = Event('EV_ANKAN_ATTEMPT', 9, (Atom('ABS_P0'), Atom('TILE_5m'), tiles))
    projected = projection.project_event(source, viewer, ankan_public=visibility)
    expected = [Atom('TILE_HIDDEN')] * 4 if viewer != 0 and visibility == 'base_only' else tiles
    assert projected == Event('EV_ANKAN_ATTEMPT', 9, (
        Atom(RELATIVE[viewer][0]), Atom('TILE_5m'), expected,
    ))
    assert source.payload[2] == tiles


@pytest.mark.parametrize('viewer', range(4))
def test_header_and_hand_projection_preserve_source(viewer):
    initial = dict(bakaze='S', kyoku=2, honba=3, kyotaku=1, oya=2,
                   scores=[21000, 22000, 23000, 24000])
    meld = dict(name='MELD_PON', actor=0, **{'from': 2}, base_tile='5p',
                called_tile='5pr', tiles=['5p', '5p', '5pr'])
    snapshot = dict(hands=[['1m'], ['2p'], ['3s'], ['E']], melds=[[meld] for _ in range(4)])
    before = deepcopy((initial, snapshot))
    header = projection.project_header(initial, viewer, 'MODE_HANCHAN')
    assert header == dict(
        mode=Atom('MODE_HANCHAN'), round_wind=Atom('ROUND_S'), round_number=2,
        honba=3, kyotaku=1, dealer=Atom(RELATIVE[viewer][2]),
        scores=initial['scores'][viewer:] + initial['scores'][:viewer],
        seat_winds=[Atom('TILE_' + wind) for wind in ('WNES', 'NESW', 'ESWN', 'SWNE')[viewer]],
    )
    hand = projection.project_hand(snapshot, viewer)
    assert hand.hands == [Atom('TILE_' + tile) for tile in snapshot['hands'][viewer]]
    assert hand.melds == [dict(name=Atom('MELD_PON'), actor=Atom('SELF'),
        **{'from': Atom(RELATIVE[viewer][2])}, base_tile=Atom('TILE_5p'),
        called_tile=Atom('TILE_5pr'), tiles=[Atom('TILE_5p'), Atom('TILE_5p'), Atom('TILE_5pr')])]
    assert projection.project_current_hand(snapshot['hands'][viewer], [meld], viewer) == hand
    assert (initial, snapshot) == before




def test_public_leaves_import_in_fresh_process_without_private_data():
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        sys.path.insert(0, sys.argv[1])
        class BlockPrivate(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'mahjong.data' or fullname.startswith('mahjong.data.'):
                    raise AssertionError('private dependency: ' + fullname)
        sys.meta_path.insert(0, BlockPrivate())
        from mahjong.envs import projection, result_constants
        assert not any(n == 'mahjong.data' or n.startswith('mahjong.data.') for n in sys.modules)
        assert 'torch' not in sys.modules
        assert 'riichienv' not in sys.modules
    ''')
    source = Path(__file__).resolve().parents[3] / 'src'
    result = subprocess.run([sys.executable, '-I', '-c', script, str(source)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
