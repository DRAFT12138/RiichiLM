"""Versioned, action-before Critic-only features (no private response intentions)."""
import math
from mahjong.contracts import MahjongError

SCHEMA = 'mahjong-oracle/v1'
TILES = tuple(f'{i}{s}' for s in 'mps' for i in range(1, 10)) + tuple('ESWNPFC')
DIM = 3 * 37 + 4 + 1 + 4 + 1 + 1 + 4 + 1


def encode_oracle(snapshot, header, viewer):
    try:
        values = []
        for offset in (1, 2, 3):
            hand = snapshot['hands'][(viewer + offset) % 4]
            counts, reds = [0.] * 34, [0.] * 3
            for tile in hand:
                counts[TILES.index(tile.removesuffix('r'))] += .25
                if tile.endswith('r'):
                    reds['mps'.index(tile[1])] += 1.
            if any(n > 1 for n in counts) or any(n > 1 for n in reds):
                raise ValueError('invalid tile inventory')
            values.extend(counts + reds)
        values += [snapshot['scores'][(viewer + i) % 4] / 10000 for i in range(4)]
        values += [snapshot['kyotaku'] / 10]
        values += [float(header['bakaze'] == wind) for wind in 'ESWN']
        values += [header['kyoku'] / 4, header['honba'] / 10]
        values += [float((header['oya'] - viewer) % 4 == i) for i in range(4)]
        values += [snapshot['live_remaining'] / 70]
        if len(values) != DIM or not all(math.isfinite(v) for v in values):
            raise ValueError('invalid Oracle features')
        return tuple(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise MahjongError('CONTRACT_MISMATCH', f'Oracle snapshot invalid: {exc}') from exc
