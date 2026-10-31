"""Canonical tile identities and unsigned IR integer validation."""
from .mjai_errors import require


TILES = tuple(f'TILE_{n}{s}' for s in 'mps' for n in range(1, 10)) + (
    'TILE_E', 'TILE_S', 'TILE_W', 'TILE_N', 'TILE_P', 'TILE_F', 'TILE_C',
    'TILE_5mr', 'TILE_5pr', 'TILE_5sr')


def natural(value, label):
    require(type(value) is int and 0 <= value < 2**64, 'TYPE_MISMATCH', label)
