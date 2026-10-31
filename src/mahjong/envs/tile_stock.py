"""Capacity of observed tile entities under explicit red-five counts.

MJAI/native replay uses canonical IDs, so duplicate native IDs do not identify
distinct physical copies. Capacity is checked by tile value, across all known
hands, melds, unclaimed discards and revealed wall indicators.
"""
from collections import Counter

from .offline_mjai import base, check


class TileStock:
    def __init__(self, aka_counts=None):
        # None preserves only the legacy artificial adapter's upper bounds.
        # A complete profile must supply counts; no preset establishes them.
        if aka_counts is not None:
            check(type(aka_counts) is list and len(aka_counts)==3 and
                  all(type(n) is int and n>=0 for n in aka_counts),
                  'TYPE_MISMATCH','aka_counts requires three nonnegative integers')
            check(all(n<=4 for n in aka_counts),'UNSUPPORTED_PROFILE','more than four red fives in a suit')
        self.aka_counts=None if aka_counts is None else tuple(aka_counts)

    def check(self, tiles):
        counts=Counter(tiles)
        bases=Counter()
        for tile,n in counts.items():bases[base(tile)]+=n
        check(all(n<=4 for n in bases.values()),
              'INVALID_INVENTORY','more than four copies')
        if self.aka_counts is None:
            check(all(n<=1 for t,n in counts.items() if t.endswith('r')),
                  'INVALID_INVENTORY','red copies')
        else:
            for suit,red in zip('mps',self.aka_counts):
                check(counts['5'+suit+'r']<=red,'INVALID_INVENTORY','red capacity: 5'+suit)
                check(counts['5'+suit]<=4-red,'INVALID_INVENTORY','ordinary five capacity: 5'+suit)
