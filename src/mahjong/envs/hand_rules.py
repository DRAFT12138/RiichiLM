"""Riichi hand checks backed by the native hand evaluator.

This component supplies no action set or training target. It checks a recorded
declaration/discard/kan against explicit parameters and the actor's own tiles.
"""
from collections import Counter
from dataclasses import dataclass

from .offline_mjai import check


@dataclass(frozen=True)
class HandRules:
    riichi_min_score: int
    riichi_min_live: int
    riichi_kan_policy: str

    def __post_init__(self):
        for name in ('riichi_min_score','riichi_min_live'):
            value=getattr(self,name)
            check(type(value) is int and value>=0,'TYPE_MISMATCH',name)
        check(self.riichi_kan_policy=='RIICHI_KAN_WAITS','UNSUPPORTED_PROFILE',
              'riichi decomposition invariance is not implemented')

    @classmethod
    def from_dict(cls,value):
        check(type(value) is dict and value.keys()=={'riichi_min_score','riichi_min_live','riichi_kan_policy'},
              'TYPE_MISMATCH','hand_rules requires all three explicit parameters')
        return cls(**value)

    @staticmethod
    def waits(tiles,melds):
        from riichienv import HandEvaluator
        check(len(tiles)+3*len(melds)==13,'INVALID_INVENTORY','waiting hand size')
        # A fifth tile already exhausted by the actor's own material cannot
        # complete the hand. Other players' hidden hands do not enter this check.
        counts=Counter(t//4 for t in tiles)
        counts.update(t//4 for meld in melds for t in meld.tiles)
        return frozenset(t for t in HandEvaluator(tiles,melds).get_waits() if counts[t]<4)

    def declaration(self,tiles,melds,score,live):
        check(score>=self.riichi_min_score,'INVALID_RIICHI','insufficient riichi score')
        check(live>=self.riichi_min_live,'INVALID_RIICHI','insufficient live wall for riichi')
        check(all(not m.opened for m in melds),'INVALID_RIICHI','open riichi')
        check(len(tiles)+3*len(melds)==14,'INVALID_INVENTORY','riichi declaration hand size')
        check(any(self.waits(tiles[:i]+tiles[i+1:],melds) for i in range(len(tiles))),
              'INVALID_RIICHI','no discard leaves a tenpai hand')

    def declaration_discard(self,tiles,melds):
        waits=self.waits(tiles,melds)
        check(bool(waits),'INVALID_RIICHI','declaration discard does not leave tenpai')
        return waits

    def riichi_ankan(self,tiles,melds,consumed,drawn,established_waits):
        from riichienv import Meld,MeldType
        check(established_waits is not None and bool(established_waits),
              'INVALID_RIICHI_KAN','no verified riichi wait checkpoint')
        check(drawn is not None and drawn in consumed and all(t//4==drawn//4 for t in consumed),
              'INVALID_RIICHI_KAN','kan must use the just drawn fourth tile')
        remaining=list(tiles)
        for tile in consumed:
            remaining.remove(tile)
        post=self.waits(remaining,[*melds,Meld(MeldType.Ankan,consumed,False,-1)])
        check(post==established_waits and bool(post),'INVALID_RIICHI_KAN','kan changes riichi waits')
