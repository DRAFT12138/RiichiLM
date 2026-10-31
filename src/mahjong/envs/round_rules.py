"""Finite round-boundary table, revision round-termination/1.

Positions are nominal (E1=1, S4=8, W4=12); repeats never consume positions.
Missing result facts produce a finite set of possibilities, not default facts.
"""
from dataclasses import dataclass

from .offline_mjai import check

END_REASONS=('END_BUST','END_DEALER_STOP','END_TARGET_REACHED','END_MAX_ROUND','END_NORMAL')


def position(header):
    return 4*'ESWN'.index(header['bakaze'])+header['kyoku']


@dataclass(frozen=True)
class RoundTransition:
    game_over: bool
    next_dealer: int | None
    next_position: int | None
    next_honba: int | None
    next_kyotaku: int
    reason: str | None


@dataclass(frozen=True)
class RoundDecision:
    outcomes: tuple

    @property
    def game_over(self):
        values={v.game_over for v in self.outcomes}
        return values.pop() if len(values)==1 else None

    @property
    def reason(self):
        values={v.reason for v in self.outcomes}
        return values.pop() if self.game_over is True and len(values)==1 else None

    def check_boundary(self, event=None):
        if event is None:
            valid=any(v.game_over for v in self.outcomes)
        else:
            valid=any(not v.game_over and
                (v.next_dealer,v.next_position,v.next_honba,v.next_kyotaku)==
                (event['oya'],position(event),event['honba'],event['kyotaku']) for v in self.outcomes)
        check(valid,'ROUND_BOUNDARY_MISMATCH','recorded boundary contradicts round-termination/1')


@dataclass(frozen=True)
class RoundRules:
    initial_scores: list
    target_points: int
    scheduled_rounds: int
    max_rounds: int
    bust_below_zero: bool
    stop_at_target: bool
    dealer_win_repeat: bool
    dealer_tenpai_repeat: bool
    dealer_win_stop: bool
    dealer_tenpai_stop: bool
    tie_order: str
    end_priority: list

    def __post_init__(self):
        check(type(self.initial_scores) is list and len(self.initial_scores)==4 and
              all(type(s) is int for s in self.initial_scores),'TYPE_MISMATCH','round initial_scores')
        check(type(self.target_points) is int and self.target_points>=0,'TYPE_MISMATCH','round target_points')
        check(type(self.scheduled_rounds) is int and self.scheduled_rounds in (4,8) and
              type(self.max_rounds) is int and self.max_rounds in (4,8,12,16) and
              self.max_rounds>=self.scheduled_rounds,'TYPE_MISMATCH','nominal round bounds')
        for name in ('bust_below_zero','stop_at_target','dealer_win_repeat','dealer_tenpai_repeat',
                     'dealer_win_stop','dealer_tenpai_stop'):
            check(type(getattr(self,name)) is bool,'TYPE_MISMATCH',name)
        check(self.tie_order=='starting_dealer_rotation','UNSUPPORTED_PROFILE','round tie order')
        check(type(self.end_priority) is list and len(self.end_priority)==5 and
              all(type(v) is str for v in self.end_priority) and set(self.end_priority)==set(END_REASONS),
              'TYPE_MISMATCH','round end_priority')

    @classmethod
    def from_dict(cls,value):
        check(type(value) is dict and value.keys()==cls.__dataclass_fields__.keys(),
              'TYPE_MISMATCH','round_rules requires full round-termination/1 parameters')
        return cls(**value)

    def _results(self,result,dealer):
        # (repeat, stop switch, next honba increment). Nagashi is draw-like in
        # this supported table: dealer tenpai governs repeat and tenpai-stop.
        if result is None:
            return [v for possible in (dict(kind='win',winners=[dealer]),
                dict(kind='win',winners=[(dealer+1)%4]),dict(kind='abort'),
                dict(kind='exhaustive',tenpai=[False]*4),dict(kind='exhaustive',tenpai=[True]*4))
                for v in self._results(possible,dealer)]
        kind=result['kind']
        if kind=='win':
            repeat=dealer in result['winners'] and self.dealer_win_repeat
            return [(repeat,self.dealer_win_stop,1 if repeat else 0)]
        if kind=='abort':
            return [(True,False,1)]
        check(kind in ('exhaustive','nagashi','unknown_draw'),'TYPE_MISMATCH','round result kind')
        tenpai=result['tenpai'][dealer]
        cases=[False,True] if tenpai is None else [tenpai]
        values=[(v and self.dealer_tenpai_repeat,self.dealer_tenpai_stop,1) for v in cases]
        if kind=='unknown_draw' and result.get('may_abort',True):
            values.append((True,False,1))
        return values

    def decide(self,header,scores,kyotaku,result,*,starting_dealer):
        current=position(header)
        check(current<=self.max_rounds,'ROUND_BOUNDARY_MISMATCH','round exceeds maximum nominal position')
        dealer=header['oya']
        # At equal scores only the explicitly proved starting dealer defines
        # first place. Unknown identity leaves both possible tie resolutions.
        leaders=[p for p in range(4) if scores[p]==max(scores)]
        if dealer not in leaders:
            first=[False]
        elif len(leaders)==1:
            first=[True]
        elif starting_dealer is None:
            first=[False,True]
        else:
            first=[dealer==min(leaders,key=lambda p:(p-starting_dealer)%4)]
        outcomes=[]
        for repeat,stop,honba_increment in self._results(result,dealer):
            for dealer_first in first:
                reasons=[]
                if self.bust_below_zero and min(scores)<0:
                    reasons.append('END_BUST')
                if current>=self.scheduled_rounds:
                    if repeat:
                        if stop and dealer_first and scores[dealer]>=self.target_points:
                            reasons.append('END_DEALER_STOP')
                    elif not self.stop_at_target:
                        reasons.append('END_NORMAL')
                    else:
                        if max(scores)>=self.target_points:
                            reasons.append('END_TARGET_REACHED')
                        if current==self.max_rounds:
                            reasons.append('END_MAX_ROUND')
                if reasons:
                    value=RoundTransition(True,None,None,None,kyotaku,min(reasons,key=self.end_priority.index))
                else:
                    value=RoundTransition(False,dealer if repeat else (dealer+1)%4,
                        current if repeat else current+1,header['honba']+1 if honba_increment else 0,kyotaku,None)
                if value not in outcomes:outcomes.append(value)
        return RoundDecision(tuple(outcomes))
