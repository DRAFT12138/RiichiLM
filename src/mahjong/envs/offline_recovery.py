"""Prove a new round boundary after quarantining a broken round suffix.

This keeps recorded financial continuity, not a second hand/rules simulation.
Unknown money events or unknown round identity cannot be repaired by guessing.
"""
from copy import deepcopy

from .offline_mjai import OfflineSession, ReplayError, check, recorded_settlement

NONFINANCIAL = frozenset(('tsumo','dahai','chi','pon','daiminkan','ankan','kakan','dora','reach','end_kyoku'))


def check_round_identity(before, after):
    old='ESWN'.index(before['bakaze'])*4+before['kyoku']-1
    new='ESWN'.index(after['bakaze'])*4+after['kyoku']-1
    check(new-old in (0,1) and after['oya']==(before['oya']+new-old)%4,
          'RECOVERY_IDENTITY_MISMATCH','new round position/dealer is not the same or next recorded round')


class RoundRecovery:
    def __init__(self, options, checkpoint, *, bad_kind, bad_event=None,
                 round_ended=False, source_line=None, rule=None, recorded_result=None):
        self.options,self.rule=options,rule
        self.checkpoint=deepcopy(checkpoint)
        self.header=deepcopy(checkpoint['header'])
        self.scores=list(checkpoint['scores'])
        self.kyotaku=checkpoint['kyotaku']
        self.settled=checkpoint['settled']
        self.ended=round_ended
        self.known=self.header is not None and bad_kind in NONFINANCIAL
        self.reason=None if self.known else 'unverified financial/round boundary'
        self.source_refs=[source_line] if source_line is not None else []
        self.settlements=[]
        if recorded_result is not None:
            self.known=self.header is not None
            self.reason=None if self.known else self.reason
            self.observe([e.value for e in recorded_result],[e.raw_line for e in recorded_result])
        if bad_kind=='start_kyoku' and bad_event is not None:
            # A typed round-start may have invalid tiles while its independently
            # recorded identity/balance is still provable. Never derive balances
            # from the engine after its failed apply_event.
            if self.header is None:
                self.known=True
            elif self.settled and self.ended:
                self._check_balances(bad_event)
                check_round_identity(self.header,bad_event)
                self.known=True
            if self.known:
                self.header=deepcopy(bad_event)
                self.scores=list(bad_event['scores'])
                self.kyotaku=bad_event['kyotaku']
                self.settled=self.ended=False
                self.reason=None

    def invalidate(self, reason):
        self.known=False
        self.reason=reason

    def _check_balances(self,event):
        check(event['scores']==self.scores and event['kyotaku']==self.kyotaku,
              'RECOVERY_BALANCE_MISMATCH','next round does not match verified recorded scores/sticks')

    def observe(self,events,lines):
        """Read only independently checkable money facts in the quarantined suffix."""
        kind=events[0]['type']
        if kind=='reach_accepted':
            self.invalidate('riichi acceptance after state failure has no verified declaration/discard context')
        elif kind in ('hora','ryukyoku') and self.known:
            try:
                check(not self.settled,'SETTLEMENT_MISMATCH','duplicate settlement in quarantined round')
                result=recorded_settlement(self.options,events,self.scores,self.kyotaku)
            except ReplayError:
                self.invalidate('invalid financial bridge')
                raise
            self.scores=list(result['scores_after'])
            self.kyotaku=result['next_kyotaku']
            self.settled=True
            self.settlements.append(dict(result=result,source_refs=list(lines)))
            self.source_refs.extend(lines)
        elif kind=='end_kyoku':
            if not self.settled:
                self.invalidate('round ended without a verified settlement')
            self.ended=True
            self.source_refs.extend(lines)
        elif kind=='start_game':
            self.invalidate('duplicate game boundary')

    def next_round(self,event,line):
        check(self.known and self.settled and self.ended,'RECOVERY_UNVERIFIED',
              self.reason or 'no verified result/end boundary')
        self._check_balances(event)
        check_round_identity(self.header,event)
        session=OfflineSession(self.options,rule=self.rule)
        session.env.set_scores(self.scores)
        session.env.riichi_sticks=self.kyotaku
        session.header=deepcopy(self.header)
        session.ended=True
        proof=dict(checkpoint=self.checkpoint,settlements=deepcopy(self.settlements),
                   source_refs=list(dict.fromkeys(self.source_refs+[line])),
                   scores=list(self.scores),kyotaku=self.kyotaku,
                   previous_round=dict(bakaze=self.header['bakaze'],kyoku=self.header['kyoku'],oya=self.header['oya']))
        return session,proof
