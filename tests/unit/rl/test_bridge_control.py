import math
import pytest
import torch
from mahjong import contracts as c
from mahjong.rl.oracle import encode_oracle
from mahjong.rl.diagnostics import check_ratios
from mahjong.rl.config import resolve
from mahjong.training.update import execute_update
from tests.unit.inference.test_bridge import rig,select


def test_evidence_keeps_both_riichi_transitions_and_oracle_precedes_intent():
    b=rig({0:['1m','2m','3m','4m','5m','6m','2p','3p','4p','6s','7s','8s','E']},drawn='9p')
    before=b.training_window();v=b.snapshot(0)
    decision=select(b,v,next(m for m in v.legal_moves if m.action.kind=='A_RIICHI_DISCARD'))
    assert before==b.training_window()
    evidence=b.apply_with_evidence(decision)
    assert len(evidence['transitions'])==2
    assert evidence['transitions'][1].transition_seq==evidence['transitions'][0].transition_seq+1
    assert evidence['before']==before
    assert b._evidence_sink is None


def test_oracle_relative_hands_and_visible_input_isolation():
    b=rig({0:['1m','2m','3m'],1:['9s']},drawn='9p')
    v=b.snapshot(0);raw=b._session.snapshot()
    original=encode_oracle(raw,b._session.header,0)
    raw['hands'][1]=['1m']
    changed=encode_oracle(raw,b._session.header,0)
    assert changed!=original
    assert b.snapshot(0).prefix_ids==v.prefix_ids
    assert b.snapshot(0).legal_moves==v.legal_moves


def test_group_clip_and_nonfinite_stop_before_optimizer():
    class Runtime:
        def __init__(self):
            self.raw_model=torch.nn.Linear(1,1,bias=False);self.steps=0
            self.optimizer=torch.optim.SGD(self.raw_model.parameters(),lr=1)
        def backward(self,loss):loss.backward()
        def check_gradients(self):
            if any(not torch.isfinite(p.grad).all() for p in self.raw_model.parameters()):
                raise c.MahjongError('NONFINITE_GRADIENT','bad gradient')
        def step(self):self.optimizer.step();self.steps+=1
    runtime=Runtime();p=next(runtime.raw_model.parameters());before=p.detach().clone()
    result=c.StepResult((p*1000).sum(),{'loss':c.MetricStat(1.,1,'loss')})
    actual=execute_update(runtime,result,groups=[('actor',[p])],limits={'actor':.5})
    assert runtime.steps==1 and torch.linalg.vector_norm(p-before)<=.50001
    assert actual.metrics['actor_grad_norm'].numerator==1000
    runtime.optimizer.zero_grad()
    p.register_hook(lambda g:torch.full_like(g,float('nan')))
    with pytest.raises(c.MahjongError):execute_update(runtime,c.StepResult(p.sum(),{'loss':c.MetricStat(1.,1,'loss')}),groups=[('actor',[p])],limits={'actor':.5})
    assert runtime.steps==1


def test_observed_rare_ratio_actually_prevents_step():
    control=resolve()['control']
    result=c.StepResult(torch.tensor(1.,requires_grad=True),{
        'loss':c.MetricStat(1.,1,'loss'), 'ratio_max':c.MetricStat(10000.,1,'ratio'),'ratio_min':c.MetricStat(1.,1,'ratio')})
    class NoStep:
        def backward(self,loss):raise AssertionError('must reject before backward')
    with pytest.raises(c.MahjongError,match='ratio'):
        execute_update(NoStep(),result,before_step=lambda r:check_ratios(r,control))
