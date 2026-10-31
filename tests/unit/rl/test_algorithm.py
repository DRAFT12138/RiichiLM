import math
from dataclasses import replace
import pytest
import torch
from mahjong.contracts import MahjongError, Batch
from mahjong.inference.policy import PolicyInference, DecodeOptions
from mahjong.models.qwen3_rl import Qwen3RLAdapter, policy_terms, SCHEMA
from mahjong.policy.scoring import score_action
from mahjong.rl.oracle import DIM
from mahjong.rl.targets import returns_and_gae
from mahjong.rl.config import resolve
from mahjong.rl.diagnostics import value_statistics, value_gate, probe_decision, check_ratios
from tests.unit.inference.test_policy import view


def test_sampling_rescoring_and_explicit_rng(tiny):
    actor, _, _ = tiny
    # Legacy fixture uses a valid standalone prefix, scoring is schema agnostic.
    from mahjong.models.qwen3 import Qwen3PolicyAdapter
    inference = PolicyInference(Qwen3PolicyAdapter(actor.causal_lm))
    request = inference.prepare(view())
    a = torch.Generator().manual_seed(43)
    b = torch.Generator().manual_seed(43)
    sequences = []
    for rng in (a,b):
        results = [inference.generate(request, options=DecodeOptions('categorical'), generator=rng) for _ in range(5)]
        sequences.append([r.suffix_ids for r in results])
        for r in results:
            score = score_action(actor, request.prefix_ids, request.legal_suffixes, r.suffix_ids)
            assert score.token_logprobs.detach().tolist() == pytest.approx(r.token_logprobs, abs=2e-5)
            assert float(score.action_logprob.detach()) == pytest.approx(r.behavior_action_logprob, abs=2e-5)
    assert sequences[0] == sequences[1]
    greedy = inference.generate(request)
    assert greedy.behavior_action_logprob == 0 and greedy.model_action_logprob < 0
    with pytest.raises(MahjongError): inference.generate(request, options=DecodeOptions('categorical'))


def test_complete_action_normalization_and_gradient(tiny):
    from mahjong.models.qwen3 import Qwen3PolicyAdapter
    actor,_,_ = tiny
    inference = PolicyInference(Qwen3PolicyAdapter(actor.causal_lm))
    request = inference.prepare(view())
    scores = [score_action(actor,request.prefix_ids,request.legal_suffixes,a) for a in request.legal_suffixes]
    assert sum(float(s.action_logprob.detach().exp()) for s in scores) == pytest.approx(1, abs=1e-6)
    (-scores[0].action_logprob).backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)>0
    for s in scores: assert s.token_logprobs[-1] == 0


def test_gae_hand_boundaries_and_mc():
    mc, adv = returns_and_gae([0, .1, -.3, .7], [1,2,3,4])
    assert mc == pytest.approx([.5,.5,.4,.7])
    assert adv == pytest.approx([-.5,-1.5,-2.6,-3.3])
    mc, adv = returns_and_gae([1,2], [.4,.6], .95)
    assert adv == pytest.approx([2.53,1.4])
    with pytest.raises(MahjongError): returns_and_gae([1], [0], gamma=.99)


def test_reverse_kl_exact_expectation_and_gradient():
    q = torch.tensor([.2,.3,.5],dtype=torch.float64)
    b = torch.tensor([.6,.1,.3],dtype=torch.float64)
    logits = torch.tensor([.2,-.3,1.],dtype=torch.float64,requires_grad=True)
    l = logits.log_softmax(0)
    terms = policy_terms(l,q.log(),b.log(),torch.tensor([1.,-2.,.5]),.1)
    estimated = (q*terms['bc_kl']).sum()
    exact = (l.exp()*(l-b.log())).sum()
    assert torch.allclose(estimated,exact,atol=1e-12)
    g1 = torch.autograd.grad(estimated,logits,retain_graph=True)[0]
    g2 = torch.autograd.grad(exact,logits)[0]
    assert torch.allclose(g1,g2,atol=1e-12)
    assert float((q*terms['old_kl']).sum().detach()) == pytest.approx(float((q*(q.log()-l.detach())).sum()), abs=1e-12)


def test_finite_kl_missing_tail_and_repeated_epochs():
    q = torch.tensor([.999999,.000001],dtype=torch.float64)
    b = torch.tensor([.5,.5],dtype=torch.float64)
    samples = torch.multinomial(q,256,replacement=True,generator=torch.Generator().manual_seed(41))
    assert samples.eq(0).all()  # explicit blind spot, NOT evidence of a safe constraint
    for include_tail in (False,True):
        selected = samples.clone()
        if include_tail: selected[-1] = 1
        logits = torch.tensor([math.log(.99),math.log(.01)],dtype=torch.float64,requires_grad=True)
        optimizer = torch.optim.SGD([logits],lr=1e-5)
        for epoch in range(2):
            l = logits.log_softmax(0)
            terms = policy_terms(l[selected],q.log()[selected],b.log()[selected],torch.zeros(256),.1)
            estimate = terms['bc_kl'].mean()
            exact = (l.exp()*(l-b.log())).sum()
            if include_tail:
                assert terms['ratio'].max()>9000 and estimate>1000
            else:
                assert estimate<exact/2
            optimizer.zero_grad(); estimate.backward()
            assert torch.isfinite(logits.grad).all()
            if include_tail: assert logits.grad.norm()>100
            optimizer.step()


def test_critic_isolation_and_prefix_position(tiny):
    from mahjong.models.qwen3 import Qwen3PolicyAdapter
    actor,_,_ = tiny
    settings = {**resolve()['ppo'],'oracle_width':8,'value_width':8}
    model = Qwen3RLAdapter(actor,settings)
    request = PolicyInference(Qwen3PolicyAdapter(actor.causal_lm)).prepare(view())
    hidden = [score_action(actor,request.prefix_ids,request.legal_suffixes,a).prefix_hidden for a in request.legal_suffixes]
    assert all(torch.allclose(hidden[0],h,atol=1e-6) for h in hidden[1:])
    model.set_stage('JOINT')
    value = model.value(hidden[0],torch.zeros(DIM))
    value.square().backward()
    assert all(p.grad is None for p in actor.parameters())
    assert any(p.grad is not None for p in model.value_head.parameters())


def test_value_statistics_bias_and_zero_variance():
    report = value_statistics([11,12,13],[1,2,3])
    assert report['ev']==1 and report['bias']==10 and report['rmse']==10
    assert not value_gate(report,4,resolve()['critic'])
    assert value_statistics([0,1],[1,1])['ev'] is None
    shifted = value_statistics([1e12+1,1e12+3],[1e12,1e12+2])
    assert shifted['ev']==1 and shifted['bias']==1


def test_controls_and_invalid_configuration():
    config = resolve()['control']
    assert probe_decision(.03,32,config)
    with pytest.raises(MahjongError): probe_decision(.2,32,config)
    with pytest.raises(MahjongError): probe_decision(.001,1,config)
    with pytest.raises(MahjongError): resolve(overrides=['ppo.lam=1.1'])
    with pytest.raises(MahjongError): resolve(overrides=['unknown=1'])
