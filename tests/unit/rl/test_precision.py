"""Execution precision is part of Actor identity and sampled behavior."""
import os
import pytest
import torch

from mahjong import contracts as c
from mahjong.app.rl import create_settings, make_runtime
from mahjong.inference.actor_loading import compatibility, publish_actor, load_actor, new_actor_bridge
from mahjong.inference.policy import DecodeOptions
from mahjong.policy.execution import policy_precision, set_policy_precision, sampling_definition
from mahjong.policy.scoring import score_action
from mahjong.rl.collector import collect
from mahjong.rl.config import resolve
from mahjong.training.update import execute_update


def test_runtime_precision_and_actor_roundtrip(tiny, tmp_path):
    actor, settings, profile = tiny
    rl = resolve()
    assert create_settings(rl,settings,{},tmp_path,'cpu')['runtime']['precision']=='32-true'
    assert create_settings(rl,settings,{},tmp_path,'cuda:0')['runtime']['precision']=='bf16-mixed'
    fp32 = compatibility(actor,profile)
    set_policy_precision(actor,'bf16-mixed')
    assert compatibility(actor,profile)!=fp32
    ref=publish_actor(tmp_path/'actors',actor,settings,profile,{'fixture':'precision only'})
    loaded=load_actor(tmp_path/'actors'/ref['artifact_id'],expected=ref)
    assert policy_precision(loaded.model)=='bf16-mixed'
    assert loaded.source['compatibility']['precision']=='bf16-mixed'
    assert all(p.dtype==torch.float32 for p in loaded.model.parameters())
    set_policy_precision(actor,'32-true')
    with pytest.raises(c.MahjongError,match='conflicting compatibility'):
        publish_actor(tmp_path/'actors',actor,settings,profile,{})


def test_mixed_precision_table_rejected_before_collection(tiny, tmp_path):
    from types import SimpleNamespace
    actor,settings,profile=tiny
    other=type(actor)(type(actor.causal_lm)(actor.causal_lm.config),input_schema=actor.input_schema)
    set_policy_precision(other,'bf16-mixed')
    with pytest.raises(c.MahjongError,match='mixed policy precision'):
        collect(tmp_path,[SimpleNamespace(model=actor)]*3+[SimpleNamespace(model=other)],
            root_seed=42,episode='mixed',learner=0,limits=resolve()['collection'],header={})
    assert not (tmp_path/'trajectories').exists()


@pytest.mark.skipif(os.environ.get('MAHJONG_RL_GPU_TEST')!='1',reason='explicit GPU probe only')
def test_bf16_behavior_rescoring_and_joint_backward(tiny,tmp_path):
    actor,settings,profile=tiny
    torch.cuda.set_per_process_memory_fraction(.35)
    rl=resolve()
    config=create_settings(rl,settings,{'profile':profile},tmp_path,'cuda:0')
    runtime=make_runtime(actor,config)
    model=runtime.raw_model
    model.set_stage('JOINT')
    ref=publish_actor(tmp_path/'actors',actor,settings,profile,{'fixture':'GPU precision gradient'})
    frozen=load_actor(tmp_path/'actors'/ref['artifact_id'],device='cuda:0')
    bridge=new_actor_bridge([frozen]*4)
    bridge.reset_with_evidence(seed=42)
    window=bridge.training_window()
    view=window['views'][0]
    request=frozen.prepare(view)
    generated=frozen.generate(request,options=DecodeOptions('categorical'),generator=torch.Generator().manual_seed(42))
    score=score_action(model.actor,request.prefix_ids,request.legal_suffixes,generated.suffix_ids)
    torch.testing.assert_close(score.token_logprobs.detach().cpu(),torch.tensor(generated.token_logprobs),atol=2e-5,rtol=2e-5)
    reference=score_action(model.actor,request.prefix_ids,request.legal_suffixes,generated.suffix_ids,
                           skip_deterministic_tail=False)
    assert torch.equal(score.token_logprobs, reference.token_logprobs)
    assert torch.equal(score.prefix_hidden, reference.prefix_hidden)
    # Omitting zero-probability-gradient suffixes must preserve the real PPO
    # gradient through both the policy head and cached prefix embeddings.
    parameters = (actor.causal_lm.lm_head.weight, actor.causal_lm.model.embed_tokens.weight)
    expected = torch.autograd.grad(reference.action_logprob, parameters)
    actual = torch.autograd.grad(score.action_logprob, parameters)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    score=score_action(model.actor,request.prefix_ids,request.legal_suffixes,generated.suffix_ids)
    runtime.zero_grad()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        value=model.value(score.prefix_hidden,torch.tensor(window['oracle'][view.key.viewer],device='cuda'))
    runtime.backward(value.float().square())
    assert all(p.grad is None for p in model.actor.parameters())
    # Critic inputs are detached, while the full cached policy path remains differentiable.
    from dataclasses import asdict
    key=asdict(view.key)
    record=dict(prefix=request.prefix_ids,suffixes=request.legal_suffixes,chosen=generated.suffix_ids,
        oracle=window['oracle'][view.key.viewer],decision_key=key,oracle_key=key,
        forced=False,sampling='categorical',origin='model')
    target=dict(value_target=.5,old_logprob=generated.behavior_action_logprob,
        bc_logprob=generated.behavior_action_logprob,policy_adv=1.,policy_mask=True)
    batch=c.Batch([dict(record=record,target=target)],('gpu-precision',),1,model.descriptor().input_schema)
    runtime.zero_grad()
    result=runtime.compute(batch,phase='train')
    before=actor.causal_lm.lm_head.weight.detach().clone()
    execute_update(runtime,result,groups=model.parameter_groups(),limits={'actor':.5,'critic':.5})
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert not torch.equal(before,actor.causal_lm.lm_head.weight)
    # The header token is never a generated legal token: its gradient must
    # traverse cached prefix K/V rather than only the final decoding step.
    prefix_grad=actor.causal_lm.model.embed_tokens.weight.grad[request.prefix_ids[0]]
    assert bool(torch.any(prefix_grad!=0))
    assert all(p.grad is None or p.grad.dtype==torch.float32 for p in model.parameters())
    assert all(v.dtype==torch.float32 for state in runtime.raw_optimizer.state.values()
               for v in state.values() if isinstance(v,torch.Tensor))
    assert any(p.grad is not None and bool(torch.any(p.grad!=0)) for p in model.oracle.parameters())
    assert sampling_definition('bf16-mixed')['precision']==frozen.source['compatibility']['precision']
