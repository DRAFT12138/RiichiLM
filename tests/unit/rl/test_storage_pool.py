from copy import deepcopy
import json
import pytest
import torch
from mahjong.contracts import MahjongError
from mahjong.inference.actor_loading import publish_actor, load_actor, new_actor_bridge
from mahjong.rl.storage import EpisodeWriter, EpisodeReader
from mahjong.rl.config import resolve
from mahjong.rl.pool import initial_pool, promote, members, select
from mahjong.rl.evaluation import evaluate_values, paired_statistics


def test_actor_self_contained_strict_roundtrip(tiny,tmp_path):
    actor,settings,profile = tiny
    ref = publish_actor(tmp_path/'actors',actor,settings,profile,{'checkpoint':'already-deleted'})
    loaded = load_actor(tmp_path/'actors'/ref['artifact_id'],expected=ref)
    assert all(torch.equal(p,loaded.model.state_dict()[k]) for k,p in actor.state_dict().items())
    assert not any(p.requires_grad for p in loaded.model.parameters())
    assert publish_actor(tmp_path/'actors',actor,settings,profile,{}) == ref
    bridge = new_actor_bridge([loaded]*4)
    evidence = bridge.reset_with_evidence(seed=17)
    assert len(evidence['transitions'])==1
    window = bridge.training_window()
    assert window['views'] and all(len(v)==127 for v in window['oracle'].values())
    corrupted = deepcopy(ref); corrupted['weights_sha256']='0'*64
    with pytest.raises(MahjongError): load_actor(tmp_path/'actors'/ref['artifact_id'],expected=corrupted)
    weights = tmp_path/'actors'/ref['artifact_id']/'actor.pt'
    with weights.open('ab') as f: f.write(b'corrupt')
    with pytest.raises(MahjongError): load_actor(weights.parent)


def test_jsonl_unicode_offsets_and_limits(tmp_path):
    limits = resolve()['collection']
    writer = EpisodeWriter(tmp_path/'trajectories','episode',limits,{'learner':0})
    for i in range(10): writer.write({'key':str(i),'type':'test','text':'麻将'*(i+1)})
    ref = writer.publish([25000]*4,[1,2,3,4])
    reader = EpisodeReader(tmp_path,ref,limits)
    assert [reader.read(str(i))['text'] for i in reversed(range(10))]==['麻将'*(i+1) for i in reversed(range(10))]
    with pytest.raises(MahjongError): EpisodeReader(tmp_path,ref,{**limits,'max_offset_bytes':1})
    with (tmp_path/ref['path']/'records.jsonl').open('ab') as f: f.write(b'{}\n')
    with pytest.raises((MahjongError,KeyError)): EpisodeReader(tmp_path,ref,limits)


def test_pool_full_rotation_and_restore_protection():
    config = resolve()['pool']; config['history_interval']=1
    ref = lambda n: {'artifact_id':str(n)}
    pool = initial_pool(ref(0))
    previous = set()
    for i in range(1,100):
        next_pool = promote(pool,ref(i),i,config)
        protected = previous | set(members(pool)) | set(members(next_pool))
        assert len(members(next_pool))<=16
        assert len(protected)<=18
        assert next_pool['bc']==ref(0)
        previous = set(members(pool)); pool=next_pool
    empty = initial_pool(ref(0))
    selected = select(empty,42,1,[0.,1.,0.])
    assert selected['requested_category']=='history' and selected['actual_category']=='bc'
    assert selected==select(empty,42,1,[0.,1.,0.])


def test_value_policy_mismatch_rejected_and_group_statistics():
    with pytest.raises(MahjongError,match='greedy'):
        evaluate_values([],None,actor_ref={},expected_sampling={'strategy':'greedy'},scope='candidate_eval')
    assert paired_statistics([.1],seed=1,samples=100,confidence=.95)['status']=='not_evaluable'
    assert paired_statistics([1,1,1],seed=1,samples=100,confidence=.95)['interval']==[1,1]


def test_paired_seeds_failure_denominators_and_no_training_rng_use(tmp_path,tiny,monkeypatch):
    from mahjong.rl.evaluation import paired_evaluate
    from types import SimpleNamespace
    actor,settings,profile=tiny
    reference={'artifact_id':'candidate'}
    policy=SimpleNamespace(source={'actor_ref':reference,'compatibility':{'profile':profile}})
    calls=[]
    def collect_fixture(root,actors,**kwargs):
        calls.append(kwargs)
        seat=kwargs['learner']
        if len(calls)==2:
            return dict(status='failed',learner=seat,reason='injected')
        return dict(status='complete',learner=seat,final_scores=[25000]*4,ranks=[1,2,3,4],
                    ref={'bytes':1,'path':'fixture'})
    monkeypatch.setattr('mahjong.rl.evaluation.collect',collect_fixture)
    rng=torch.get_rng_state().clone()
    report=paired_evaluate(tmp_path,policy,policy,policy,model=None,config=resolve(),iteration=0,cancel=lambda:False)
    assert torch.equal(rng,torch.get_rng_state())
    assert report['attempts']==8 and report['failures']==1 and not report['accepted']
    assert report['statistics']['status']=='not_evaluable'
    for a,b in zip(calls[::2],calls[1::2]):
        assert a['environment_seed']==b['environment_seed']
        assert a['sampling_episode']==b['sampling_episode']
        assert a['learner']==b['learner']
