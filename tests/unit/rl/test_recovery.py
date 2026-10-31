from dataclasses import asdict, replace
import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from mahjong import contracts as c
from mahjong.app import runs
from mahjong.app.rl import create_settings, make_runtime
from mahjong.inference.actor_loading import publish_actor,load_actor,new_actor_bridge
from mahjong.inference.policy import digest
from mahjong.policy.scoring import score_action
from mahjong.rl.config import resolve
from mahjong.rl.storage import EpisodeWriter,EpisodeReader
from mahjong.rl.targets import build_targets
from mahjong.rl.pool import initial_pool
from mahjong.rl.source import RLTrainSource
from mahjong.rl.controller import RLController
from mahjong.rl.recovery import reconcile
from mahjong.observability.events import LocalObserver
from mahjong.training.checkpoint import FileCheckpointStore
from mahjong.training.signals import SignalStopToken


def fixture_episode(root, loaded, model, rl, name='fixture'):
    bridge = new_actor_bridge([loaded]*4);bridge.reset(seed=11)
    window=bridge.training_window();view=window['views'][0];learner=view.key.viewer
    request=loaded.prepare(view)
    writer=EpisodeWriter(root/'trajectories',name,rl['collection'],dict(episode=name,learner=learner,
        initial_scores=[25000]*4,actors=[loaded.source['actor_ref']]*4,
        sampling=dict(strategy='categorical',temperature=1,precision='32-true',trie='complete')))
    scores=[25000]*4
    for i,action in enumerate((request.legal_suffixes[0],request.legal_suffixes[-1])):
        with torch.no_grad(): score=score_action(model.actor,request.prefix_ids,request.legal_suffixes,action)
        key=asdict(view.key);key['window']['round_id']=i
        writer.write(dict(type='decision',key=f'd{i}',decision_key=key,oracle_key=key,round_id=i,viewer=learner,
            prefix=list(request.prefix_ids),suffixes=[list(s) for s in request.legal_suffixes],chosen=list(action),
            old_logprob=float(score.action_logprob),token_logprobs=score.token_logprobs.tolist(),
            sampling='categorical',origin='model',actor_ref=loaded.source['actor_ref'],forced=False,
            oracle=list(window['oracle'][learner])))
        scores[learner]+=1000 if i==0 else -2000
        writer.write(dict(type='boundary',key=f'b{i}',status='round_ended',round_id=i,scores=scores.copy()))
    writer.write(dict(type='boundary',key='end',status='game_ended',round_id=1,scores=scores.copy()))
    return writer.publish(scores,[1,2,3,4])


def setup_run(tiny,tmp_path):
    actor,settings,profile=tiny
    rl=resolve(overrides=['ppo.max_updates=2','ppo.epochs=2','ppo.actor_lr=0.0001',
        'critic.oracle_width=8','critic.value_width=8','control.min_probe_samples=1',
        'control.probe_size=2','control.full_readback_rounds=[]'])
    origin=dict(profile=profile,fixture='not-real-match')
    config=create_settings(rl,settings,origin,tmp_path,'cpu')
    context=runs.create_run(config)
    source=RLTrainSource(context.run_dir,rl,origin)
    runtime=make_runtime(actor,config)
    metadata=c.RunMetadata('a'*64,'b'*64,{},(source.descriptor(),runtime.descriptor()),digest(config),{'train_seed':42})
    runs.publish_metadata(context,config,metadata,argv=[]);runs.activate(context)
    bc=publish_actor(context.run_dir/'actors',actor,settings,profile,origin)
    source.state['pool']=initial_pool(bc);source.state['stage']='JOINT';runtime.raw_model.set_stage('JOINT')
    observer=LocalObserver(context,config,background=False)
    controller=RLController(context=context,config=config,metadata=metadata,runtime=runtime,source=source,
        observer=observer,store=FileCheckpointStore(),stop=SignalStopToken())
    runtime.seed_training(42);controller.save()
    loaded=load_actor(context.run_dir/'actors'/bc['artifact_id'])
    ref=fixture_episode(context.run_dir,loaded,runtime.raw_model,rl)
    reader=EpisodeReader(context.run_dir,ref,rl['collection'])
    bundle=build_targets([reader],runtime.raw_model,loaded,stage='JOINT',lam=1,
        max_bytes=rl['collection']['max_target_bytes'])
    source.prepare([ref],bundle);controller.save()
    return controller,observer,loaded


def close(controller,observer):
    observer.close(c.Completion('completed',0,controller.progress.global_step,controller.last_checkpoint,None,None))


def test_finite_source_targets_and_full_recovery(tiny,tmp_path):
    controller,observer,loaded=setup_run(tiny,tmp_path)
    try:
        prepared=controller.store.load(controller.last_checkpoint,run_dir=controller.context.run_dir)
        targets=controller.source.state_dict()['target_bundle']
        before={k:v.clone() for k,v in controller.model.actor.state_dict().items()}
        controller.learn()
        assert any(not torch.equal(v,controller.model.actor.state_dict()[k]) for k,v in before.items())
        first=controller.runtime.export_state()
        assert controller.source.state['target_bundle']==targets
        controller.runtime.import_state(prepared.runtime)
        controller.source.load_state_dict(prepared.data)
        controller.runtime.restore_rng(prepared.rng)
        controller.model.set_stage('JOINT')
        controller.learn()
        controller._compare(first,controller.runtime.export_state())
        # Fresh process uses the real RL source, runtime and public checkpoint loader.
        output=tmp_path/'new-process.pt'
        code='''
import sys,torch
from pathlib import Path
from mahjong.training.checkpoint import FileCheckpointStore
from mahjong.app.rl import make_runtime
from mahjong.models.qwen3 import Qwen3PolicyAdapter,build_qwen3
from mahjong.rl.source import RLTrainSource
from mahjong.training.update import execute_update
root=Path(sys.argv[1]);store=FileCheckpointStore();ref=store.resolve(Path(sys.argv[2]));s=store.load(ref,run_dir=root)
actor=Qwen3PolicyAdapter(build_qwen3(s.config['model'],init_seed=0),input_schema=s.config['model']['input_schema'])
runtime=make_runtime(actor,s.config);runtime.import_state(s.runtime);runtime.restore_rng(s.rng)
source=RLTrainSource(root,s.config['rl'],s.config['rl_origin']);source.load_state_dict(s.data);runtime.raw_model.set_stage(source.state['stage'])
for _ in range(2):
 lease=source.acquire();runtime.zero_grad();result=runtime.compute(lease.batch,phase='train')
 execute_update(runtime,result,groups=runtime.raw_model.parameter_groups(),limits={'actor':.5,'critic':.5})
 source.commit(lease.token);runtime.zero_grad()
torch.save({'model':runtime.export_state().model,'optimizer':runtime.export_state().optimizer},sys.argv[3])
'''
        process=subprocess.run([sys.executable,'-c',code,str(controller.context.run_dir),
            str(controller.context.run_dir/controller.last_checkpoint.path),str(output)],capture_output=True,text=True,timeout=60)
        assert process.returncode==0,process.stderr
        result=torch.load(output,weights_only=True)
        controller._compare(first.model,result['model']);controller._compare(first.optimizer,result['optimizer'])
        # finite epochs and release semantics
        source=controller.source
        while True:
            try: lease=source.acquire()
            except StopIteration: break
            cursor=source.state['cursor'];source.release(lease.token)
            assert source.state['cursor']==cursor
            lease=source.acquire();source.commit(lease.token)
        assert source.state['cursor']==4
    finally: close(controller,observer)


@pytest.mark.parametrize('phase',['PREPARED','READY'])
def test_rename_failure_protects_real_committed_dependencies(tiny,tmp_path,phase):
    controller,observer,loaded=setup_run(tiny,tmp_path)
    try:
        if phase=='READY':
            controller.learn()
            from mahjong.rl.pool import promote
            ref=publish_actor(controller.context.run_dir/'actors',controller.model.actor,
                controller.config['model'],controller.config['rl_origin']['profile'],{'base':'fixture'})
            pool=promote(controller.source.state['pool'],ref,1,controller.rl['pool'])
            controller.source.ready(pool,'JOINT',0)
        def fail(stage,paths):
            if stage=='after_rename_before_latest': raise OSError('injected latest failure')
        controller.store.hook=fail
        with pytest.raises(c.MahjongError,match='publication failed'):controller.save()
        restored=controller.store.load(controller.last_checkpoint,run_dir=controller.context.run_dir)
        assert restored.data['phase']==phase
        assert len(list((controller.context.run_dir/'checkpoints').glob('*/manifest.json')))==2
        latest=reconcile(controller.store,controller.context.run_dir,controller.rl['collection'])
        assert latest==controller.last_checkpoint
        for actor in [restored.data['pool']['latest'],restored.data['pool']['bc']]:
            load_actor(controller.context.run_dir/'actors'/actor['artifact_id'],expected=actor)
        # A missing dependency is fatal, never silently resolved via old READY.
        import shutil
        shutil.rmtree(controller.context.run_dir/'actors'/restored.data['pool']['latest']['artifact_id'])
        with pytest.raises(c.MahjongError):reconcile(controller.store,controller.context.run_dir,controller.rl['collection'])
    finally: close(controller,observer)


def test_incomplete_or_changed_targets_rejected(tiny,tmp_path):
    controller,observer,_=setup_run(tiny,tmp_path)
    try:
        state=controller.source.state_dict();state['target_bundle']['targets'][0]['old_logprob']+=1
        with pytest.raises(c.MahjongError,match='binding'):controller.source.load_state_dict(state)
    finally:close(controller,observer)


def test_candidate_gate_registers_observer_and_rejection_does_not_publish(tiny,tmp_path,monkeypatch):
    controller,observer,loaded=setup_run(tiny,tmp_path)
    stats=dict(samples=10,ev=.5,rmse=.1,bias=0.,matches=4)
    monkeypatch.setattr('mahjong.rl.controller.evaluate_values',lambda *a,**k:stats)
    monkeypatch.setattr('mahjong.rl.controller.paired_evaluate',lambda *a,**k:dict(accepted=True,value=stats,attempts=8))
    try:
        controller.learn()
        controller.validate_candidate()
        assert controller.source.state['phase']=='READY'
        assert controller.progress.last_evaluation_ref in observer._reports
        # Reach a fresh PREPARED then reject the candidate after a real update.
        loaded=controller.actor(controller.source.state['pool']['latest'])
        ref=fixture_episode(controller.context.run_dir,loaded,controller.model,controller.rl,'second')
        bundle=build_targets([EpisodeReader(controller.context.run_dir,ref,controller.rl['collection'])],controller.model,
            controller.actor(controller.source.state['pool']['bc']),stage='JOINT',lam=1,max_bytes=1000000)
        controller.source.prepare([ref],bundle);controller.save()
        before=controller.last_checkpoint
        controller.learn()
        monkeypatch.setattr('mahjong.rl.controller.paired_evaluate',lambda *a,**k:dict(accepted=False,value=stats,attempts=8))
        with pytest.raises(c.MahjongError,match='rejected'):controller.validate_candidate()
        assert controller.last_checkpoint==before
        assert controller.store.load(before,run_dir=controller.context.run_dir).data['phase']=='PREPARED'
    finally:close(controller,observer)


def test_warmup_no_actor_change_or_optimizer_momentum(tiny,tmp_path):
    controller,observer,loaded=setup_run(tiny,tmp_path)
    try:
        reader=next(iter(controller.source.readers.values()))
        bundle=build_targets([reader],controller.model,loaded,stage='WARMUP',lam=.95,max_bytes=1000000)
        state=controller.source.state_dict();state['stage']='WARMUP';state['target_bundle']=bundle
        controller.source.load_state_dict(state);controller.model.set_stage('WARMUP')
        original={k:v.clone() for k,v in controller.model.actor.state_dict().items()}
        controller.learn()
        controller._compare(original,controller.model.actor.state_dict())
        assert all(p not in controller.runtime.raw_optimizer.state for p in controller.model.actor.parameters())
    finally:close(controller,observer)
