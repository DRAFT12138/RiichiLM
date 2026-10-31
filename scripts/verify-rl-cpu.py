#!/usr/bin/env python3
"""Bounded CPU P0 probe: two real complete matches and real Actor updates.

Random micro-model fixture, not a pretrained NTP provenance or A0 quality gate
acceptance. Default artifacts are on tmpfs; no GPU discovery or allocation.
"""
import os
for name in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):
    os.environ[name] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import tempfile
import torch
from mahjong import contracts as c
from mahjong.app import assembly,runs
from mahjong.app.config import parse_model
from mahjong.app.rl import create_settings,make_runtime
from mahjong.inference.actor_loading import publish_actor,load_actor
from mahjong.inference.policy import digest
from mahjong.models.qwen3 import build_qwen3,Qwen3PolicyAdapter,SETTINGS_FIELDS
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
from mahjong.rl.config import resolve
from mahjong.rl.collector import collect
from mahjong.rl.controller import RLController
from mahjong.rl.source import RLTrainSource
from mahjong.rl.storage import EpisodeReader
from mahjong.rl.targets import build_targets,episode_ledger
from mahjong.rl.pool import initial_pool,promote
from mahjong.training.checkpoint import FileCheckpointStore
from mahjong.training.signals import SignalStopToken
from mahjong.observability.events import LocalObserver


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=Path(args.output or tempfile.mkdtemp(prefix='majang-rl-cpu-',dir='/dev/shm'))
    output.mkdir(exist_ok=True,parents=True)
    base=parse_model(['model.num_hidden_layers=1','model.hidden_size=8','model.intermediate_size=16',
        'model.num_attention_heads=1','model.num_key_value_heads=1','model.head_dim=8','model.max_position_embeddings=8192'])
    settings={k:base['model'][k] for k in SETTINGS_FIELDS};settings['input_schema']=CONTEXT_SCHEMA
    actor=Qwen3PolicyAdapter(build_qwen3(settings,init_seed=42),input_schema=CONTEXT_SCHEMA)
    profile=json.loads(Path('configs/inference-profile.json').read_text())
    rl=resolve(overrides=['ppo.max_updates=2','ppo.actor_lr=0.0001','critic.oracle_width=8','critic.value_width=8',
        'control.min_probe_samples=1','control.probe_size=8','control.full_readback_rounds=[0,1,2]',
        'collection.max_seconds=600.0'])
    origin=dict(profile=profile,fixture='random-micro-actor; P0 only; no NTP provenance or A0 gate acceptance')
    config=create_settings(rl,settings,origin,output,'cpu')
    context=runs.create_run(config)
    report=dict(scope=origin['fixture'],device='cpu',gpu_used=False,rounds=[],run=str(context.run_dir))
    with runs.lock_run(context.run_dir):
        runtime=make_runtime(actor,config);runtime.raw_model.set_stage('JOINT')
        source=RLTrainSource(context.run_dir,rl,origin);source.state['stage']='JOINT'
        metadata=assembly.collect_metadata(config,runtime,source)
        runs.publish_metadata(context,config,metadata,argv=['scripts/verify-rl-cpu.py']);runs.activate(context)
        store=FileCheckpointStore();stop=SignalStopToken();observer=LocalObserver(context,config,background=False)
        controller=RLController(context=context,config=config,metadata=metadata,runtime=runtime,source=source,
            observer=observer,store=store,stop=stop)
        bc_ref=publish_actor(context.run_dir/'actors',actor,settings,profile,origin)
        source.state['pool']=initial_pool(bc_ref)
        runtime.seed_training(42);controller.save()
        try:
            for iteration in range(2):
                ref=source.state['pool']['latest']
                loaded=load_actor(context.run_dir/'actors'/ref['artifact_id'],expected=ref)
                bc=load_actor(context.run_dir/'actors'/bc_ref['artifact_id'],expected=bc_ref)
                result=collect(context.run_dir,[loaded,bc,bc,bc],root_seed=42,episode=f'p0-{iteration}',
                    learner=0,limits=rl['collection'],header=dict(initial_scores=profile['initial_scores'],fixture=origin['fixture']))
                print(json.dumps(dict(iteration=iteration,collection=result)),flush=True)
                if result['status']!='complete':raise RuntimeError(result)
                reader=EpisodeReader(context.run_dir,result['ref'],rl['collection'])
                keys,rewards,pre=episode_ledger(reader)
                bundle=build_targets([reader],runtime.raw_model,bc,stage='JOINT',lam=1,max_bytes=rl['collection']['max_target_bytes'])
                source.prepare([result['ref']],bundle);controller.save()
                before={k:v.clone() for k,v in runtime.raw_model.actor.state_dict().items()}
                controller.learn()
                changed=any(not torch.equal(v,runtime.raw_model.actor.state_dict()[k]) for k,v in before.items())
                if not changed: raise RuntimeError('Actor did not update')
                candidate=publish_actor(context.run_dir/'actors',runtime.raw_model.actor,settings,profile,
                    dict(base=c.encode_json_contract(controller.last_checkpoint),fixture=origin['fixture']))
                controller.publish_report(dict(scope=origin['fixture'],accepted=True,actor_changed=changed,collection=result))
                source.ready(promote(source.state['pool'],candidate,iteration+1,rl['pool']),'JOINT',0)
                controller.save('final' if iteration==1 else 'periodic')
                saved=store.load(controller.last_checkpoint,run_dir=context.run_dir)
                controller._compare(runtime.export_state(),saved.runtime)
                report['rounds'].append(dict(collection=result,learner_decisions=len(keys),reward=sum(rewards),pre_decision=pre,
                    actor_before=ref,actor_after=candidate,actor_changed=changed,full_state_readback=True,
                    checkpoint=c.encode_json_contract(controller.last_checkpoint)))
            report.update(status='passed',optimizer_steps=controller.progress.global_step,source_fingerprint=metadata.code_fingerprint)
        except Exception as exc:
            report.update(status='failed',error=str(exc));raise
        finally:
            runs.atomic_json(output/'report.json',report)
            observer.close(c.Completion('completed' if report.get('status')=='passed' else 'failed',
                0 if report.get('status')=='passed' else 1,controller.progress.global_step,controller.last_checkpoint,controller.progress.last_evaluation_ref,None))
    print(f'report={output / "report.json"}',flush=True)

if __name__=='__main__':main()
