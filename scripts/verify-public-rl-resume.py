#!/usr/bin/env python3
"""Public RL cold-process resume with real micro-Qwen3/Adam state.

Two scripted decisions exercise PREPARED replay and joint Actor/Critic updates.
The candidate stops after learning, before quality evaluation. This fixture is
neither a complete match nor a quality/strength acceptance run. All artifacts
belong to a new disposable run, never to an existing user experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import torch

from mahjong import contracts as c
from mahjong.app import assembly, rl as app, runs
from mahjong.app.config import parse_model
from mahjong.inference.actor_loading import publish_actor, load_actor
from mahjong.models.qwen3 import Qwen3PolicyAdapter, build_qwen3, SETTINGS_FIELDS
from mahjong.observability.events import LocalObserver
from mahjong.rl.config import resolve
from mahjong.rl.controller import RLController
from mahjong.rl.pool import initial_pool
from mahjong.rl.source import RLTrainSource
from mahjong.rl.storage import EpisodeReader
from mahjong.rl.targets import build_targets
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
from mahjong.training.checkpoint import FileCheckpointStore
from mahjong.training.signals import SignalStopToken
from tests.unit.rl.test_recovery import fixture_episode


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif is_dataclass(item):
            for field in fields(item):
                digest.update(field.name.encode())
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def mathematical_state(controller):
    return dict(runtime=fingerprint(controller.runtime.export_state()),
                rng=fingerprint(controller.runtime.capture_rng()),
                source=fingerprint(controller.source.state_dict()),
                step=controller.progress.global_step,
                samples=controller.progress.samples_updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'resume'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == 'prepare':
        base = parse_model(['model.num_hidden_layers=1', 'model.hidden_size=8',
            'model.intermediate_size=16', 'model.num_attention_heads=1',
            'model.num_key_value_heads=1', 'model.head_dim=8', 'model.max_position_embeddings=8192'])
        settings = {key: base['model'][key] for key in SETTINGS_FIELDS}
        settings['input_schema'] = CONTEXT_SCHEMA
        actor = Qwen3PolicyAdapter(build_qwen3(settings, init_seed=42), input_schema=CONTEXT_SCHEMA)
        profile = json.loads(Path('configs/inference-profile.json').read_text())
        rl = resolve(overrides=['ppo.max_updates=2', 'ppo.epochs=2', 'ppo.actor_lr=0.0001',
            'critic.oracle_width=8', 'critic.value_width=8', 'control.min_probe_samples=1',
            'control.probe_size=2', 'control.full_readback_rounds=[]'])
        origin = dict(profile=profile, fixture='synthetic cold-process P0; no quality claim')
        config = app.create_settings(rl, settings, origin, args.output.parent/'rl-runs', 'cpu')
        context = runs.create_run(config)
        runtime = app.make_runtime(actor, config)
        source = RLTrainSource(context.run_dir, rl, origin)
        metadata = assembly.collect_metadata(config, runtime, source)
        runs.publish_metadata(context, config, metadata, argv=['public-rl-prepare'])
        runs.activate(context)
        observer = LocalObserver(context, config, background=False)
        controller = RLController(context=context, config=config, metadata=metadata,
            runtime=runtime, source=source, observer=observer, store=FileCheckpointStore(), stop=SignalStopToken())
        try:
            ref = publish_actor(context.run_dir/'actors', actor, settings, profile, origin)
            source.state.update(pool=initial_pool(ref), stage='JOINT')
            runtime.raw_model.set_stage('JOINT')
            runtime.seed_training(42)
            loaded = load_actor(context.run_dir/'actors'/ref['artifact_id'])
            episode = fixture_episode(context.run_dir, loaded, runtime.raw_model, rl)
            targets = build_targets([EpisodeReader(context.run_dir, episode, rl['collection'])],
                runtime.raw_model, loaded, stage='JOINT', lam=1, max_bytes=rl['collection']['max_target_bytes'])
            source.prepare([episode], targets)
            controller.save()
            checkpoint = context.run_dir/controller.last_checkpoint.path
            digest = hashlib.sha256((checkpoint/'state.pt').read_bytes()).hexdigest()
            before = fingerprint(runtime.raw_model.actor.state_dict())
            controller.learn()
            assert before != fingerprint(runtime.raw_model.actor.state_dict())
            report = dict(expected=mathematical_state(controller), checkpoint=str(checkpoint),
                original_state_sha256=digest, source_fingerprint=metadata.code_fingerprint,
                actor_updated=True, fixture=origin['fixture'])
        finally:
            observer.close(c.Completion('interrupted', 130, controller.progress.global_step,
                                        controller.last_checkpoint, None, None))
    else:
        reference = json.loads(args.reference.read_text())
        checkpoint = Path(reference['checkpoint'])
        report = dict(fixture=reference['fixture'])
        def stop_after_learning(controller):
            assert mathematical_state(controller) == reference['expected']
            assert controller.metadata.code_fingerprint == reference['source_fingerprint']
            report.update(actual=mathematical_state(controller), source_fingerprint=controller.metadata.code_fingerprint,
                          original_source_fingerprint=reference['source_fingerprint'], exact=True)
            controller.publish_report(dict(scope=reference['fixture'], accepted=True,
                                           purpose='bounded_replay_equality_only'))
            controller.save('final')
            return 'bounded_refactor_fixture_complete'
        original = RLController.validate_candidate
        RLController.validate_candidate = stop_after_learning
        try:
            code = app.resume(SimpleNamespace(run=None, checkpoint=str(checkpoint)),
                              stop=SignalStopToken(), argv=['public-rl-resume'])
            assert code == 0
        finally:
            RLController.validate_candidate = original
        assert hashlib.sha256((checkpoint/'state.pt').read_bytes()).hexdigest() == reference['original_state_sha256']
        report['original_checkpoint_unchanged'] = True
    report['status'] = 'passed'
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'{args.mode}: passed', flush=True)


if __name__ == '__main__':
    main()
