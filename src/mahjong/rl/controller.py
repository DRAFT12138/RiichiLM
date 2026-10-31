"""READY/PREPARED orchestration. Never checkpoint an unaccepted candidate."""
from datetime import datetime, timezone
import time
import uuid
import torch
from mahjong import contracts as c
from mahjong.app.runs import atomic_json
from mahjong.inference.actor_loading import load_actor, publish_actor
from mahjong.inference.policy import digest
from mahjong.rl.collector import collect
from mahjong.rl.pool import select, promote
from mahjong.rl.storage import EpisodeReader
from mahjong.rl.targets import build_targets, critic_digest
from mahjong.rl.diagnostics import check_ratios, probe_decision, value_gate
from mahjong.rl.evaluation import evaluate_values, paired_evaluate
from mahjong.rl.recovery import reconcile
from mahjong.training.update import execute_update
from mahjong.policy.execution import policy_precision, sampling_definition

class RLController:
    def __init__(self, *, context, config, metadata, runtime, source, observer, store, stop):
        self.context, self.config, self.metadata = context, config, metadata
        self.runtime, self.source, self.observer, self.store, self.stop = runtime, source, observer, store, stop
        self.rl, self.model = config['rl'], runtime.raw_model
        self.progress = c.TrainProgress(initial_validation_done=True)
        self.last_checkpoint = context.source_checkpoint
        self.commit_seq = 0
        self.lineage = ()
        self.cache = {}

    def actor(self, ref):
        key = ref['artifact_id']
        if key not in self.cache:
            # Only BC/latest/current opponent are needed; no whole-pool residency.
            if len(self.cache) >= 3: self.cache.clear()
            self.cache[key] = load_actor(self.context.run_dir/'actors'/key, expected=ref,
                                         device=str(self.runtime.device))
        return self.cache[key]

    def emit(self, name, value):
        self.observer.event('rl_'+name, value, self.progress)

    def check_stop(self):
        if self.stop.requested(): raise c.MahjongError('CANCELLED', 'RL stop requested')

    def save(self, reason='periodic'):
        self.observer.phase('checkpointing', self.progress)
        self.emit('checkpoint_started', dict(phase=self.source.state['phase']))
        cutoff = self.observer.flush(durable=True)
        self.commit_seq += 1
        identity = c.SnapshotIdentity(self.context.run_id, self.context.attempt_id, uuid.uuid4().hex,
            self.progress.global_step, self.commit_seq, datetime.now(timezone.utc).isoformat(), self.last_checkpoint,
            self.metadata.code_fingerprint, self.metadata.dependency_fingerprint, c.clone_state(self.metadata.environment))
        snapshot = c.Snapshot(1, 1, identity, c.clone_state(self.config), self.metadata.frozen_config_hash,
            self.metadata.descriptors, self.runtime.export_state(), self._progress_copy(),
            self.source.state_dict(), self.runtime.capture_rng(),
            c.ObservationState(self.observer.state_dict(), cutoff, self.lineage), reason,
            self.rl['rounds']*self.rl['ppo']['max_updates'])
        try:
            self.last_checkpoint = self.store.save(snapshot, self.context)
        except Exception:
            # A failed return does not imply a failed durable rename. On any
            # uncertainty this routine raises without deleting dependencies.
            self.last_checkpoint = reconcile(self.store, self.context.run_dir, self.rl['collection'], verify=True) or self.last_checkpoint
            raise
        reconcile(self.store, self.context.run_dir, self.rl['collection'], verify=False)
        self.emit('checkpoint_saved', dict(checkpoint=c.encode_json_contract(self.last_checkpoint), phase=self.source.state['phase']))
        if self.source.state['iteration_id'] in self.rl['control']['full_readback_rounds']:
            loaded = self.store.load(self.last_checkpoint, run_dir=self.context.run_dir)
            self._compare(snapshot.runtime, loaded.runtime)

    def _progress_copy(self):
        from mahjong.training.state import copy_progress
        return copy_progress(self.progress)

    @staticmethod
    def _compare(a, b):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or not torch.equal(a.cpu(), b.cpu()):
                raise c.MahjongError('CHECKPOINT_INVALID', 'full state tensor readback mismatch')
        elif isinstance(a, dict):
            if a.keys()!=b.keys(): raise c.MahjongError('CHECKPOINT_INVALID', 'state keys differ')
            for key in a: RLController._compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if len(a)!=len(b): raise c.MahjongError('CHECKPOINT_INVALID', 'state length differs')
            for x,y in zip(a,b): RLController._compare(x,y)
        elif hasattr(a, '__dataclass_fields__'):
            for key in a.__dataclass_fields__: RLController._compare(getattr(a,key), getattr(b,key))
        elif a!=b: raise c.MahjongError('CHECKPOINT_INVALID', 'state scalar differs')

    def restore(self, snapshot):
        self.runtime.import_state(snapshot.runtime)
        self.source.load_state_dict(snapshot.data)
        self.model.set_stage(self.source.state['stage'])
        if self.source.state['phase'] == 'PREPARED' and self.source.state['target_bundle']['critic_digest'] != critic_digest(self.model):
            raise c.MahjongError('CHECKPOINT_INVALID', 'PREPARED targets differ from the saved Critic')
        from mahjong.training.state import copy_progress
        self.progress = copy_progress(snapshot.progress)
        self.commit_seq = snapshot.identity.commit_seq
        self.observer.load_state_dict(snapshot.observation.window, self.last_checkpoint)
        self.lineage = snapshot.observation.lineage + (snapshot.observation.cutoff,)
        self.runtime.restore_rng(snapshot.rng)

    def collect_round(self):
        self.observer.phase('preparing_data', self.progress)
        state, limits = self.source.state, self.rl['collection']
        pool = state['pool']
        old, bc = self.actor(pool['latest']), self.actor(pool['bc'])
        refs, attempts, total_bytes = [], [], 0
        needed = limits['train_matches'] + limits['validation_matches']
        for _ in range(limits['max_attempts']):
            self.check_stop()
            sequence = state['episode_seq']; state['episode_seq'] += 1
            episode = f'train-{sequence}'
            table = select(pool, self.rl['seed'], sequence, self.rl['pool']['table_weights'])
            actors = [self.actor(table['opponent'])]*4; actors[table['learner']] = old
            result = collect(self.context.run_dir, actors, root_seed=self.rl['seed'], episode=episode,
                learner=table['learner'], limits=limits, cancel=self.stop.requested,
                header=dict(initial_scores=self.config['rl_origin']['profile']['initial_scores'],
                    run_id=self.context.run_id, attempt_id=self.context.attempt_id,
                    config_hash=self.metadata.frozen_config_hash, iteration_id=state['iteration_id'],
                    source_identity=self.metadata.code_fingerprint, native_identity=self.metadata.dependency_fingerprint,
                    reward='match_score_delta_hand_emission/v1', oracle='mahjong-oracle/v1', table=table))
            attempts.append(result); self.emit('collection_attempt', result)
            if result['status']=='cancelled': self.check_stop()
            if result['status']=='complete':
                refs.append(result['ref']); total_bytes += result['ref']['bytes']
                if total_bytes > limits['max_round_bytes']:
                    raise c.MahjongError('DATA_READ_FAILED', 'round trajectory byte budget exceeded')
            if len(refs)>=needed: break
        failures = sum(a['status']!='complete' for a in attempts)
        if len(refs)<needed or failures/len(attempts)>limits['max_failed_fraction']:
            raise c.MahjongError('DATA_READ_FAILED', 'collection completion/quality budget failed')
        train_refs, validation_refs = refs[:limits['train_matches']], refs[limits['train_matches']:]
        targets_started = time.monotonic()
        bundle = build_targets([EpisodeReader(self.context.run_dir, ref, limits) for ref in train_refs], self.model, bc,
            stage=state['stage'], lam=self.rl['ppo']['lam'], max_bytes=limits['max_target_bytes'],
            tolerance=self.rl['control']['score_tolerance'])
        self.emit('targets_built', dict(seconds=time.monotonic()-targets_started, records=len(bundle['targets']),
            advantage_std=bundle['advantage_std'], trajectory_bytes=total_bytes, attempts=len(attempts), failures=failures))
        state['validation_refs'] = validation_refs
        self.source.prepare(train_refs, bundle)
        self.save()

    def probe(self, indices):
        values, count = [], 0
        with torch.no_grad():
            for i in indices:
                result = self.runtime.compute(self.source.batch([i]), phase='eval')
                check_ratios(result, self.rl['control'])
                if 'old_kl' in result.metrics:
                    stat = result.metrics['old_kl']; values.append(stat.numerator); count += stat.denominator
        kl = sum(values)/count if count else 0.
        self.emit('post_update_probe', dict(old_kl=kl, samples=count, indices=indices))
        return probe_decision(kl, count, self.rl['control'])

    def learn(self):
        self.observer.phase('training', self.progress)
        learn_started = time.monotonic()
        targets = self.source.state['target_bundle']['targets']
        import random
        indices = [i for i,t in enumerate(targets) if t['policy_mask']]
        random.Random(c.derive_seed(self.rl['seed'], 'mahjong-rl/probe/v1')).shuffle(indices)
        indices = indices[:self.rl['control']['probe_size']]
        joint = self.source.state['stage']=='JOINT'
        if joint: self.probe(indices)
        count = 0
        while count < self.rl['ppo']['max_updates']:
            self.check_stop()
            try: lease = self.source.acquire()
            except StopIteration: break
            started = time.monotonic()
            self.runtime.zero_grad()
            result = self.runtime.compute(lease.batch, phase='train')
            result = execute_update(self.runtime, result, groups=self.model.parameter_groups(),
                limits=dict(actor=self.rl['control']['actor_grad_norm'],critic=self.rl['control']['critic_grad_norm']),
                before_step=lambda r: check_ratios(r, self.rl['control']))
            actor_update = any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in self.model.actor.parameters())
            if joint and self.rl['control']['strict_post_step']:
                with torch.no_grad(): check_ratios(self.runtime.compute(lease.batch, phase='eval'), self.rl['control'])
            self.source.commit(lease.token)
            self.runtime.zero_grad()
            count += 1
            self.progress.global_step += 1
            self.progress.batches_read += 1
            for name in ('samples_read','samples_processed','samples_updated'):
                setattr(self.progress, name, getattr(self.progress,name)+lease.batch.sample_count)
            self.source.state['actor_updates'] += int(actor_update)
            self.source.state['critic_updates'] += 1
            self.observer.record_step(result, self.progress, c.Timing(time.monotonic()-started), self.runtime.learning_rates())
            self.observer.publish_resources(self.runtime.resource_snapshot())
            epoch_end = self.source.state['cursor'] % len(targets)==0
            if joint and (count % self.rl['control']['probe_interval']==0 or epoch_end):
                if self.probe(indices):
                    self.emit('soft_kl_stop', dict(updates=count)); break
        if joint: self.probe(indices)
        self.emit('learning_completed', dict(updates=count, cursor=self.source.state['cursor'], total=len(self.source.order), seconds=time.monotonic()-learn_started))

    def publish_report(self, details, *, purpose='final'):
        """Register the RL report with the existing Observer completion protocol."""
        evaluation_id = uuid.uuid4().hex
        details_path = self.context.attempt_dir / f'rl-details-{evaluation_id}.json'
        atomic_json(details_path, details)
        report_path = self.context.attempt_dir / f'rl-report-{evaluation_id}.json'
        now = datetime.now(timezone.utc).isoformat()
        metrics = {}
        if 'value' in details:
            for name in ('rmse', 'bias', 'ev'):
                value = details['value'].get(name)
                if value is not None:
                    metrics['critic_'+name] = c.AggregatedMetric(float(value), 1, float(value), 'score', None)
        report = c.EvaluationReport(1, evaluation_id, 'ok' if details.get('accepted', True) else 'failed', purpose,
            c.ModelSource(self.context.run_id, self.context.attempt_id, self.progress.global_step, None),
            self.source.descriptor(), self.rl['evaluation']['monitor_seed'], 1, self.runtime.precision, 0,
            int(details.get('attempts', 0)), metrics, None, now, now, float(details.get('duration_seconds', 0.)), self.metadata.frozen_config_hash,
            f'attempts/{self.context.attempt_id}/manifest.json', str(report_path.relative_to(self.context.run_dir)))
        atomic_json(report_path, dict(report=c.encode_json_contract(report), details_ref=str(details_path.relative_to(self.context.run_dir))))
        self.observer.record_evaluation(report)
        self.progress.last_evaluation_ref = report.report_ref
        self.progress.last_validated_step = self.progress.global_step
        return report

    def validate_candidate(self):
        evaluation_started = time.monotonic()
        self.check_stop()
        self.observer.phase('evaluating', self.progress)
        state, pool = self.source.state, self.source.state['pool']
        old, bc = self.actor(pool['latest']), self.actor(pool['bc'])
        if state['stage']=='WARMUP':
            self._compare(self.model.actor.state_dict(), old.model.state_dict())
        iteration = state['iteration_id']
        if iteration in self.rl['control']['parameter_check_rounds']:
            delta = sum(float((p.detach().cpu()-old.model.state_dict()[k].cpu()).double().square().sum()) for k,p in self.model.actor.state_dict().items())**.5
            self.emit('actor_parameter_change', dict(norm=delta, iteration_id=iteration))
        heldout = evaluate_values([EpisodeReader(self.context.run_dir, r, self.rl['collection']) for r in state['validation_refs']],
            self.model, actor_ref=pool['latest'], expected_sampling=sampling_definition(policy_precision(self.model.actor)), scope='behavior_holdout')
        self.emit('value_diagnostics', heldout)
        # Snapshot provenance names the PREPARED base, never a future READY.
        candidate_ref = publish_actor(self.context.run_dir/'actors', self.model.actor, self.config['model'],
            self.config['rl_origin']['profile'], dict(base_checkpoint=c.encode_json_contract(self.last_checkpoint),
                through_step=self.progress.global_step, initialization=self.config['rl_origin']),
            max_files=self.rl['pool']['physical_capacity'], byte_budget=self.rl['pool']['byte_budget'])
        candidate = self.actor(candidate_ref)
        evaluation_config = c.clone_state(self.rl)
        used_bytes = sum(r['bytes'] for r in state['refs'] + state['validation_refs'])
        evaluation_config['collection']['max_round_bytes'] -= used_bytes
        report = paired_evaluate(self.context.run_dir, candidate, old, bc, model=self.model,
            config=evaluation_config, iteration=iteration, cancel=self.stop.requested,
            event=lambda result: self.emit('evaluation_match', result))
        report['behavior_holdout'] = heldout
        report['duration_seconds'] = time.monotonic()-evaluation_started
        self.publish_report(report)
        self.emit('candidate_evaluation', dict(report=self.progress.last_evaluation_ref, accepted=report['accepted']))
        if not report['accepted'] or 'value' not in report:
            raise c.MahjongError('EVALUATION_FAILED', 'candidate paired evaluation rejected')
        passed = value_gate(report['value'], report['value']['matches'], self.rl['critic'])
        windows = state['gate_windows']+1 if passed else 0
        stage = state['stage']
        if stage=='JOINT' and not passed:
            raise c.MahjongError('EVALUATION_FAILED', 'candidate same-policy value gate rejected')
        if stage=='WARMUP' and windows>=self.rl['critic']['consecutive_windows']:
            stage='JOINT'
        self.check_stop()
        next_pool = promote(pool, candidate_ref, iteration+1, self.rl['pool'])
        self.source.ready(next_pool, stage, windows)
        self.save('final' if self.source.state['iteration_id']>=self.rl['rounds'] else 'periodic')
        self.model.set_stage(stage)
        if stage=='WARMUP' and self.source.state['iteration_id']>=self.rl['critic']['warmup_rounds']:
            return 'warmup_not_ready'
        return None

    def run(self, snapshot=None):
        try:
            if snapshot is not None:
                self.restore(snapshot)
            else:
                self.runtime.seed_training(self.metadata.derived_seeds['train_seed'])
                self.save()
            while self.source.state['iteration_id'] < self.rl['rounds']:
                self.check_stop()
                self.model.set_stage(self.source.state['stage'])
                if self.source.state['phase']=='READY': self.collect_round()
                self.learn()
                reason = self.validate_candidate()
                if reason:
                    self.emit(reason, {})
                    return c.Completion('completed', 0, self.progress.global_step, self.last_checkpoint,
                                        self.progress.last_evaluation_ref, reason)
            if self.last_checkpoint.attempt_id != self.context.attempt_id:
                self.publish_report(dict(scope='already_completed_budget', accepted=True,
                    inherited_report=self.progress.last_evaluation_ref))
                self.save('final')
            self.observer.write_partial(self.progress)
            return c.Completion('completed', 0, self.progress.global_step, self.last_checkpoint,
                                self.progress.last_evaluation_ref, None)
        except Exception as exc:
            # Restore the last authoritative state in memory too. No candidate
            # checkpoint is written on cancellation, rejection or optimizer error.
            if self.last_checkpoint is not None:
                saved = self.store.load(self.last_checkpoint, run_dir=self.context.run_dir)
                self.runtime.import_state(saved.runtime)
                self.source._lease = None
                self.source.load_state_dict(saved.data)
                self.model.set_stage(saved.data['stage'])
                self.runtime.restore_rng(saved.rng)
                from mahjong.training.state import copy_progress
                self.progress = copy_progress(saved.progress)
            if isinstance(exc, c.MahjongError) and exc.code=='CANCELLED':
                return c.Completion('interrupted', 128+(self.stop.signal_number() or 2), self.progress.global_step,
                                    self.last_checkpoint, self.progress.last_evaluation_ref, 'stop_requested')
            if isinstance(exc, c.MahjongError):
                exc.details.update(progress=c.encode_json_contract(self.progress), last_checkpoint=c.encode_json_contract(self.last_checkpoint))
            raise
