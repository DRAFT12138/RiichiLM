"""Single-process update boundaries, evaluation and checkpoint scheduling."""

from datetime import datetime, timezone
import hashlib
import time
import uuid

from mahjong import contracts as c
from mahjong.training.state import copy_progress, seeded_rng


class Trainer:
    def run(self, *, context: c.RunContext, metadata: c.RunMetadata, config: c.ResolvedConfig,
            source: c.TrainSource, validation_source: c.EvalSource, runtime: c.Runtime,
            evaluator: c.Evaluator, observer: c.Observer, store: c.CheckpointStore,
            stop: c.StopToken, resume: c.ResumePlan | None = None) -> c.Completion:
        self.context, self.metadata, self.config = context, metadata, config
        self.source, self.validation_source, self.runtime = source, validation_source, runtime
        self.evaluator, self.observer, self.store, self.stop = evaluator, observer, store, stop
        self.target = config["training"]["max_updates"]
        configure = getattr(runtime, 'configure_training_execution', None)
        if configure is not None:
            configure(self.target - (resume.snapshot.progress.global_step if resume else 0))
        self.validation = config["validation"]
        self.checkpoint_interval = config["checkpoint"]["every_updates"]
        self.checkpoint_seconds = config["checkpoint"].get("every_seconds")
        self.progress = c.TrainProgress(
            initial_validation_done=not (self.validation["enabled"] and self.validation["at_start"]),
            next_validation_step=self.validation["every_updates"] if self.validation["enabled"] else None,
            next_checkpoint_step=self.checkpoint_interval,
        )
        self.last_checkpoint = context.source_checkpoint
        self._lineage = ()
        self._commit_seq = 0
        self._safe = False
        self._phase = "restoring" if resume else "initializing"
        self._stop_logged = False
        self._cpu_lookahead = None
        self._config_hash = hashlib.sha256(c.canonical_json(config).encode()).hexdigest()
        try:
            if resume is not None:
                self._set_phase("restoring")
                self.observer.event("resume_started", {"checkpoint": c.encode_json_contract(resume.source)}, self.progress)
                self.runtime.import_state(resume.snapshot.runtime)
                self.source.load_state_dict(resume.snapshot.data)
                self.progress = copy_progress(resume.snapshot.progress)
                self.observer.load_state_dict(resume.snapshot.observation.window, resume.source)
                self._lineage = resume.snapshot.observation.lineage + (resume.snapshot.observation.cutoff,)
                self.last_checkpoint = resume.source
                # Nothing after this point initializes a training random stream.
                self.runtime.restore_rng(resume.snapshot.rng)
                self.observer.event("resume_completed", {"source": c.encode_json_contract(resume.source),
                                                         "target_updates": self.target}, self.progress)
            else:
                self.runtime.restore_rng(seeded_rng(metadata.derived_seeds["train_seed"], self.runtime.capture_rng()))
            # Monotonic deadlines belong to this live attempt. Downtime is not
            # training time and must not cause catch-up saves after recovery.
            self._reset_checkpoint_deadline()
            if resume is not None and any(change.path.startswith('checkpoint.') for change in resume.changes):
                interval = self.checkpoint_interval
                self.progress.next_checkpoint_step = (
                    (self.progress.global_step // interval + 1) * interval if interval is not None else None)
            if resume is not None and any(change.path == 'validation.every_updates' for change in resume.changes):
                interval = self.validation['every_updates'] if self.validation['enabled'] else None
                self.progress.next_validation_step = (
                    (self.progress.global_step // interval + 1) * interval if interval is not None else None)
            self._last_rates = self.runtime.learning_rates()
            if self.progress.global_step > self.target:
                raise c.MahjongError("RESUME_INCOMPATIBLE", "Target updates are behind saved progress", "restoring")
            self._safe = not self.source.has_open_lease()
            if not self._safe:
                raise c.MahjongError("CONTRACT_MISMATCH", "Run starts with an open batch lease", "training")
            self._configure_cpu_lookahead()
            while True:
                if self.stop.requested():
                    return self._interrupt()
                if self.progress.global_step == self.target:
                    return self._finish()
                if not self.progress.initial_validation_done:
                    self.progress.pending_validation = c.PendingValidation("initial", self.progress.global_step)
                    if not self._evaluate(self.progress.pending_validation):
                        return self._interrupt()
                    if self.stop.requested():
                        continue
                if self.progress.pending_validation is not None:
                    if not self._evaluate(self.progress.pending_validation):
                        return self._interrupt()
                    if self.stop.requested():
                        continue
                threshold = self.progress.next_checkpoint_step
                step_due = threshold is not None and self.progress.global_step >= threshold
                time_due = self._checkpoint_deadline is not None and time.monotonic() >= self._checkpoint_deadline
                if step_due or time_due:
                    self._save("periodic")
                    if self.stop.requested():
                        continue
                if self.stop.requested():
                    continue
                self._update()
        except Exception as exc:
            if not isinstance(exc, c.MahjongError):
                exc = c.MahjongError("CONTRACT_MISMATCH", str(exc), self._phase)
            # A contract failure raised by a model during running is a runtime
            # failure, even if its generic constructor used initializing.
            if self._phase in {"training", "evaluating", "checkpointing", "finalizing"}:
                exc.phase = self._phase
            exc.details.update({"progress": c.encode_json_contract(self.progress),
                                "last_checkpoint": None if self.last_checkpoint is None else c.encode_json_contract(self.last_checkpoint),
                                "safe_to_checkpoint": self._safe})
            raise exc
        finally:
            self._clear_cpu_lookahead()

    def _configure_cpu_lookahead(self):
        from mahjong.training.lookahead import CPUBatchLookahead
        self._cpu_lookahead = CPUBatchLookahead.create(self.source, self.runtime)

    def _clear_cpu_lookahead(self):
        lookahead = getattr(self, '_cpu_lookahead', None)
        if lookahead is not None:
            lookahead.clear()

    def _prepare_next_cpu_batch(self):
        if self.stop.requested():
            return
        next_step = self.progress.global_step + 1
        if next_step >= self.target:
            return
        if (self.progress.next_validation_step is not None and
                next_step >= self.progress.next_validation_step):
            return
        if (self.progress.next_checkpoint_step is not None and
                next_step >= self.progress.next_checkpoint_step):
            return
        if self._checkpoint_deadline is not None and time.monotonic() >= self._checkpoint_deadline:
            return
        self._cpu_lookahead.prepare()

    def _set_phase(self, phase: c.Phase) -> None:
        if phase != 'training':
            self._clear_cpu_lookahead()
        self._phase = phase
        self.observer.phase(phase, self.progress)

    def _update(self) -> None:
        self._set_phase("training")
        started = time.monotonic()
        try:
            lease = self.source.acquire()
        except StopIteration as exc:
            code = "EMPTY_TRAIN_DATA" if self.progress.batches_read == 0 else "DATA_READ_FAILED"
            raise c.MahjongError(code, "Training source ended", "training") from exc
        except c.MahjongError:
            raise
        except Exception as exc:
            raise c.MahjongError("DATA_READ_FAILED", str(exc), "training") from exc
        c.validate_contract(lease)
        self.progress.batches_read += 1
        self.progress.samples_read += lease.batch.sample_count
        self._safe = False
        self.runtime.zero_grad()
        result = self.runtime.compute(lease.batch, phase="train")
        c.validate_step_result(result, phase="train")
        self.progress.samples_processed += lease.batch.sample_count
        rates = self._last_rates
        duration = None
        if result.skip_reason is None:
            rates = self.runtime.learning_rates()
            from mahjong.training.update import execute_update
            lookahead = getattr(self, '_cpu_lookahead', None)
            result = execute_update(self.runtime, result,
                after_backward=self._prepare_next_cpu_batch if lookahead is not None else None)
            duration = time.monotonic() - started
            self.progress.global_step += 1
            self.progress.samples_updated += lease.batch.sample_count
            self.progress.consecutive_skips = 0
            self._last_rates = rates
        else:
            self.progress.skipped_batches += 1
            self.progress.consecutive_skips += 1
        self.source.commit(lease.token)
        self.runtime.zero_grad()
        self._safe = True
        self.observer.record_step(result, self.progress, c.Timing(duration), rates)
        self.observer.publish_resources(self.runtime.resource_snapshot())
        if result.skip_reason is not None:
            self.observer.event("update_skipped", {"reason": result.skip_reason}, self.progress)
            if self.progress.consecutive_skips >= self.config["training"]["max_consecutive_skips"]:
                raise c.MahjongError("NO_PROGRESS", "Consecutive skip limit reached", "training")
        threshold = self.progress.next_validation_step
        if (threshold is not None and self.progress.global_step >= threshold
                and self.progress.pending_validation is None):
            self.progress.pending_validation = c.PendingValidation("periodic", self.progress.global_step)

    def _evaluate(self, pending: c.PendingValidation) -> bool:
        self._set_phase("evaluating")
        evaluation_id = uuid.uuid4().hex
        request = c.EvaluationRequest(
            evaluation_id, pending.purpose,
            c.ModelSource(self.context.run_id, self.context.attempt_id, self.progress.global_step, None),
            self.config["evaluation"]["seed"], self.config["evaluation"]["batch_size"],
            self.config["runtime"]["precision"],
            self.context.run_dir / "reports" / self.context.attempt_id / f"{evaluation_id}.json",
            self._config_hash, f"attempts/{self.context.attempt_id}/manifest.json",
        )
        if self.validation["enabled"]:
            report = self.evaluator.run(self.runtime, self.validation_source, request, self.observer, self.stop)
        else:
            report = self.evaluator.disabled(self.validation_source, request, self.observer)
        c.validate_contract(report)
        if report.status == "cancelled":
            return False
        if report.status == "failed":
            raise c.MahjongError("EVALUATION_FAILED", report.reason or "Evaluation failed", "evaluating")
        if (report.status == "disabled") == self.validation["enabled"]:
            raise c.MahjongError("EVALUATION_FAILED", "Evaluation status disagrees with validation.enabled", "evaluating")
        self.progress.last_evaluation_ref = report.report_ref
        if report.status == "ok":
            self.progress.last_validated_step = self.progress.global_step
        if pending.purpose == "initial":
            self.progress.initial_validation_done = True
        else:
            interval = self.validation["every_updates"] if self.validation["enabled"] else None
            self.progress.next_validation_step = ((self.progress.global_step // interval + 1) * interval
                                                  if interval is not None else None)
        self.progress.pending_validation = None
        if pending.purpose == "final" and self.progress.global_step == self.target and report.status == "not_evaluable":
            raise c.MahjongError("EVALUATION_FAILED", "Final enabled validation is not evaluable", "evaluating")
        return True

    def _finish(self) -> c.Completion:
        self._set_phase("finalizing")
        # Target handling consumes initial/periodic pending at this step.
        self.progress.initial_validation_done = True
        self.progress.pending_validation = c.PendingValidation("final", self.progress.global_step)
        if not self._evaluate(self.progress.pending_validation):
            return self._interrupt()
        if self.stop.requested():
            return self._interrupt()
        self._save("final", partial=True)
        if self.stop.requested():
            return self._interrupt()
        return c.Completion("completed", 0, self.progress.global_step, self.last_checkpoint,
                            self.progress.last_evaluation_ref, None)

    def _interrupt(self) -> c.Completion:
        if not self._stop_logged:
            self.observer.event("stop_requested", {"signal_number": self.stop.signal_number()}, self.progress)
            self._stop_logged = True
        if self._safe:
            self._save("stop", partial=True)
        return c.Completion("interrupted", 128 + (self.stop.signal_number() or 2), self.progress.global_step,
                            self.last_checkpoint, self.progress.last_evaluation_ref, "stop_requested")

    def _save(self, reason: str, *, partial=False) -> None:
        if not self._safe or self.source.has_open_lease():
            raise c.MahjongError("CONTRACT_MISMATCH", "Checkpoint requested outside a complete update boundary", "checkpointing")
        if partial:
            self.observer.write_partial(self.progress)
        self._set_phase("checkpointing")
        self.observer.event("checkpoint_started", {"reason": reason}, self.progress)
        cutoff = self.observer.flush(durable=True)
        if cutoff is None or cutoff.attempt_id != self.context.attempt_id:
            raise c.MahjongError("CONTRACT_MISMATCH", "Durable training flush must return this attempt cutoff", "checkpointing")
        snapshot = self.collect_snapshot(reason, cutoff)
        try:
            reference = self.store.save(snapshot, self.context)
        except Exception as exc:
            self.observer.event("checkpoint_failed", {"reason": reason, "error": str(exc)}, self.progress)
            if isinstance(exc, c.MahjongError):
                raise
            raise c.MahjongError("CHECKPOINT_WRITE_FAILED", str(exc), "checkpointing") from exc
        self.last_checkpoint = reference
        self.progress.next_checkpoint_step = snapshot.progress.next_checkpoint_step
        self._reset_checkpoint_deadline()
        self.observer.event("checkpoint_saved", {"checkpoint": c.encode_json_contract(reference), "reason": reason}, self.progress)

    def _reset_checkpoint_deadline(self):
        self._checkpoint_deadline = (time.monotonic() + self.checkpoint_seconds
                                     if self.checkpoint_seconds is not None else None)

    def collect_snapshot(self, reason: str, cutoff: c.LogCutoff) -> c.Snapshot:
        self._commit_seq += 1
        progress = copy_progress(self.progress)
        interval = self.checkpoint_interval
        progress.next_checkpoint_step = ((progress.global_step // interval + 1) * interval if interval is not None else None)
        identity = c.SnapshotIdentity(
            self.context.run_id, self.context.attempt_id, uuid.uuid4().hex, progress.global_step,
            self._commit_seq, datetime.now(timezone.utc).isoformat(), self.last_checkpoint,
            self.metadata.code_fingerprint, self.metadata.dependency_fingerprint,
            c.clone_state(self.metadata.environment),
        )
        return c.Snapshot(1, 1, identity, c.clone_state(self.config), self.metadata.frozen_config_hash,
                          self.metadata.descriptors, self.runtime.export_state(), progress,
                          self.source.state_dict(), self.runtime.capture_rng(),
                          c.ObservationState(self.observer.state_dict(), cutoff, self._lineage), reason, self.target)
