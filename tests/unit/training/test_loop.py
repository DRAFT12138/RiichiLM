from dataclasses import replace
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from mahjong import contracts as c
from mahjong.training.loop import Trainer
from mahjong.training.runtime import FabricRuntime
from mahjong.training.signals import SignalStopToken
from tests.fixtures.contracts import doubles as d
from tests.unit.training.test_runtime import assert_equal


def config(n=23, val=50, save=50, *, enabled=True, at_start=True, skip_limit=32):
    return {"seed": 20261011, "training": {"max_updates": n, "max_consecutive_skips": skip_limit},
            "runtime": {"precision": "32-true"}, "validation": {"enabled": enabled, "at_start": at_start, "every_updates": val},
            "checkpoint": {"every_updates": save}, "evaluation": {"seed": 43, "batch_size": 2}}


class Source(d.KnownSource):
    def __init__(self, batch_size=32, on_commit=None):
        super().__init__()
        self.batch_size = batch_size
        self.on_commit = on_commit

    def acquire(self):
        if self.lease is not None:
            return super().acquire()
        ids = tuple(f"sample/{self.index * self.batch_size + i}" for i in range(self.batch_size))
        self.lease = c.BatchLease(c.Batch({"x": torch.linspace(1, 2, self.batch_size)}, ids, self.batch_size, d.SCHEMA), str(self.index))
        return self.lease

    def commit(self, token):
        super().commit(token)
        if self.on_commit:
            self.on_commit(self.index)


class Runtime(FabricRuntime):
    def __init__(self):
        super().__init__()
        self.updates = 0
        self.train_calls = self.eval_calls = 0

    def step(self):
        super().step()
        self.updates += 1

    def compute(self, batch, *, phase):
        if phase == "train":
            self.train_calls += 1
        else:
            self.eval_calls += 1
        return super().compute(batch, phase=phase)


class Evaluator(d.KnownEvaluator):
    def __init__(self, status=None, cancel=None):
        self.calls = []
        self.status = status or {}
        self.cancel = cancel

    def run(self, runtime, source, request, observer, stop):
        self.calls.append((request.purpose, request.model_source.global_step))
        if self.cancel == (request.purpose, request.model_source.global_step):
            stop.request(signal.SIGINT)
        report = super().run(runtime, source, request, observer, stop)
        if request.purpose in self.status:
            report = replace(report, status=self.status[request.purpose], metrics={}, reason="test_no_valid_statistics")
        return report

    def disabled(self, source, request, observer):
        self.calls.append(("disabled", request.model_source.global_step))
        return super().disabled(source, request, observer)


class Run:
    def __init__(self, tmp_path, cfg=None, *, source=None, model=None, evaluator=None, observer=None, store=None, stop=None, resume=None):
        self.config = cfg or config()
        self.source, self.model = source or Source(), model or d.KnownModel()
        self.runtime = Runtime()
        self.runtime.setup(self.model, torch.optim.Adam(self.model.parameters(), lr=.003))
        self.stop = stop or SignalStopToken()
        attempt = "A2" if resume else "A1"
        self.context = c.RunContext("R", attempt, tmp_path, tmp_path / "attempts" / attempt, "resume" if resume else "new", resume.source if resume else None)
        self.observer = observer or d.RecordingObserver(c.RecordIdentity("training", "R", attempt, None))
        self.evaluator, self.store = evaluator or Evaluator(), store or d.MemoryStore()
        self.trainer, self.resume = Trainer(), resume

    def execute(self):
        return self.trainer.run(context=self.context, metadata=d.make_metadata(), config=self.config,
                                source=self.source, validation_source=d.KnownEvalSource(), runtime=self.runtime,
                                evaluator=self.evaluator, observer=self.observer, store=self.store, stop=self.stop, resume=self.resume)

    def saved(self):
        return [snapshot for _, snapshot in self.store.items.values()]


@pytest.mark.parametrize("n", [0, 1, 200])
def test_exact_updates_and_sample_counts(tmp_path, n):
    run = Run(tmp_path, config(n))
    result = run.execute()
    assert result.outcome == "completed" and result.exit_code == 0
    assert result.global_step == run.runtime.updates == n
    progress = run.trainer.progress
    assert progress.batches_read == n
    assert progress.samples_read == progress.samples_processed == progress.samples_updated == 32*n
    assert progress.skipped_batches == progress.consecutive_skips == 0
    assert len([snap for snap in run.saved() if snap.save_reason == "final"]) == 1
    assert run.evaluator.calls[-1] == ("final", n)


@pytest.mark.parametrize("n,expected_eval,expected_save", [
    (120, [("initial", 0), ("periodic", 50), ("periodic", 100), ("final", 120)], [40, 80, 120]),
    (100, [("initial", 0), ("periodic", 50), ("final", 100)], [40, 80, 100]),
])
def test_validation_before_save_and_unique_final(tmp_path, n, expected_eval, expected_save):
    run = Run(tmp_path, config(n, val=50, save=40))
    run.execute()
    assert run.evaluator.calls == expected_eval
    assert [snap.progress.global_step for snap in run.saved()] == expected_save
    assert [snap.identity.commit_seq for snap in run.saved()] == [1, 2, 3]
    assert run.saved()[-1].progress.last_evaluation_ref is not None
    assert run.saved()[-1].identity.parent_checkpoint.global_step == 80


def test_periods_disabled_only_initial_final_and_final_save(tmp_path):
    run = Run(tmp_path, config(5, val=None, save=None))
    run.execute()
    assert run.evaluator.calls == [("initial", 0), ("final", 5)]
    assert len(run.saved()) == 1 and run.saved()[0].save_reason == "final"
    assert run.saved()[0].progress.next_checkpoint_step is None


@pytest.mark.parametrize("cfg,expected", [(config(0), [("final", 0)]),
                                         (config(3, at_start=False), [("final", 3)]),
                                         (config(3, enabled=False), [("disabled", 3)])])
def test_initial_disabled_and_zero_target_semantics(tmp_path, cfg, expected):
    run = Run(tmp_path, cfg)
    run.execute()
    assert run.evaluator.calls == expected
    assert run.trainer.progress.initial_validation_done
    if not cfg["validation"]["enabled"]:
        assert run.runtime.eval_calls == 0


@pytest.mark.parametrize("final_status", ["ok", "not_evaluable"])
def test_not_evaluable_initial_periodic_once_final_rejects(tmp_path, final_status):
    evaluator = Evaluator(status={"initial": "not_evaluable", "periodic": "not_evaluable", "final": final_status})
    run = Run(tmp_path, config(120, val=50, save=40), evaluator=evaluator)
    if final_status == "ok":
        assert run.execute().exit_code == 0
        assert run.trainer.progress.last_validated_step == 120
    else:
        with pytest.raises(c.MahjongError) as error:
            run.execute()
        assert error.value.code == "EVALUATION_FAILED" and error.value.exit_code == 1
        assert [snap.progress.global_step for snap in run.saved()] == [40, 80]
        assert error.value.details["last_checkpoint"]["global_step"] == 80
        assert error.value.details["progress"]["last_evaluation_ref"] is not None
    assert evaluator.calls == [("initial", 0), ("periodic", 50), ("periodic", 100), ("final", 120)]


class SkipModel(d.KnownModel):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = iter(pattern)

    def forward(self, batch, *, phase):
        if phase == "train" and next(self.pattern):
            return c.StepResult(None, {}, "no_target")
        return super().forward(batch, phase=phase)


def test_mixed_skip_exact_counts_stop_on_third_commit(tmp_path):
    stop = SignalStopToken()
    source = Source(2, on_commit=lambda index: stop.request(signal.SIGTERM) if index == 3 else None)
    run = Run(tmp_path, config(200), source=source, model=SkipModel([True, False, True]), stop=stop)
    result = run.execute()
    assert result.exit_code == 143 and result.global_step == run.runtime.updates == 1
    progress = run.trainer.progress
    assert (progress.samples_read, progress.samples_processed, progress.samples_updated) == (6, 6, 2)
    assert (progress.skipped_batches, progress.consecutive_skips) == (2, 1)
    assert run.saved()[-1].save_reason == "stop"
    assert len([call for call in run.observer.calls if call[0] == "record_step"]) == 3


@pytest.mark.parametrize("limit", [3, 32])
def test_all_skip_fails_on_configured_limit(tmp_path, limit):
    run = Run(tmp_path, config(200, skip_limit=limit), model=SkipModel([True]*limit))
    with pytest.raises(c.MahjongError) as error:
        run.execute()
    assert error.value.code == "NO_PROGRESS" and error.value.exit_code == 1
    assert run.trainer.progress.global_step == run.runtime.updates == 0
    assert run.trainer.progress.consecutive_skips == run.trainer.progress.skipped_batches == limit
    assert run.source.index == limit and run.saved() == []


def test_skip_limit_resets_after_success(tmp_path):
    run = Run(tmp_path, config(2, skip_limit=3), model=SkipModel([True, True, False, True, True, False]))
    assert run.execute().global_step == 2
    assert run.trainer.progress.skipped_batches == 4 and run.trainer.progress.consecutive_skips == 0


@pytest.mark.parametrize("purpose,step,target,raised_target", [("initial", 0, 5, 5), ("periodic", 2, 5, 5),
                                                              ("final", 5, 5, 5), ("final", 0, 0, 5)])
def test_cancelled_validation_preserved_and_replayed_on_resume(tmp_path, purpose, step, target, raised_target):
    run = Run(tmp_path, config(target, val=2, save=2), evaluator=Evaluator(cancel=(purpose, step)))
    result = run.execute()
    assert result.exit_code == 130
    snapshot = run.saved()[-1]
    assert snapshot.progress.pending_validation == c.PendingValidation(purpose, step)
    if purpose == "final":
        assert snapshot.progress.initial_validation_done
    ref = result.last_checkpoint
    cfg = config(raised_target, val=2, save=2)
    plan = c.ResumePlan(ref, snapshot, cfg, ())
    resumed = Run(tmp_path, cfg, resume=plan)
    done = resumed.execute()
    assert done.global_step == raised_target
    assert resumed.runtime.updates == raised_target-step
    assert resumed.evaluator.calls[0] == (purpose, step)
    assert resumed.saved()[0].identity.commit_seq == 1
    assert resumed.saved()[0].identity.parent_checkpoint == ref
    assert resumed.saved()[0].observation.lineage == snapshot.observation.lineage + (snapshot.observation.cutoff,)
    assert snapshot.progress.pending_validation == c.PendingValidation(purpose, step)  # No alias mutation.


@pytest.mark.parametrize("new_target", [5, 8])
def test_completed_checkpoint_resume_zero_or_only_remaining_updates(tmp_path, new_target):
    original = Run(tmp_path, config(5))
    result = original.execute()
    saved = original.saved()[-1]
    cfg = config(new_target)
    resumed = Run(tmp_path, cfg, resume=c.ResumePlan(result.last_checkpoint, saved, cfg, ()))
    resumed.execute()
    assert resumed.runtime.updates == new_target-5
    assert resumed.evaluator.calls == [("final", new_target)]
    assert len(resumed.saved()) == 1


def test_stop_before_first_update_complete_g0(tmp_path):
    stop = SignalStopToken()
    stop.request(signal.SIGINT)
    run = Run(tmp_path, config(5), stop=stop)
    result = run.execute()
    assert result.exit_code == 130 and result.global_step == 0
    assert run.evaluator.calls == [] and run.runtime.updates == 0
    assert run.saved()[0].progress.initial_validation_done is False
    assert run.saved()[0].save_reason == "stop"


class StopOnSave(d.MemoryStore):
    def __init__(self, stop, reason):
        super().__init__()
        self.stop, self.reason = stop, reason

    def save(self, snapshot, context):
        reference = super().save(snapshot, context)
        if snapshot.save_reason == self.reason:
            self.stop.request(signal.SIGTERM)
        return reference


@pytest.mark.parametrize("reason,target", [("periodic", 3), ("final", 2)])
def test_stop_during_save_finishes_transaction_then_saves_stop(tmp_path, reason, target):
    stop = SignalStopToken()
    run = Run(tmp_path, config(target, val=2, save=2), stop=stop, store=StopOnSave(stop, reason))
    result = run.execute()
    assert result.exit_code == 143
    assert [snap.save_reason for snap in run.saved()] == [reason, "stop"]
    assert [snap.identity.commit_seq for snap in run.saved()] == [1, 2]
    assert run.saved()[1].identity.parent_checkpoint == run.store.calls[0][1]


def test_save_failure_preserves_prior_ref_without_memory_rescue(tmp_path):
    class BrokenStore(d.MemoryStore):
        def save(self, snapshot, context):
            if snapshot.identity.commit_seq == 2:
                raise OSError("disk full")
            return super().save(snapshot, context)
    run = Run(tmp_path, config(5, save=2), store=BrokenStore())
    with pytest.raises(c.MahjongError) as error:
        run.execute()
    assert error.value.code == "CHECKPOINT_WRITE_FAILED" and error.value.exit_code == 1
    assert error.value.details["last_checkpoint"]["global_step"] == 2
    assert len(run.saved()) == 1


def test_half_update_failure_does_not_commit_or_publish(tmp_path):
    run = Run(tmp_path, config(5))
    def broken_step():
        with torch.no_grad():
            run.model.weight.add_(1)
        raise RuntimeError("half updated")
    run.runtime.step = broken_step
    with pytest.raises(c.MahjongError) as error:
        run.execute()
    assert error.value.exit_code == 1 and error.value.details["safe_to_checkpoint"] is False
    assert run.source.index == 0 and run.source.has_open_lease()
    assert run.trainer.progress.global_step == 0 and run.saved() == []


@pytest.mark.parametrize("kind,expected", [("empty", "EMPTY_TRAIN_DATA"), ("read", "DATA_READ_FAILED"),
                                          ("nan", "NONFINITE_LOSS"), ("inf", "NONFINITE_LOSS"),
                                          ("grad", "NONFINITE_GRADIENT"), ("no_grad", "CONTRACT_MISMATCH")])
def test_invalid_data_loss_gradient_failure_codes(tmp_path, kind, expected):
    run = Run(tmp_path, config(2, at_start=False))
    if kind in {"empty", "read"}:
        def fail_read():
            raise StopIteration() if kind == "empty" else OSError("read failed")
        run.source.acquire = fail_read
    elif kind in {"nan", "inf", "no_grad"}:
        def forward(batch, *, phase):
            loss = torch.tensor(float(kind) if kind != "no_grad" else 1., requires_grad=True)
            return c.StepResult(loss, {"loss": c.MetricStat(1., batch.sample_count, "squared_target")})
        run.model.forward = forward
    else:
        run.model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(c.MahjongError) as error:
        run.execute()
    assert error.value.code == expected and error.value.exit_code == 1
    assert run.runtime.updates == 0 and run.saved() == []


def test_periodic_evaluation_does_not_change_training_parameters_or_rng(tmp_path):
    with_eval = Run(tmp_path / "yes", config(8, val=2, save=None))
    with_eval.execute()
    without_periodic = Run(tmp_path / "no", config(8, val=None, save=None))
    without_periodic.execute()
    assert_equal(with_eval.runtime.export_state(), without_periodic.runtime.export_state())
    assert_equal(with_eval.saved()[-1].rng, without_periodic.saved()[-1].rng)


def test_snapshot_is_independent_of_later_progress_state(tmp_path):
    run = Run(tmp_path, config(5, save=2))
    run.execute()
    first = run.saved()[0]
    assert first.progress.global_step == 2 and first.progress.next_checkpoint_step == 4
    assert run.trainer.progress.global_step == 5 and run.trainer.progress.next_checkpoint_step == 6
    assert first.data["index"] == 2
    assert first.observation.cutoff.attempt_id == "A1"
    assert [snap.identity.commit_seq for snap in run.saved()] == [1, 2, 3]


@pytest.mark.parametrize("enabled", [True, False])
def test_final_report_status_must_match_validation_enabled(tmp_path, enabled):
    class WrongStatus(Evaluator):
        def run(self, runtime, source, request, observer, stop):
            return replace(super().run(runtime, source, request, observer, stop), status="disabled")
        def disabled(self, source, request, observer):
            return replace(super().disabled(source, request, observer), status="ok")
    run = Run(tmp_path, config(0, enabled=enabled), evaluator=WrongStatus())
    with pytest.raises(c.MahjongError) as error:
        run.execute()
    assert error.value.code == "EVALUATION_FAILED" and error.value.exit_code == 1
    assert run.saved() == []


def test_signal_after_final_report_before_save_uses_stop_path(tmp_path):
    class StopAfterReport(Evaluator):
        def run(self, runtime, source, request, observer, stop):
            report = super().run(runtime, source, request, observer, stop)
            if request.purpose == "final":
                stop.request(signal.SIGINT)
            return report
    run = Run(tmp_path, config(1), evaluator=StopAfterReport())
    assert run.execute().exit_code == 130
    assert [snap.save_reason for snap in run.saved()] == ["stop"]


def test_skip_records_last_successful_learning_rate(tmp_path):
    run = Run(tmp_path, config(2), model=SkipModel([False, True, False]))
    def after_commit(index):
        if index == 1:
            run.runtime.raw_optimizer.param_groups[0]["lr"] = .009
    run.source.on_commit = after_commit
    run.execute()
    rates = [call[1][3] for call in run.observer.calls if call[0] == "record_step"]
    assert rates == [(.003,), (.003,), (.009,)]


def test_timed_checkpoints_use_safe_boundaries_and_do_not_count_saving_time(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('mahjong.training.loop.time', SimpleNamespace(monotonic=lambda: clock[0]))
    cfg = config(10, val=None, save=None, at_start=False)
    cfg['checkpoint']['every_seconds'] = 1800.0
    source = Source(on_commit=lambda index: clock.__setitem__(0, clock[0] + 600))
    run = Run(tmp_path, cfg, source=source)
    save = run.store.save
    def slow_save(snapshot, context):
        assert not source.has_open_lease()
        clock[0] += 4000  # A slow save must not cause immediate catch-up saves.
        return save(snapshot, context)
    run.store.save = slow_save
    assert run.execute().global_step == 10
    assert [(snap.save_reason, snap.progress.global_step) for snap in run.saved()] == [
        ('periodic', 3), ('periodic', 6), ('periodic', 9), ('final', 10)]


def test_timed_checkpoint_at_target_is_only_final_and_stop_always_saves(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('mahjong.training.loop.time', SimpleNamespace(monotonic=lambda: clock[0]))
    cfg = config(6, val=None, save=None, at_start=False)
    cfg['checkpoint']['every_seconds'] = 1800.0
    source = Source(on_commit=lambda index: clock.__setitem__(0, clock[0] + 600))
    run = Run(tmp_path, cfg, source=source)
    run.execute()
    assert [(snap.save_reason, snap.progress.global_step) for snap in run.saved()] == [
        ('periodic', 3), ('final', 6)]
    stopped = Run(tmp_path / 'stop', cfg)
    stopped.source.on_commit = lambda index: stopped.stop.request(signal.SIGINT) if index == 2 else None
    assert stopped.execute().exit_code == 130
    assert [(snap.save_reason, snap.progress.global_step) for snap in stopped.saved()] == [('stop', 2)]


def test_resume_timed_checkpoint_restarts_clock_and_keeps_full_state(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('mahjong.training.loop.time', SimpleNamespace(monotonic=lambda: clock[0]))
    cfg = config(4, val=None, save=None, at_start=False)
    cfg['checkpoint']['every_seconds'] = 1800.0
    first = Run(tmp_path, cfg)
    first.source.on_commit = lambda index: clock.__setitem__(0, clock[0] + 600)
    completion = first.execute()
    saved = first.saved()[-1]
    clock[0] += 100000  # Time while stopped must not cause a save on resume.
    cfg['training']['max_updates'] = 8
    resumed = Run(tmp_path, cfg, resume=c.ResumePlan(completion.last_checkpoint, saved, cfg, ()))
    resumed.source.on_commit = lambda index: clock.__setitem__(0, clock[0] + 600)
    assert resumed.execute().global_step == 8
    assert [(snap.save_reason, snap.progress.global_step) for snap in resumed.saved()] == [
        ('periodic', 7), ('final', 8)]
    continuous = Run(tmp_path / 'continuous', cfg)
    continuous.execute()
    assert_equal(continuous.runtime.export_state(), resumed.runtime.export_state())
    assert_equal(continuous.saved()[-1].rng, resumed.saved()[-1].rng)
    assert_equal(continuous.saved()[-1].data, resumed.saved()[-1].data)


def test_resume_can_disable_step_schedule_and_enable_timer(tmp_path):
    cfg = config(3, val=None, save=2)
    first = Run(tmp_path, cfg)
    completed = first.execute()
    saved = first.saved()[-1]
    cfg['training']['max_updates'] = 6
    cfg['checkpoint'] = {'every_updates': None, 'every_seconds': 1800.0}
    changes = (c.ConfigChange('checkpoint.every_updates', 2, None, 'explicit allowed override'),)
    resumed = Run(tmp_path, cfg, resume=c.ResumePlan(completed.last_checkpoint, saved, cfg, changes))
    assert resumed.execute().global_step == 6
    assert [snap.save_reason for snap in resumed.saved()] == ['final']
    assert resumed.saved()[-1].progress.next_checkpoint_step is None


@pytest.mark.parametrize('remaining',[0,999,1000])
def test_execution_policy_receives_remaining_updates_on_resume(tmp_path,remaining):
    original=Run(tmp_path,config(1))
    completion=original.execute()
    snapshot=original.saved()[-1]
    settings=config(1+remaining)
    resumed=Run(tmp_path,settings,resume=c.ResumePlan(completion.last_checkpoint,snapshot,settings,()))
    observed=[]
    resumed.runtime.configure_training_execution=observed.append
    resumed.stop.request(signal.SIGTERM)
    resumed.execute()
    assert observed==[remaining]


@pytest.mark.parametrize('interval,periodic',[(5,[5]),(None,[])])
def test_resume_validation_cadence_rebases_without_changing_training_state(tmp_path,interval,periodic):
    original=Run(tmp_path,config(3,val=2,save=None))
    completed=original.execute();snapshot=original.saved()[-1]
    settings=config(8,val=interval,save=None)
    change=c.ConfigChange('validation.every_updates',2,interval,'explicit allowed override')
    resumed=Run(tmp_path,settings,resume=c.ResumePlan(completed.last_checkpoint,snapshot,settings,(change,)))
    resumed.execute()
    assert resumed.evaluator.calls==[('periodic',step) for step in periodic]+[('final',8)]
    continuous=Run(tmp_path/'continuous',config(8,val=2,save=None))
    continuous.execute()
    assert_equal(resumed.runtime.export_state(),continuous.runtime.export_state())
    assert_equal(resumed.saved()[-1].rng,continuous.saved()[-1].rng)
    assert_equal(resumed.saved()[-1].data,continuous.saved()[-1].data)
    assert snapshot.progress.next_validation_step==4


def test_changed_cadence_retains_pending_validation(tmp_path):
    original=Run(tmp_path,config(8,val=2,save=None),evaluator=Evaluator(cancel=('periodic',2)))
    completed=original.execute();snapshot=original.saved()[-1]
    assert snapshot.progress.pending_validation.purpose=='periodic'
    settings=config(8,val=5,save=None)
    change=c.ConfigChange('validation.every_updates',2,5,'explicit allowed override')
    resumed=Run(tmp_path,settings,resume=c.ResumePlan(completed.last_checkpoint,snapshot,settings,(change,)))
    resumed.execute()
    assert resumed.evaluator.calls==[('periodic',2),('periodic',5),('final',8)]
