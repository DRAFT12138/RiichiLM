"""Contract-only examples for train, resume, and independent evaluation.

These exercise ownership and call order. They are not production Trainer,
aggregation, observer, storage, or recovery implementations.
"""

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
import random

from mahjong import contracts as c

torch = c.torch
UTC = '2026-10-11T00:00:00Z'
SCHEMA = 'contract-example/v1'


def copy_contract(value):
    return c.decode_contract(type(value), c.encode_contract(value))


class MutableStopToken:
    def __init__(self):
        self.number = None

    def request(self, signal_number: int) -> None:
        self.number = signal_number

    def requested(self) -> bool:
        return self.number is not None

    def signal_number(self) -> int | None:
        return self.number


class KnownMetricAccumulator:
    """Tiny in-memory arithmetic fixture; P06 owns the production algorithm."""
    def __init__(self):
        self.sums = {}

    def add(self, stats: c.Mapping[str, c.MetricStat]) -> None:
        for name, stat in stats.items():
            prior = self.sums.get(name, c.MetricStat(0., 0, stat.unit))
            self.sums[name] = c.MetricStat(prior.numerator + stat.numerator,
                                           prior.denominator + stat.denominator, stat.unit)

    def result(self) -> c.Mapping[str, c.AggregatedMetric]:
        return {name: c.AggregatedMetric(stat.numerator, stat.denominator,
                    stat.numerator / stat.denominator if stat.denominator else None,
                    stat.unit, None if stat.denominator else 'no_valid_statistics')
                for name, stat in self.sums.items()}

    def state_dict(self) -> c.State:
        return c.clone_state({'schema_version': 1, 'metrics':
                              {name: c.encode_contract(stat) for name, stat in self.sums.items()}})

    def load_state_dict(self, state: c.State) -> None:
        self.sums = {name: c.decode_contract(c.MetricStat, stat) for name, stat in state['metrics'].items()}

    def reset(self) -> None:
        self.sums.clear()


class KnownSource:
    """Infinite training leases with an explicit committed cursor."""
    def __init__(self):
        self.index = 0
        self.lease = None
        self.owned = {'tensor': torch.tensor([1.]), 'list': [1, {7: torch.tensor([2.])}]}

    def descriptor(self) -> c.ComponentDescriptor:
        return c.ComponentDescriptor('data', 'contract-source/v1', 1, SCHEMA,
                                     'source-config', 'known-data', None,
                                     'committed_batch', ('cpu',))

    def acquire(self) -> c.BatchLease:
        if self.lease is not None:
            raise c.MahjongError('CONTRACT_MISMATCH', 'A lease is already open', 'training')
        batch = c.Batch({'x': torch.tensor([1., 2.])},
                        (f'fixture/{2*self.index}', f'fixture/{2*self.index+1}'), 2, SCHEMA)
        self.lease = c.BatchLease(batch, str(self.index))
        return self.lease

    def commit(self, token: str) -> None:
        if self.lease is None or token != self.lease.token:
            raise c.MahjongError('CONTRACT_MISMATCH', 'Wrong or repeated lease token', 'training')
        self.index += 1
        self.lease = None

    def has_open_lease(self) -> bool:
        return self.lease is not None

    def state_dict(self) -> c.State:
        return c.clone_state({'version': 1, 'index': self.index, 'owned': self.owned})

    def load_state_dict(self, state: c.State) -> None:
        state = c.clone_state(state)
        self.index, self.owned = state['index'], state['owned']
        self.lease = None


class KnownEvalSource:
    def descriptor(self) -> c.ComponentDescriptor:
        return KnownSource().descriptor()

    def iter_batches(self):
        yield c.Batch({'x': torch.tensor([1., 2.])}, ('eval/0', 'eval/1'), 2, SCHEMA)


class KnownModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.private = torch.Generator().manual_seed(9)
        self.auxiliary = {'nested': [1, {11: torch.tensor([3.])}]}

    def descriptor(self) -> c.ComponentDescriptor:
        return c.ComponentDescriptor('model', 'contract-model/v1', 1, SCHEMA,
                                     'model-config', None, 1, 'state_dict', ('cpu',))

    def prepare_batch(self, batch: c.Batch, device: torch.device) -> c.Batch:
        return c.Batch({'x': batch.payload['x'].to(device)}, batch.sample_ids,
                       batch.sample_count, batch.input_schema)

    def forward(self, batch: c.Batch, *, phase: str) -> c.StepResult:
        torch.rand((), generator=self.private)
        squared = (self.weight * batch.payload['x']).square()
        loss = squared.mean()
        return c.StepResult(loss, {'loss': c.MetricStat(float(squared.detach().sum()),
                                                      batch.sample_count, 'squared_target')})

    def auxiliary_state_dict(self) -> c.State:
        return c.clone_state({'rng': self.private.get_state(), 'owned': self.auxiliary})

    def load_auxiliary_state_dict(self, state: c.State) -> None:
        state = c.clone_state(state)
        self.private.set_state(state['rng'])
        self.auxiliary = state['owned']

    def reset_auxiliary_rng(self, seed: int) -> None:
        self.private.manual_seed(seed)


class KnownRuntime:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.calls = []

    def descriptor(self) -> c.ComponentDescriptor:
        return self.model.descriptor()

    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> None:
        self.model, self.optimizer = model, optimizer
        self.calls.append('setup')

    def compute(self, batch: c.Batch, *, phase: str) -> c.StepResult:
        self.calls.append(f'compute:{phase}')
        result = self.model(self.model.prepare_batch(batch, torch.device('cpu')), phase=phase)
        c.validate_step_result(result, phase=phase)
        return result

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: c.Tensor) -> None:
        loss.backward()

    def check_gradients(self) -> None:
        if not any(p.grad is not None for p in self.model.parameters()):
            raise c.MahjongError('CONTRACT_MISMATCH', 'No gradients', 'training')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise c.MahjongError('NONFINITE_GRADIENT', 'Nonfinite gradients', 'training')

    def step(self) -> None:
        self.optimizer.step()

    def learning_rates(self) -> tuple[float, ...]:
        return () if self.optimizer is None else tuple(group['lr'] for group in self.optimizer.param_groups)

    def export_state(self) -> c.RuntimeState:
        return c.RuntimeState(c.clone_state(self.model.state_dict()), self.model.auxiliary_state_dict(),
                              None if self.optimizer is None else c.clone_state(self.optimizer.state_dict()), None, None)

    def import_state(self, state: c.RuntimeState, *, model_only: bool = False) -> None:
        self.calls.append(f'import_state:{model_only}')
        self.model.load_state_dict(state.model)
        self.model.load_auxiliary_state_dict(state.auxiliary)
        if not model_only and self.optimizer is not None:
            self.optimizer.load_state_dict(c.clone_state(state.optimizer))

    def reset_auxiliary_rng(self, seed: int) -> None:
        self.model.reset_auxiliary_rng(seed)

    def capture_rng(self) -> c.RngState:
        return c.RngState(c.clone_state(random.getstate()), None, torch.get_rng_state().clone(), None, {})

    def restore_rng(self, state: c.RngState) -> None:
        self.calls.append('restore_rng')
        random.setstate(state.python)
        torch.set_rng_state(state.torch_cpu)

    @contextmanager
    def evaluation_scope(self, seed: int):
        self.calls.append('evaluation_scope:enter')
        rng = self.capture_rng()
        auxiliary = self.model.auxiliary_state_dict()
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            self.reset_auxiliary_rng(seed)
            self.model.eval()
            with torch.inference_mode():
                yield
        finally:
            self.model.load_auxiliary_state_dict(auxiliary)
            for module, mode in modes:
                module.training = mode
            self.restore_rng(rng)
            self.calls.append('evaluation_scope:exit')

    def resource_snapshot(self) -> c.ResourceSnapshot:
        return c.ResourceSnapshot('cpu', UTC, {
            'allocated': c.ResourceValue(None, 'bytes', 'unavailable', 'cpu_execution', UTC, 0.)})


class RecordingObserver:
    def __init__(self, identity: c.RecordIdentity | None = None):
        self.identity = identity or c.RecordIdentity('training', 'R', 'A1', None)
        self.calls = []
        self.event_seq = self.metric_seq = 0
        self.window = c.WindowState(1, {'schema_version': 1, 'metrics': {}}, 0, 10, (0.1,))
        self.closed = False

    def event(self, type: str, payload: dict, progress: c.TrainProgress | None) -> int:
        self.event_seq += 1
        self.calls.append(('event', (type, c.clone_state(payload), None if progress is None else copy_contract(progress))))
        return self.event_seq

    def phase(self, name: c.Phase, progress: c.TrainProgress | None) -> None:
        self.calls.append(('phase', name))

    def record_step(self, result: c.StepResult, progress: c.TrainProgress,
                    timing: c.Timing, lr: tuple[float, ...]) -> None:
        self.metric_seq += 1
        self.calls.append(('record_step', (copy_contract(result), copy_contract(progress), timing, lr)))

    def record_evaluation(self, report: c.EvaluationReport) -> None:
        self.calls.append(('record_evaluation', copy_contract(report)))

    def publish_resources(self, snapshot: c.ResourceSnapshot) -> None:
        self.calls.append(('publish_resources', copy_contract(snapshot)))

    def write_partial(self, progress: c.TrainProgress) -> None:
        self.calls.append(('write_partial', copy_contract(progress)))

    def flush(self, durable: bool = False) -> c.LogCutoff | None:
        self.calls.append(('flush', durable))
        return None if self.identity.kind == 'evaluation' else c.LogCutoff(self.identity.attempt_id, self.event_seq, self.metric_seq)

    def state_dict(self) -> c.WindowState:
        if self.identity.kind != 'training':
            raise c.MahjongError('CONTRACT_MISMATCH', 'Independent evaluation has no training window')
        return copy_contract(self.window)

    def load_state_dict(self, state: c.WindowState, source: c.CheckpointRef) -> None:
        self.window = copy_contract(state)
        self.calls.append(('load_state_dict', source))

    def close(self, completion: c.Completion) -> c.Summary:
        if self.closed:
            raise c.MahjongError('CONTRACT_MISMATCH', 'Observer.close must be called once')
        self.closed = True
        self.calls.append(('close', completion))
        return c.Summary(1, self.identity, completion, UTC, {})


class KnownEvaluator:
    def run(self, runtime: c.Runtime, source: c.EvalSource, request: c.EvaluationRequest,
            observer: c.Observer, stop: c.StopToken) -> c.EvaluationReport:
        observer.phase('evaluating', None)
        sums, batches, samples = {}, 0, 0
        with runtime.evaluation_scope(request.seed):
            for batch in source.iter_batches():
                if stop.requested():
                    break
                result = runtime.compute(batch, phase='eval')
                batches += 1
                samples += batch.sample_count
                for name, stat in result.metrics.items():
                    old = sums.get(name, c.MetricStat(0., 0, stat.unit))
                    sums[name] = c.MetricStat(old.numerator + stat.numerator,
                                             old.denominator + stat.denominator, stat.unit)
                observer.publish_resources(runtime.resource_snapshot())
        status = 'cancelled' if stop.requested() else ('ok' if any(s.denominator for s in sums.values()) else 'not_evaluable')
        reason = {'ok': None, 'cancelled': 'stop_requested', 'not_evaluable': 'empty_source' if not batches else 'no_valid_statistics'}[status]
        metrics = {name: c.AggregatedMetric(stat.numerator, stat.denominator,
                    stat.numerator/stat.denominator if stat.denominator else None, stat.unit,
                    None if stat.denominator else 'no_valid_statistics') for name, stat in sums.items()}
        report = make_report(source, request, status, batches, samples, metrics, reason)
        observer.record_evaluation(report)
        return report

    def disabled(self, source: c.EvalSource, request: c.EvaluationRequest,
                 observer: c.Observer) -> c.EvaluationReport:
        report = make_report(source, request, 'disabled', 0, 0, {}, 'validation_disabled')
        observer.record_evaluation(report)
        return report


def make_report(source, request, status='ok', batches=1, samples=2, metrics=None, reason=None):
    return c.EvaluationReport(1, request.evaluation_id, status, request.purpose,
        request.model_source, source.descriptor(), request.seed, request.batch_size,
        request.precision, batches, samples,
        {'loss': c.AggregatedMetric(5., 2, 2.5, 'squared_target', None)} if metrics is None else metrics,
        reason, UTC, UTC, 0., request.config_hash, request.environment_ref, request.report_path.as_posix())


class MemoryStore:
    def __init__(self):
        self.items = {}
        self.calls = []

    def save(self, snapshot: c.Snapshot, context: c.RunContext) -> c.CheckpointRef:
        identity = snapshot.identity
        ref = c.CheckpointRef(identity.checkpoint_id, identity.run_id, identity.attempt_id,
                              identity.global_step, f'checkpoints/{identity.checkpoint_id}', 'fixture-hash', identity.commit_seq)
        self.items[ref.checkpoint_id] = (ref, copy_contract(snapshot))
        self.calls.append(('save', ref))
        return ref

    def load(self, ref: c.CheckpointRef, *, run_dir: Path) -> c.Snapshot:
        self.calls.append(('load', ref))
        return copy_contract(self.items[ref.checkpoint_id][1])

    def resolve(self, path: Path) -> c.CheckpointRef:
        for ref, _ in self.items.values():
            if path == Path(ref.path) or path.name == ref.checkpoint_id:
                return ref
        raise c.MahjongError('CHECKPOINT_INVALID', 'Unknown fixture checkpoint')

    def find_latest(self, run_dir: Path, attempt_id: str) -> c.CheckpointRef | None:
        branch = [ref for ref, _ in self.items.values() if ref.attempt_id == attempt_id]
        return max(branch, key=lambda ref: ref.commit_seq) if branch else None


class RecordingRunner:
    def run(self, *, context, metadata, config, source, validation_source, runtime,
            evaluator, observer, store, stop, resume=None):
        progress = c.TrainProgress()
        if resume is not None:
            runtime.import_state(resume.snapshot.runtime)
            source.load_state_dict(resume.snapshot.data)
            observer.load_state_dict(resume.snapshot.observation.window, resume.source)
            progress = copy_contract(resume.snapshot.progress)
            runtime.restore_rng(resume.snapshot.rng)
        target = config['training']['max_updates']
        while progress.global_step < target and not stop.requested():
            observer.phase('training', progress)
            lease = source.acquire()
            progress.batches_read += 1
            progress.samples_read += lease.batch.sample_count
            runtime.zero_grad()
            result = runtime.compute(lease.batch, phase='train')
            progress.samples_processed += lease.batch.sample_count
            rates = runtime.learning_rates()
            runtime.backward(result.loss)
            runtime.check_gradients()
            runtime.step()
            progress.global_step += 1
            progress.samples_updated += lease.batch.sample_count
            source.commit(lease.token)
            runtime.zero_grad()
            observer.record_step(result, progress, c.Timing(0.01), rates)
            observer.publish_resources(runtime.resource_snapshot())
        report = evaluator.run(runtime, validation_source, make_request(context, progress.global_step), observer, stop)
        progress.last_evaluation_ref = report.report_ref
        observer.write_partial(progress)
        observer.phase('checkpointing', progress)
        cutoff = observer.flush(durable=True)
        snapshot = make_snapshot(runtime, source, observer, progress=progress,
                                 context=context, metadata=metadata, config=config, cutoff=cutoff)
        ref = store.save(snapshot, context)
        return c.Completion('interrupted' if stop.requested() else 'completed',
                            128+stop.signal_number() if stop.requested() else 0,
                            progress.global_step, ref, report.report_ref, None)


def make_context(attempt='A1', mode='new', source=None):
    return c.RunContext('R', attempt, Path('.'), Path(f'attempts/{attempt}'), mode, source)


def make_metadata(runtime=None, source=None):
    return c.RunMetadata('fixture-code', 'fixture-dependencies', {'device': 'cpu'},
                         (KnownSource().descriptor(), KnownModel().descriptor()),
                         'fixture-config', {'train_seed': 1})


def make_request(context=None, step=0, standalone=False, source=None):
    context = context or make_context()
    model_source = c.ModelSource(context.run_id, context.attempt_id, step, None)
    if source is not None:
        model_source = c.ModelSource(source.run_id, source.attempt_id, source.global_step, source.checkpoint_id)
    return c.EvaluationRequest('E1', 'standalone' if standalone else 'final', model_source,
                              43, 2, '32-true', Path('evaluations/E1/report.json' if standalone else f'reports/{context.attempt_id}/E1.json'),
                              'fixture-config', 'environment.json')


def make_snapshot(runtime=None, source=None, observer=None, *, progress=None, context=None,
                  metadata=None, config=None, cutoff=None):
    if runtime is None:
        model = KnownModel()
        runtime = KnownRuntime()
        runtime.setup(model, torch.optim.Adam(model.parameters(), lr=0.1))
    source, observer = source or KnownSource(), observer or RecordingObserver()
    progress, context = progress or c.TrainProgress(), context or make_context()
    metadata = metadata or make_metadata(runtime, source)
    config = config or {'training': {'max_updates': max(2, progress.global_step)}}
    cutoff = cutoff or observer.flush(durable=True)
    return c.Snapshot(1, 1,
        c.SnapshotIdentity(context.run_id, context.attempt_id, f'C{context.attempt_id}', progress.global_step,
                           1, UTC, context.source_checkpoint, metadata.code_fingerprint,
                           metadata.dependency_fingerprint, metadata.environment),
        c.clone_state(config), metadata.frozen_config_hash, metadata.descriptors,
        runtime.export_state(), copy_contract(progress), source.state_dict(), runtime.capture_rng(),
        c.ObservationState(observer.state_dict(), cutoff, ()), 'final', config['training']['max_updates'])


def drive_training(runner: c.Runner, context: c.RunContext, metadata: c.RunMetadata,
                   config: c.ResolvedConfig, source: c.TrainSource, validation: c.EvalSource,
                   runtime: c.Runtime, evaluator: c.Evaluator, observer: c.Observer,
                   store: c.CheckpointStore, stop: c.StopToken, plan: c.ResumePlan | None = None) -> c.Summary:
    result = runner.run(context=context, metadata=metadata, config=config, source=source,
                        validation_source=validation, runtime=runtime, evaluator=evaluator,
                        observer=observer, store=store, stop=stop, resume=plan)
    return observer.close(result)


def drive_evaluation(runtime: c.Runtime, snapshot: c.Snapshot, evaluator: c.Evaluator,
                     source: c.EvalSource, request: c.EvaluationRequest,
                     observer: c.Observer, stop: c.StopToken) -> c.EvaluationReport:
    runtime.import_state(snapshot.runtime, model_only=True)
    return evaluator.run(runtime, source, request, observer, stop)


def value_object_fixtures():
    """One legal object for every documented dataclass, including nullable fields."""
    snapshot = make_snapshot()
    ref = c.CheckpointRef('CA1', 'R', 'A1', 0, 'checkpoints/CA1', 'fixture-hash', 1)
    report = make_report(KnownEvalSource(), make_request())
    values = [KnownSource().descriptor(), c.Batch({'x': torch.ones(1)}, ('sample',), 1, SCHEMA),
        c.BatchLease(c.Batch({}, ('sample',), 1, SCHEMA), 'lease'), c.MetricStat(0., 0, 'count'),
        c.AggregatedMetric(0., 0, None, 'count', 'no_valid_statistics'),
        c.StepResult(None, {}, 'no_targets'), ref, make_metadata(), make_context(),
        c.EvaluationContext('E1', Path('evaluations/E1'), Path('.'), ref),
        c.RecordIdentity('evaluation', 'R', None, 'E1'), c.PendingValidation('periodic', 2),
        c.TrainProgress(), c.ModelSource('R', 'A1', 0, None), make_request(), report,
        c.LogCutoff('A1', 0, 0), c.Timing(None), c.ResourceValue(None, 'bytes', 'unavailable', 'cpu_execution', UTC, 0.),
        c.ResourceSnapshot('cpu', UTC, {}),
        c.Event(1, c.RecordIdentity('training', 'R', 'A1', None), 1, UTC, 'training', 0, 'attempt_started', {}),
        c.MetricRecord(1, c.RecordIdentity('training', 'R', 'A1', None), 1, UTC, 'train', 1, 'window', 0, 1,
                       None, None, {'loss': c.AggregatedMetric(1., 1, 1., 'count', None)}, (0.1,), {}, None, None),
        c.Completion('completed', 0, 0, ref, report.report_ref, None),
        c.Summary(1, c.RecordIdentity('training', 'R', 'A1', None), c.Completion('failed', 1, 0, None, None, 'example'), UTC, {}),
        snapshot.runtime, snapshot.rng, snapshot.observation.window, snapshot.observation,
        snapshot.identity, snapshot, c.ConfigChange('training.max_updates', 2, 3, 'allowed'),
        c.ResumePlan(ref, snapshot, {'training': {'max_updates': 3}}, ())]
    return {type(value).__name__: value for value in values}
