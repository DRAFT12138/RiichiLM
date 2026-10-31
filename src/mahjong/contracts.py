"""Phase 1 cross-module value objects and protocols (contract/schema version 1).

This module depends only on Python and PyTorch. Runtime implementations remain
in their owning modules; checkpoint codecs store ordinary dictionaries only.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Mapping, Protocol, runtime_checkable
import torch
from torch import Tensor

type StateValue = Tensor | str | bool | int | float | None | list[StateValue] | tuple[StateValue, ...] | dict[str | int, StateValue]
type State = dict[str | int, StateValue]
type JsonValue = str | bool | int | float | None | list[JsonValue] | dict[str, JsonValue]
type ResolvedConfig = dict[str, JsonValue]
type Phase = Literal['initializing', 'preparing_data', 'restoring', 'training', 'evaluating', 'checkpointing', 'finalizing', 'inference']
type Outcome = Literal['completed', 'interrupted', 'failed', 'not_evaluable']
type Purpose = Literal['initial', 'periodic', 'final', 'standalone']



CONTRACT_VERSION = 1
FORMAT_VERSION = 1
SCHEMA_VERSION = 1

ErrorCode = Literal[
    'CONFIG_INVALID', 'DEVICE_UNAVAILABLE', 'CONTRACT_MISMATCH',
    'EMPTY_TRAIN_DATA', 'DATA_READ_FAILED', 'NONFINITE_LOSS',
    'NONFINITE_GRADIENT', 'NO_PROGRESS', 'CHECKPOINT_INVALID',
    'RESUME_INCOMPATIBLE', 'CHECKPOINT_WRITE_FAILED', 'LOCAL_RECORD_FAILED',
    'EVALUATION_FAILED',
    'LEGAL_SET_EMPTY', 'LEGAL_SET_INCOMPLETE', 'ACTION_UNREPRESENTABLE',
    'CONTEXT_TOO_LONG', 'CANCELLED', 'DEADLINE_EXCEEDED', 'STALE_DECISION',
    'NONFINITE_LOGITS', 'INCOMPLETE_GENERATION', 'RESPONSE_BATCH_REQUIRED',
    'RESPONSE_BATCH_INCOMPLETE', 'WINDOW_ALREADY_CLOSED', 'EVENT_STREAM_INCOMPLETE',
    'UNSUPPORTED_PROFILE', 'ENVIRONMENT_STEP_FAILED',
    'MODEL_EXECUTION_FAILED',
    'LLM_HTTP_FAILED', 'LLM_RETRYABLE', 'LLM_RESPONSE_INVALID', 'LLM_DECISION_FAILED',
]

class MahjongError(RuntimeError):
    """A public failure with stable code, phase, and structured diagnostics."""

    def __init__(self, code: ErrorCode, message: str,
                 phase: Phase = 'initializing', details: dict[str, JsonValue] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.phase = phase
        self.details = {} if details is None else details

    @property
    def exit_code(self) -> int:
        startup = self.phase in {'initializing', 'preparing_data', 'restoring'}
        return 2 if startup and self.code in {'CONFIG_INVALID', 'DEVICE_UNAVAILABLE',
                                             'CONTRACT_MISMATCH', 'RESUME_INCOMPATIBLE'} else 1

    def __str__(self) -> str:
        return f'{self.code}: {self.message}'

class _Contract:
    def __post_init__(self) -> None:
        validate_contract(self)

@dataclass(frozen=True)
class ComponentDescriptor(_Contract):
    kind: Literal['data', 'model']
    implementation_id: str
    state_version: int
    input_schema: str
    config_fingerprint: str
    data_fingerprint: str | None
    metric_schema_version: int | None
    recovery_capability: Literal['committed_batch', 'state_dict']
    supported_execution: tuple[str, ...]

@dataclass(frozen=True)
class Batch(_Contract):
    payload: object
    sample_ids: tuple[str, ...]
    sample_count: int
    input_schema: str

@dataclass(frozen=True)
class BatchLease(_Contract):
    batch: Batch
    token: str

@runtime_checkable
class TrainSource(Protocol):
    def descriptor(self) -> ComponentDescriptor: ...
    def acquire(self) -> BatchLease: ...
    def commit(self, token: str) -> None: ...
    def has_open_lease(self) -> bool: ...
    def state_dict(self) -> State: ...
    def load_state_dict(self, state: State) -> None: ...

@runtime_checkable
class EvalSource(Protocol):
    def descriptor(self) -> ComponentDescriptor: ...
    def iter_batches(self) -> Iterator[Batch]: ...

@dataclass(frozen=True)
class MetricStat(_Contract):
    numerator: float
    denominator: int
    unit: str

@dataclass(frozen=True)
class AggregatedMetric(_Contract):
    numerator: float
    denominator: int
    value: float | None
    unit: str
    reason: str | None

@dataclass(frozen=True)
class StepResult(_Contract):
    loss: Tensor | None
    metrics: Mapping[str, MetricStat]
    skip_reason: str | None = None

@runtime_checkable
class Model(Protocol):  # 实现必须同时是 torch.nn.Module
    def descriptor(self) -> ComponentDescriptor: ...
    def prepare_batch(self, batch: Batch, device: torch.device) -> Batch: ...
    def forward(self, batch: Batch, *, phase: Literal['train', 'eval']) -> StepResult: ...
    def auxiliary_state_dict(self) -> State: ...
    def load_auxiliary_state_dict(self, state: State) -> None: ...
    def reset_auxiliary_rng(self, seed: int) -> None: ...

@runtime_checkable
class MetricAccumulator(Protocol):
    def add(self, stats: Mapping[str, MetricStat]) -> None: ...
    def result(self) -> Mapping[str, AggregatedMetric]: ...
    def state_dict(self) -> State: ...
    def load_state_dict(self, state: State) -> None: ...
    def reset(self) -> None: ...


@dataclass(frozen=True)
class CheckpointRef(_Contract):
    checkpoint_id: str
    run_id: str
    attempt_id: str
    global_step: int
    path: str
    manifest_hash: str
    commit_seq: int

@dataclass(frozen=True)
class RunMetadata(_Contract):
    code_fingerprint: str
    dependency_fingerprint: str
    environment: dict[str, JsonValue]
    descriptors: tuple[ComponentDescriptor, ...]
    frozen_config_hash: str
    derived_seeds: dict[str, int]

@dataclass(frozen=True)
class RunContext(_Contract):
    run_id: str
    attempt_id: str
    run_dir: Path
    attempt_dir: Path
    mode: Literal['new', 'resume', 'weights']
    source_checkpoint: CheckpointRef | None

@dataclass(frozen=True)
class EvaluationContext(_Contract):
    evaluation_id: str
    evaluation_dir: Path
    run_dir: Path
    source_checkpoint: CheckpointRef

@dataclass(frozen=True)
class RecordIdentity(_Contract):
    kind: Literal['training', 'evaluation']
    run_id: str
    attempt_id: str | None
    evaluation_id: str | None

@dataclass(frozen=True)
class PendingValidation(_Contract):
    purpose: Literal['initial', 'periodic', 'final']
    global_step: int

@dataclass
class TrainProgress(_Contract):
    global_step: int = 0
    batches_read: int = 0
    samples_read: int = 0
    samples_processed: int = 0
    samples_updated: int = 0
    skipped_batches: int = 0
    consecutive_skips: int = 0
    last_validated_step: int | None = None
    next_validation_step: int | None = None
    next_checkpoint_step: int | None = None
    pending_validation: PendingValidation | None = None
    initial_validation_done: bool = False
    last_evaluation_ref: str | None = None

@dataclass(frozen=True)
class ModelSource(_Contract):
    run_id: str
    attempt_id: str
    global_step: int
    checkpoint_id: str | None

@dataclass(frozen=True)
class EvaluationRequest(_Contract):
    evaluation_id: str
    purpose: Purpose
    model_source: ModelSource
    seed: int
    batch_size: int
    precision: str
    report_path: Path
    config_hash: str
    environment_ref: str

@dataclass(frozen=True)
class EvaluationReport(_Contract):
    schema_version: int
    evaluation_id: str
    status: Literal['ok', 'not_evaluable', 'cancelled', 'disabled', 'failed']
    purpose: Purpose
    model_source: ModelSource
    data: ComponentDescriptor
    seed: int
    batch_size: int
    precision: str
    batches: int
    samples: int
    metrics: Mapping[str, AggregatedMetric]
    reason: str | None
    started_at: str
    finished_at: str
    duration_seconds: float
    config_hash: str
    environment_ref: str
    report_ref: str

@dataclass(frozen=True)
class LogCutoff(_Contract):
    attempt_id: str
    event_seq: int
    metric_seq: int

@dataclass(frozen=True)
class Timing(_Contract):
    step_seconds: float | None  # skip为None；attempt耗时由Observer自有monotonic计算

@dataclass(frozen=True)
class ResourceValue(_Contract):
    value: float | int | None
    unit: str
    status: Literal['available', 'unavailable', 'stale']
    reason: str | None
    measured_at: str
    age_seconds: float

@dataclass(frozen=True)
class ResourceSnapshot(_Contract):
    device: str
    measured_at: str
    values: Mapping[str, ResourceValue]

@dataclass(frozen=True)
class Event(_Contract):
    schema_version: int
    identity: RecordIdentity
    event_seq: int
    utc: str
    phase: Phase
    global_step: int
    type: str
    payload: dict[str, JsonValue]

@dataclass(frozen=True)
class MetricRecord(_Contract):
    schema_version: int
    identity: RecordIdentity
    metric_seq: int
    utc: str
    phase: Literal['train', 'eval']
    global_step: int
    kind: Literal['window', 'partial', 'evaluation']
    window_start_step: int | None
    window_end_step: int | None
    source_checkpoint: CheckpointRef | None
    inherited_through_step: int | None
    stats: Mapping[str, AggregatedMetric]
    learning_rates: tuple[float, ...]
    performance: dict[str, JsonValue]
    evaluation_id: str | None
    purpose: Purpose | None

@dataclass(frozen=True)
class Completion(_Contract):
    outcome: Outcome
    exit_code: int
    global_step: int
    last_checkpoint: CheckpointRef | None
    report_ref: str | None
    reason: str | None

@dataclass(frozen=True)
class Summary(_Contract):
    schema_version: int
    identity: RecordIdentity
    completion: Completion
    finished_at: str
    artifacts: dict[str, str]  # 相对路径 -> produced/disabled/missing


@dataclass(frozen=True)
class RuntimeState(_Contract):
    model: State
    auxiliary: State
    optimizer: State | None
    scheduler: None
    scaler: None

@dataclass(frozen=True)
class RngState(_Contract):
    python: StateValue
    numpy: StateValue  # 未安装/未启用为 None；其数组编码为基础值/CPU Tensor
    torch_cpu: Tensor
    torch_device: Tensor | None
    runtime_private: State  # 本批为空；组件私有流由各自 state 保存

@dataclass(frozen=True)
class WindowState(_Contract):
    schema_version: int
    accumulator: State
    window_start_step: int
    next_record_step: int
    last_learning_rates: tuple[float, ...]

@dataclass(frozen=True)
class ObservationState(_Contract):
    window: WindowState
    cutoff: LogCutoff
    lineage: tuple[LogCutoff, ...]  # 祖先有效截点，按执行顺序；每个 attempt 一项

@dataclass(frozen=True)
class SnapshotIdentity(_Contract):
    run_id: str
    attempt_id: str
    checkpoint_id: str
    global_step: int
    commit_seq: int
    created_at: str
    parent_checkpoint: CheckpointRef | None
    code_fingerprint: str
    dependency_fingerprint: str
    environment: dict[str, JsonValue]

@dataclass(frozen=True)
class Snapshot(_Contract):
    format_version: int
    contract_version: int
    identity: SnapshotIdentity
    config: ResolvedConfig
    frozen_config_hash: str
    descriptors: tuple[ComponentDescriptor, ...]
    runtime: RuntimeState
    progress: TrainProgress
    data: State
    rng: RngState
    observation: ObservationState
    save_reason: Literal['periodic', 'stop', 'final']
    target_updates: int

@dataclass(frozen=True)
class ConfigChange(_Contract):
    path: str
    old: JsonValue
    new: JsonValue
    reason: str

@dataclass(frozen=True)
class ResumePlan(_Contract):
    source: CheckpointRef
    snapshot: Snapshot
    effective_config: ResolvedConfig
    changes: tuple[ConfigChange, ...]


@runtime_checkable
class StopToken(Protocol):
    def request(self, signal_number: int) -> None: ...
    def requested(self) -> bool: ...
    def signal_number(self) -> int | None: ...

@runtime_checkable
class Runtime(Protocol):
    def descriptor(self) -> ComponentDescriptor: ...
    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> None: ...
    def compute(self, batch: Batch, *, phase: Literal['train', 'eval']) -> StepResult: ...
    def zero_grad(self) -> None: ...
    def backward(self, loss: Tensor) -> None: ...
    def check_gradients(self) -> None: ...
    def step(self) -> None: ...
    def learning_rates(self) -> tuple[float, ...]: ...
    def export_state(self) -> RuntimeState: ...
    def import_state(self, state: RuntimeState, *, model_only: bool = False) -> None: ...
    def reset_auxiliary_rng(self, seed: int) -> None: ...
    def capture_rng(self) -> RngState: ...
    def restore_rng(self, state: RngState) -> None: ...
    def evaluation_scope(self, seed: int) -> AbstractContextManager[None]: ...
    def resource_snapshot(self) -> ResourceSnapshot: ...

@runtime_checkable
class Observer(Protocol):
    def event(self, type: str, payload: dict[str, JsonValue], progress: TrainProgress | None) -> int: ...
    def phase(self, name: Phase, progress: TrainProgress | None) -> None: ...
    def record_step(self, result: StepResult, progress: TrainProgress, timing: Timing,
                    lr: tuple[float, ...]) -> None: ...
    def record_evaluation(self, report: EvaluationReport) -> None: ...
    def publish_resources(self, snapshot: ResourceSnapshot) -> None: ...
    def write_partial(self, progress: TrainProgress) -> None: ...
    def flush(self, durable: bool = False) -> LogCutoff | None: ...
    def state_dict(self) -> WindowState: ...
    def load_state_dict(self, state: WindowState, source: CheckpointRef) -> None: ...
    def close(self, completion: Completion) -> Summary: ...

@runtime_checkable
class Evaluator(Protocol):
    def run(self, runtime: Runtime, source: EvalSource, request: EvaluationRequest,
            observer: Observer, stop: StopToken) -> EvaluationReport: ...
    def disabled(self, source: EvalSource, request: EvaluationRequest,
                 observer: Observer) -> EvaluationReport: ...

@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, snapshot: Snapshot, context: RunContext) -> CheckpointRef: ...
    def load(self, ref: CheckpointRef, *, run_dir: Path) -> Snapshot: ...
    def resolve(self, path: Path) -> CheckpointRef: ...
    def find_latest(self, run_dir: Path, attempt_id: str) -> CheckpointRef | None: ...

@runtime_checkable
class Runner(Protocol):
    def run(self, *, context: RunContext, metadata: RunMetadata, config: ResolvedConfig, source: TrainSource,
            validation_source: EvalSource, runtime: Runtime, evaluator: Evaluator,
            observer: Observer, store: CheckpointStore, stop: StopToken,
            resume: ResumePlan | None = None) -> Completion: ...


# Value validation and encoding belong here; algorithms belong to module owners.
import collections.abc as _abc
import dataclasses as _dc
import hashlib as _hashlib
import json as _json
import math as _math
import types as _types
import typing as _typing


def _mismatch(message: str, *, checkpoint: bool = False) -> MahjongError:
    return MahjongError('CHECKPOINT_INVALID' if checkpoint else 'CONTRACT_MISMATCH', message)


def clone_state(value: StateValue) -> StateValue:
    """Take independent ownership of a pure component-state value."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is list:
        return [clone_state(item) for item in value]
    if type(value) is tuple:
        return tuple(clone_state(item) for item in value)
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise _mismatch('State dictionary keys must be strings or integers')
        return {key: clone_state(item) for key, item in value.items()}
    raise _mismatch(f'State cannot contain a live {type(value).__name__} object')


def derive_seed(root: int, namespace: str) -> int:
    """Derive the documented version-1 independent RNG namespace seed."""
    if type(root) is not int or not 0 <= root < 2**63:
        raise _mismatch('Seed must be an integer in [0, 2**63)')
    if not isinstance(namespace, str) or not namespace or not namespace.isascii() or any(c.isspace() for c in namespace):
        raise _mismatch('Seed namespace must be nonempty ASCII without whitespace')
    # Frozen domain: changing spelling would change all reproducible seed streams.
    digest = _hashlib.sha256(f'majang/v1|{root}|{namespace}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % 2**63


def _validate_json(value: object, path: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and _math.isfinite(value):
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json(item, f'{path}[{index}]')
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            _validate_json(item, f'{path}.{key}')
        return
    raise _mismatch(f'{path}: expected JSON value with string keys and finite numbers')


def _validate_state(value: object, path: str) -> None:
    if isinstance(value, Tensor) or value is None or type(value) in (str, bool, int, float):
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_state(item, f'{path}[{index}]')
        return
    if type(value) is dict and all(type(key) in (str, int) for key in value):
        for key, item in value.items():
            _validate_state(item, f'{path}.{key}')
        return
    raise _mismatch(f'{path}: expected pure component state')


def _validate_type(value: object, annotation: object, path: str) -> None:
    if isinstance(annotation, _typing.TypeAliasType):
        if annotation.__name__ in {'State', 'StateValue'}:
            if annotation.__name__ == 'State' and type(value) is not dict:
                raise _mismatch(f'{path}: expected state dictionary')
            _validate_state(value, path)
            return
        if annotation.__name__ in {'JsonValue', 'ResolvedConfig'}:
            if annotation.__name__ == 'ResolvedConfig' and type(value) is not dict:
                raise _mismatch(f'{path}: expected configuration dictionary')
            _validate_json(value, path)
            return
        annotation = annotation.__value__
    if annotation in (object, _typing.Any):
        return
    origin = _typing.get_origin(annotation)
    args = _typing.get_args(annotation)
    if origin in (_types.UnionType, _typing.Union):
        for candidate in args:
            try:
                _validate_type(value, candidate, path)
                return
            except MahjongError:
                pass
        raise _mismatch(f'{path}: value does not match {annotation}')
    if origin is _typing.Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            raise _mismatch(f'{path}: expected one of {args}')
        return
    if origin in (dict, _abc.Mapping):
        if not isinstance(value, _abc.Mapping):
            raise _mismatch(f'{path}: expected mapping')
        for key, item in value.items():
            _validate_type(key, args[0], f'{path}.key')
            _validate_type(item, args[1], f'{path}.{key}')
        return
    if origin in (list, tuple):
        if type(value) is not origin:
            raise _mismatch(f'{path}: expected {origin.__name__}')
        if origin is tuple and len(args) > 1 and args[1] is not Ellipsis:
            if len(value) != len(args):
                raise _mismatch(f'{path}: tuple length mismatch')
            for index, (item, expected) in enumerate(zip(value, args)):
                _validate_type(item, expected, f'{path}[{index}]')
        else:
            for index, item in enumerate(value):
                _validate_type(item, args[0], f'{path}[{index}]')
        return
    if annotation is type(None):
        if value is not None:
            raise _mismatch(f'{path}: expected null')
        return
    if annotation in (str, bool, int):
        if type(value) is not annotation:
            raise _mismatch(f'{path}: expected {annotation.__name__}')
        return
    if annotation is float:
        if type(value) not in (int, float) or not _math.isfinite(value):
            raise _mismatch(f'{path}: expected finite number')
        return
    if isinstance(annotation, type) and isinstance(value, annotation):
        if _dc.is_dataclass(value):
            validate_contract(value)
        return
    raise _mismatch(f'{path}: expected {annotation}')


def validate_contract(value: object) -> None:
    """Validate documented field types and basic boundary invariants."""
    if not _dc.is_dataclass(value):
        raise _mismatch('Expected a contract dataclass')
    hints = _typing.get_type_hints(type(value))
    for item in _dc.fields(value):
        _validate_type(getattr(value, item.name), hints[item.name], f'{type(value).__name__}.{item.name}')
        if item.name in {'schema_version', 'format_version', 'contract_version', 'state_version'} and getattr(value, item.name) != 1:
            raise _mismatch(f'{type(value).__name__}.{item.name}: unsupported version', checkpoint=isinstance(value, Snapshot))
    for name in ('global_step', 'commit_seq', 'event_seq', 'metric_seq', 'batches', 'samples', 'window_start_step',
                 'batches_read', 'samples_read', 'samples_processed', 'samples_updated', 'skipped_batches',
                 'consecutive_skips', 'target_updates'):
        if hasattr(value, name) and getattr(value, name) is not None and getattr(value, name) < 0:
            raise _mismatch(f'{type(value).__name__}.{name}: must be nonnegative')
    if isinstance(value, Batch) and (value.sample_count <= 0 or value.sample_count != len(value.sample_ids)):
        raise _mismatch('Batch.sample_count must be positive and match sample_ids')
    if isinstance(value, ComponentDescriptor) and value.metric_schema_version not in (None, 1):
        raise _mismatch('ComponentDescriptor.metric_schema_version: unsupported version')
    if isinstance(value, (MetricStat, AggregatedMetric)):
        if value.denominator < 0 or (value.denominator == 0 and value.numerator != 0):
            raise _mismatch('Metric denominator must be nonnegative; zero requires zero numerator')
    if isinstance(value, AggregatedMetric) and ((value.denominator == 0) != (value.value is None)):
        raise _mismatch('AggregatedMetric.value is null exactly for zero denominator')
    if isinstance(value, RecordIdentity):
        valid = ((value.kind == 'training' and value.attempt_id is not None and value.evaluation_id is None)
                 or (value.kind == 'evaluation' and value.attempt_id is None and value.evaluation_id is not None))
        if not valid:
            raise _mismatch('RecordIdentity must use the exclusive training or evaluation identity')


def validate_step_result(result: StepResult, *, phase: Literal['train', 'eval']) -> None:
    validate_contract(result)
    if result.loss is not None and (result.loss.ndim != 0 or not bool(torch.isfinite(result.loss).item())):
        raise MahjongError('NONFINITE_LOSS' if not bool(torch.isfinite(result.loss).all().item()) else 'CONTRACT_MISMATCH',
                          'Loss must be a finite scalar', 'training' if phase == 'train' else 'evaluating')
    if phase == 'train':
        if result.skip_reason is not None:
            if not result.skip_reason or result.loss is not None:
                raise _mismatch('Skip requires loss=None and a nonempty reason')
        elif result.loss is None or not result.loss.requires_grad or 'loss' not in result.metrics:
            raise _mismatch('Training result requires a gradient-bearing scalar loss and loss statistics')
    elif result.skip_reason is not None:
        raise _mismatch('Evaluation uses zero-denominator metrics rather than training skips')


def encode_contract(value: object, *, run_dir: Path | None = None) -> StateValue:
    """Encode to pure values; no custom dataclass is placed in state.pt."""
    if _dc.is_dataclass(value):
        validate_contract(value)
        return {item.name: encode_contract(getattr(value, item.name), run_dir=run_dir) for item in _dc.fields(value)}
    if isinstance(value, Path):
        if value.is_absolute():
            if run_dir is None:
                raise _mismatch('Absolute contract paths require run_dir for relative encoding')
            try:
                return value.relative_to(run_dir).as_posix()
            except ValueError as exc:
                raise _mismatch('Contract path is outside run_dir') from exc
        return value.as_posix()
    if isinstance(value, _abc.Mapping):
        if any(type(key) not in (str, int) for key in value):
            raise _mismatch('Encoded mapping keys must be strings or integers')
        return {key: encode_contract(item, run_dir=run_dir) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(encode_contract(item, run_dir=run_dir) for item in value)
    if type(value) is list:
        return [encode_contract(item, run_dir=run_dir) for item in value]
    return clone_state(value)


def _decode_value(annotation: object, value: object, run_dir: Path | None) -> object:
    if annotation in (object, _typing.Any):
        # Batch payload is opaque during component calls. State encoding still
        # refuses live objects; Batch itself is not part of Snapshot.
        return value
    if isinstance(annotation, _typing.TypeAliasType):
        if annotation.__name__ in {'State', 'StateValue', 'JsonValue', 'ResolvedConfig'}:
            _validate_type(value, annotation, annotation.__name__)
            return clone_state(value)
        annotation = annotation.__value__
    origin, args = _typing.get_origin(annotation), _typing.get_args(annotation)
    if origin in (_types.UnionType, _typing.Union):
        for candidate in args:
            try:
                result = _decode_value(candidate, value, run_dir)
                _validate_type(result, candidate, 'value')
                return result
            except (MahjongError, TypeError, ValueError):
                pass
        raise _mismatch(f'Value does not match {annotation}')
    if isinstance(annotation, type) and _dc.is_dataclass(annotation):
        return decode_contract(annotation, value, run_dir=run_dir)
    if annotation is Path:
        if not isinstance(value, str):
            raise _mismatch('Encoded Path must be a string')
        path = Path(value)
        if path.is_absolute() or '..' in path.parts:
            raise _mismatch('Encoded Path must stay relative to run_dir')
        return path if run_dir is None else run_dir / path
    if origin in (dict, _abc.Mapping):
        if not isinstance(value, _abc.Mapping):
            raise _mismatch('Expected encoded mapping')
        return {_decode_value(args[0], key, run_dir): _decode_value(args[1], item, run_dir) for key, item in value.items()}
    if origin in (list, tuple):
        if type(value) is not origin and not (origin is tuple and type(value) is list):
            raise _mismatch(f'Expected encoded {origin.__name__}')
        if origin is tuple and len(args) > 1 and args[1] is not Ellipsis:
            if len(value) != len(args):
                raise _mismatch('Tuple length mismatch')
            return tuple(_decode_value(expected, item, run_dir) for expected, item in zip(args, value))
        return origin(_decode_value(args[0], item, run_dir) for item in value)
    _validate_type(value, annotation, 'value')
    return clone_state(value)


def decode_contract(cls: type, payload: object, *, run_dir: Path | None = None):
    """Rebuild and validate all required fields of a persisted value object."""
    checkpoint = cls is Snapshot
    try:
        if not _dc.is_dataclass(cls) or type(payload) is not dict:
            raise _mismatch('Expected contract class and encoded dictionary')
        names = {item.name for item in _dc.fields(cls)}
        missing = names - payload.keys()
        extra = payload.keys() - names
        if missing or (extra and cls is not Event):
            raise _mismatch(f'{cls.__name__}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}')
        hints = _typing.get_type_hints(cls)
        return cls(**{name: _decode_value(hints[name], payload[name], run_dir) for name in names})
    except (MahjongError, TypeError, ValueError) as exc:
        if isinstance(exc, MahjongError) and not checkpoint:
            raise
        raise _mismatch(f'{cls.__name__}: {exc}', checkpoint=checkpoint) from exc


def canonical_json(value: JsonValue) -> str:
    _validate_json(value, 'JSON')
    return _json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def encode_json_contract(value: object, *, run_dir: Path | None = None) -> JsonValue:
    """Encode a report/event as JSON values, converting tuple fields to arrays.

    Use encode_contract for state.pt: it preserves tuples and CPU tensors.
    This JSON encoder rejects tensors and integer dictionary keys.
    """
    def convert(item: StateValue) -> JsonValue:
        if type(item) in (tuple, list):
            return [convert(child) for child in item]
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise _mismatch('JSON contract dictionary keys must be strings')
            return {key: convert(child) for key, child in item.items()}
        _validate_json(item, 'JSON contract')
        return item
    return convert(encode_contract(value, run_dir=run_dir))

