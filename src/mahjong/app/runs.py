"""Run identities, relative references and single-writer ownership."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from omegaconf import OmegaConf

from mahjong.contracts import (
    CheckpointRef, EvaluationContext, MahjongError, RunContext, canonical_json,
    encode_json_contract,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def config_hash(config):
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (canonical_json(encode_json_contract(value)) + '\n').encode())


def write_config(path, config):
    atomic_bytes(path, OmegaConf.to_yaml(OmegaConf.create(config)).encode())


def preflight(config, *, output=True):
    import torch
    device = config['runtime']['device']
    try:
        if device == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('cuda:0 is unavailable')
        probe = torch.ones(1, device=device)
        if device == 'cuda:0':
            torch.cuda.synchronize()
        del probe
    except Exception as exc:
        raise MahjongError('DEVICE_UNAVAILABLE', f'runtime.device={device}: {exc}') from exc
    if output:
        root = Path(config['output']['root'])
        try:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=root) as stream:
                stream.write(b'mahjong-output-preflight')
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise MahjongError('CONFIG_INVALID', f'output.root={root}: {exc}') from exc


@contextmanager
def lock_run(run_dir: Path):
    try:
        stream = (run_dir / '.writer.lock').open('a+')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if 'stream' in locals():
            stream.close()
        raise MahjongError('CONFIG_INVALID', f'run writer lock unavailable: {run_dir}: {exc}') from exc
    try:
        yield
    finally:
        fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def _identity(run_dir):
    try:
        identity = json.loads((run_dir / 'run.json').read_text())
        if type(identity['run_id']) is not str or not identity['run_id']:
            raise ValueError('run_id is missing')
        return identity['run_id']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MahjongError('CHECKPOINT_INVALID', f'run identity invalid: {run_dir}: {exc}') from exc


def _attempt(context, config):
    context.attempt_dir.mkdir(parents=True, exist_ok=False)
    write_config(context.attempt_dir / 'resolved_config.yaml', config)
    atomic_json(context.attempt_dir / 'manifest.json', {
        'schema_version': 1, 'run_id': context.run_id, 'attempt_id': context.attempt_id,
        'mode': context.mode, 'source_checkpoint': context.source_checkpoint,
        'started_at': utc_now(), 'config_hash': config_hash(config),
    })
    atomic_json(context.attempt_dir / 'lineage.json', {
        'schema_version': 1, 'source_checkpoint': context.source_checkpoint,
        'lineage': [],
    })


def create_run(config) -> RunContext:
    run_id, attempt_id = str(uuid.uuid4()), str(uuid.uuid4())
    run_dir = Path(config['output']['root']) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    context = RunContext(run_id, attempt_id, run_dir, run_dir / 'attempts' / attempt_id, 'new', None)
    atomic_json(run_dir / 'run.json', {
        'schema_version': 1, 'run_id': run_id, 'experiment': config['experiment']['name'],
        'created_at': utc_now(),
    })
    write_config(run_dir / 'resolved_config.yaml', config)
    _attempt(context, config)
    return context


def create_attempt(run_dir, source, config) -> RunContext:
    run_id = _identity(run_dir)
    if source.run_id != run_id:
        raise MahjongError('CHECKPOINT_INVALID', 'source checkpoint run_id disagrees with run identity')
    attempt_id = str(uuid.uuid4())
    context = RunContext(run_id, attempt_id, run_dir, run_dir / 'attempts' / attempt_id, 'resume', source)
    _attempt(context, config)
    return context


def activate(context):
    atomic_json(context.run_dir / 'active_attempt.json', {
        'schema_version': 1, 'run_id': context.run_id, 'attempt_id': context.attempt_id,
    })


def record_weights_source(context, reference):
    path = context.attempt_dir / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(mode='weights', init_from=encode_json_contract(reference))
    atomic_json(path, manifest)


def create_evaluation(run_dir, source) -> EvaluationContext:
    if source.run_id != _identity(run_dir):
        raise MahjongError('CHECKPOINT_INVALID', 'evaluation source run_id mismatch')
    evaluation_id = str(uuid.uuid4())
    path = run_dir / 'evaluations' / evaluation_id
    path.mkdir(parents=True, exist_ok=False)
    return EvaluationContext(evaluation_id, path, run_dir, source)


def publish_metadata(context, config, metadata, *, argv, changes=(), observation=None, init_from=None):
    path = context.attempt_dir / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update({
        'mode': context.mode,
        'metadata': encode_json_contract(metadata), 'argv': list(argv),
        'config_changes': encode_json_contract(changes), 'init_from': encode_json_contract(init_from),
        'implementation_constants': {
            'runtime': 'Fabric', 'optimizer_foreach': False, 'optimizer_fused': False,
            'checkpoint_retention': 'last_2_per_run', 'tensorboard_max_queue': 256, 'retry_count': 0,
        },
    })
    atomic_json(path, manifest)
    if context.mode != 'resume':
        atomic_json(context.run_dir / 'manifest.json', manifest)
    lineage = [] if observation is None else list(observation.lineage) + [observation.cutoff]
    atomic_json(context.attempt_dir / 'lineage.json', {
        'schema_version': 1, 'source_checkpoint': context.source_checkpoint,
        'lineage': encode_json_contract(lineage),
    })


def active_attempt(run_dir):
    try:
        value = json.loads((run_dir / 'active_attempt.json').read_text())
        if value['run_id'] != _identity(run_dir) or type(value['attempt_id']) is not str:
            raise ValueError('active identity mismatch')
        return value['attempt_id']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MahjongError('CHECKPOINT_INVALID', f'active attempt unavailable: {exc}') from exc


def writer_is_locked(run_dir):
    # Read-only inspect does not create a missing lock file or write PID data.
    try:
        stream = (run_dir / '.writer.lock').open('r')
    except FileNotFoundError:
        return False
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False
