"""Synchronous publication and validated reading of schema-1 checkpoints.

The caller owns the run lock and collects a complete safe-boundary Snapshot.
This store never queries runtime, data, or observer components.
"""

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch

from mahjong.contracts import (
    CheckpointRef, LogCutoff, MahjongError, RunContext, Snapshot, canonical_json,
    decode_contract, encode_contract, encode_json_contract,
)


PUBLICATION_STAGES = (
    'state_writing', 'before_manifest', 'before_rename',
    'after_rename_before_latest', 'after_latest',
)
PublicationHook = Callable[[str, Mapping[str, Path]], None]


def _invalid(message: str, **details) -> MahjongError:
    return MahjongError('CHECKPOINT_INVALID', message, 'restoring', details)


def _component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value in {'.', '..'} or '/' in value or '\\' in value or value.startswith('.'):
        raise _invalid(f'{label}: expected a nonempty path component')
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(value: dict) -> bytes:
    return (canonical_json(value) + '\n').encode('utf-8')


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError('expected JSON object')
        canonical_json(value)  # Reject nonfinite numbers and non-JSON values.
        return value, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, MahjongError) as exc:
        raise _invalid(f'Cannot read checkpoint identity JSON: {path}: {exc}', path=str(path)) from exc


def _stream_digest(stream) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _filesystem(path: Path) -> dict:
    result = {'path': str(path.resolve()), 'type': 'unknown'}
    try:
        longest = -1
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            before, after = line.split(' - ', 1)
            mount = Path(before.split()[4].replace('\\040', ' '))
            if path.resolve().is_relative_to(mount) and len(str(mount)) > longest:
                longest = len(str(mount))
                result['type'] = after.split()[0]
    except (OSError, ValueError, IndexError):
        pass  # Environment metadata still records the actual output path.
    return result


class _StateWriter:
    """Expose a barrier after bytes have actually reached the temporary file."""
    def __init__(self, raw, barrier):
        self.raw = raw
        self.barrier = barrier
        self.called = False

    def write(self, data):
        written = self.raw.write(data)
        if written and not self.called:
            self.called = True
            self.raw.flush()
            self.barrier()
        return written

    def flush(self):
        return self.raw.flush()


class FileCheckpointStore:
    """Immutable checkpoint directories with a replaceable attempt latest hint."""
    keep_last = 2
    def __init__(self, *, hook: PublicationHook | None = None):
        self.hook = hook

    def _barrier(self, stage: str, paths: Mapping[str, Path]) -> None:
        if self.hook is not None:
            self.hook(stage, paths)

    def save(self, snapshot: Snapshot, context: RunContext) -> CheckpointRef:
        paths = {}
        try:
            identity = snapshot.identity
            checkpoint_id = _component(identity.checkpoint_id, 'checkpoint_id')
            _component(identity.run_id, 'run_id')
            _component(identity.attempt_id, 'attempt_id')
            if (identity.run_id, identity.attempt_id, identity.global_step) != (context.run_id, context.attempt_id, snapshot.progress.global_step):
                raise ValueError('Snapshot identity differs from context/progress')
            if identity.commit_seq < 1:
                raise ValueError('Checkpoint commit_seq starts at 1')
            if snapshot.observation.cutoff.attempt_id != context.attempt_id:
                raise ValueError('Log cutoff belongs to another attempt')
            root = context.run_dir.resolve()
            attempt = root / 'attempts' / context.attempt_id
            if context.attempt_dir.resolve() != attempt or not attempt.is_dir():
                raise ValueError('RunContext does not refer to its existing attempt directory')
            checkpoints = root / 'checkpoints'
            checkpoints.mkdir(exist_ok=True)
            temporary = checkpoints / f'.tmp-{checkpoint_id}'
            final = checkpoints / checkpoint_id
            latest = attempt / 'latest.json'
            latest_temporary = attempt / f'.tmp-latest-{checkpoint_id}.json'
            paths = {'run_dir': root, 'temporary': temporary, 'checkpoint': final,
                     'state': temporary / 'state.pt', 'manifest': temporary / 'manifest.json',
                     'latest': latest, 'latest_temporary': latest_temporary}
            if final.exists():
                raise FileExistsError(f'Checkpoint is immutable and already exists: {final}')
            temporary.mkdir()
            encoded = encode_contract(snapshot, run_dir=root)
            with paths['state'].open('xb') as raw:
                torch.save(encoded, _StateWriter(raw, lambda: self._barrier('state_writing', paths)))
                raw.flush()
                os.fsync(raw.fileno())
            with paths['state'].open('rb') as stream:
                state_bytes, state_sha256 = _stream_digest(stream)
            self._barrier('before_manifest', paths)
            data = next((descriptor.data_fingerprint for descriptor in snapshot.descriptors if descriptor.kind == 'data'), None)
            manifest = {'schema_version': 1, 'checkpoint_id': checkpoint_id,
                        'run_id': identity.run_id, 'attempt_id': identity.attempt_id,
                        'global_step': identity.global_step, 'commit_seq': identity.commit_seq,
                        'source_checkpoint_id': None if identity.parent_checkpoint is None else identity.parent_checkpoint.checkpoint_id,
                        'created_at': identity.created_at, 'state_bytes': state_bytes,
                        'state_sha256': state_sha256, 'config_hash': snapshot.frozen_config_hash,
                        'data_fingerprint': data, 'log_cutoff': encode_json_contract(snapshot.observation.cutoff),
                        'save_reason': snapshot.save_reason, 'report_ref': snapshot.progress.last_evaluation_ref,
                        'filesystem': _filesystem(root)}
            if any(d.implementation_id == 'mahjong.rl_source/v1' for d in snapshot.descriptors):
                from mahjong.rl.recovery import dependencies
                manifest['rl_dependencies'] = dependencies(snapshot.data)
            manifest_bytes = _json_bytes(manifest)
            _write_bytes(paths['manifest'], manifest_bytes)
            _fsync_directory(temporary)
            self._barrier('before_rename', paths)
            # The caller holds the run lock; the explicit existence check also
            # protects an already existing empty destination directory.
            if final.exists():
                raise FileExistsError(f'Checkpoint destination already exists: {final}')
            os.rename(temporary, final)
            _fsync_directory(checkpoints)
            ref = CheckpointRef(checkpoint_id, identity.run_id, identity.attempt_id,
                                identity.global_step, f'checkpoints/{checkpoint_id}',
                                hashlib.sha256(manifest_bytes).hexdigest(), identity.commit_seq)
            self._barrier('after_rename_before_latest', paths)
            _write_bytes(latest_temporary, _json_bytes(encode_json_contract(ref)))
            os.replace(latest_temporary, latest)
            _fsync_directory(attempt)
            self._barrier('after_latest', paths)
            self.prune(run_dir=root, latest=ref)
            return ref
        except Exception as exc:
            if isinstance(exc, MahjongError) and exc.code == 'CHECKPOINT_WRITE_FAILED':
                raise
            raise MahjongError('CHECKPOINT_WRITE_FAILED', f'Checkpoint publication failed: {exc}',
                              'checkpointing', {'checkpoint_id': snapshot.identity.checkpoint_id,
                                                'published_path': str(paths.get('checkpoint', ''))}) from exc

    def prune(self, *, run_dir: Path, latest: CheckpointRef) -> tuple[str, ...]:
        """Keep the published state and its predecessor, across all attempts.

        The caller holds the run writer lock. Parent references and attempt
        metadata remain historical identities; full snapshots are standalone.
        Never prune before the new state and latest hint are durably published.
        """
        root = run_dir.resolve()
        manifest, actual = self._manifest(root / latest.path)
        if actual != latest or latest.run_id != self._run_identity(root):
            raise _invalid('Cannot prune with an inconsistent latest checkpoint')
        checkpoints = root / 'checkpoints'
        candidates = {}
        for directory in checkpoints.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            if directory.name.startswith('.prune-'):
                shutil.rmtree(directory)  # Finish an interrupted old deletion.
                continue
            if directory.name.startswith('.'):
                continue
            try:
                _, ref = self._manifest(directory)
            except MahjongError:
                continue
            if ref.run_id == latest.run_id:
                candidates[ref.checkpoint_id] = directory
        keep = {latest.checkpoint_id}
        parent = manifest['source_checkpoint_id']
        if parent in candidates and parent not in keep:
            keep.add(parent)
        elif len(keep) < self.keep_last and len(candidates) > len(keep):
            keep.add(max((key for key in candidates if key not in keep),
                         key=lambda key: (candidates[key].stat().st_mtime_ns, key)))
        removed = []
        for checkpoint_id, directory in candidates.items():
            if checkpoint_id in keep:
                continue
            retiring = checkpoints / f'.prune-{checkpoint_id}'
            os.rename(directory, retiring)
            _fsync_directory(checkpoints)
            shutil.rmtree(retiring)
            removed.append(checkpoint_id)
        _fsync_directory(checkpoints)
        return tuple(removed)

    def _manifest(self, directory: Path) -> tuple[dict, CheckpointRef]:
        if directory.name.startswith('.') or directory.parent.name != 'checkpoints' or not directory.is_dir():
            raise _invalid(f'Not a published checkpoint directory: {directory}')
        manifest, manifest_hash = _read_json(directory / 'manifest.json')
        required = {'schema_version', 'checkpoint_id', 'run_id', 'attempt_id', 'global_step', 'commit_seq',
                    'source_checkpoint_id', 'created_at', 'state_bytes', 'state_sha256', 'config_hash',
                    'data_fingerprint', 'log_cutoff', 'save_reason', 'report_ref'}
        missing = required - manifest.keys()
        if missing:
            raise _invalid(f'Missing checkpoint manifest fields: {sorted(missing)}')
        if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
            raise _invalid('Unsupported checkpoint manifest schema_version')
        for name in ('checkpoint_id', 'run_id', 'attempt_id'):
            _component(manifest[name], name)
        if manifest['checkpoint_id'] != directory.name:
            raise _invalid('Checkpoint directory and manifest identity differ')
        for name in ('global_step', 'commit_seq', 'state_bytes'):
            if type(manifest[name]) is not int or manifest[name] < (0 if name == 'global_step' else 1):
                raise _invalid(f'Invalid checkpoint manifest {name}')
        digest = manifest['state_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
            raise _invalid('Invalid checkpoint state SHA256')
        try:
            decode_contract(LogCutoff, manifest['log_cutoff'])
            ref = CheckpointRef(manifest['checkpoint_id'], manifest['run_id'], manifest['attempt_id'],
                                manifest['global_step'], f'checkpoints/{directory.name}', manifest_hash,
                                manifest['commit_seq'])
        except MahjongError as exc:
            raise _invalid(f'Invalid checkpoint reference: {exc}') from exc
        return manifest, ref

    def _run_identity(self, run_dir: Path) -> str:
        manifest, _ = _read_json(run_dir / 'manifest.json')
        return _component(manifest.get('run_id'), 'run manifest.run_id')

    def load(self, ref: CheckpointRef, *, run_dir: Path) -> Snapshot:
        try:
            root = run_dir.resolve()
            checkpoint_id = _component(ref.checkpoint_id, 'checkpoint_id')
            if ref.path != f'checkpoints/{checkpoint_id}':
                raise _invalid('CheckpointRef.path differs from checkpoint identity')
            directory = root / ref.path
            manifest, actual_ref = self._manifest(directory)
            if actual_ref != ref or self._run_identity(root) != ref.run_id:
                raise _invalid('Checkpoint reference or run identity differs from manifest')
            with (directory / 'state.pt').open('rb') as stream:
                size, digest = _stream_digest(stream)
                if size != manifest['state_bytes'] or digest != manifest['state_sha256']:
                    raise _invalid('Checkpoint state length or SHA256 differs from manifest')
                stream.seek(0)
                value = torch.load(stream, map_location='cpu', weights_only=True)
            snapshot = decode_contract(Snapshot, value, run_dir=root)
            identity = snapshot.identity
            data = next((descriptor.data_fingerprint for descriptor in snapshot.descriptors if descriptor.kind == 'data'), None)
            expected = {'checkpoint_id': identity.checkpoint_id, 'run_id': identity.run_id,
                        'attempt_id': identity.attempt_id, 'global_step': identity.global_step,
                        'commit_seq': identity.commit_seq, 'created_at': identity.created_at,
                        'source_checkpoint_id': None if identity.parent_checkpoint is None else identity.parent_checkpoint.checkpoint_id,
                        'config_hash': snapshot.frozen_config_hash, 'data_fingerprint': data,
                        'log_cutoff': encode_json_contract(snapshot.observation.cutoff),
                        'save_reason': snapshot.save_reason, 'report_ref': snapshot.progress.last_evaluation_ref}
            if any(manifest[name] != expected[name] for name in expected):
                raise _invalid('Checkpoint manifest and Snapshot duplicate fields differ')
            if identity.global_step != snapshot.progress.global_step or snapshot.observation.cutoff.attempt_id != identity.attempt_id:
                raise _invalid('Checkpoint progress or observation identity differs')
            return snapshot
        except MahjongError as exc:
            if exc.code == 'CHECKPOINT_INVALID':
                raise
            raise _invalid(f'Invalid checkpoint: {exc}') from exc
        except Exception as exc:
            raise _invalid(f'Cannot load checkpoint: {exc}', checkpoint_id=ref.checkpoint_id) from exc

    def resolve(self, path: Path) -> CheckpointRef:
        directory = path.absolute()
        _, ref = self._manifest(directory)
        if self._run_identity(directory.parent.parent) != ref.run_id:
            raise _invalid('Checkpoint run identity differs from run manifest')
        return ref

    def _attempt_source(self, run_dir: Path, run_id: str, attempt_id: str) -> CheckpointRef | None:
        _component(attempt_id, 'attempt_id')
        metadata, _ = _read_json(run_dir / 'attempts' / attempt_id / 'manifest.json')
        if metadata.get('run_id') != run_id or metadata.get('attempt_id') != attempt_id or 'source_checkpoint' not in metadata:
            raise _invalid('Missing or inconsistent attempt identity/source metadata', attempt_id=attempt_id)
        source = metadata['source_checkpoint']
        if source is None:
            return None
        try:
            ref = decode_contract(CheckpointRef, source)
        except MahjongError as exc:
            raise _invalid(f'Invalid attempt source checkpoint: {exc}') from exc
        if ref.run_id != run_id or ref.commit_seq < 1 or ref.path != f'checkpoints/{ref.checkpoint_id}':
            raise _invalid('Attempt source checkpoint crosses run or has invalid identity')
        return ref

    def find_latest(self, run_dir: Path, attempt_id: str) -> CheckpointRef | None:
        try:
            return self._find_latest(run_dir, attempt_id)
        except MahjongError:
            raise
        except OSError as exc:
            raise _invalid(f'Cannot inspect checkpoint branch: {exc}', attempt_id=attempt_id) from exc

    def _find_latest(self, run_dir: Path, attempt_id: str) -> CheckpointRef | None:
        root = run_dir.resolve()
        run_id = self._run_identity(root)
        seen = set()
        ancestry = []
        bound: CheckpointRef | None = None
        # Identity/source metadata is required even when this attempt already
        # has a valid checkpoint. Validate ancestry before selecting a state.
        while True:
            if attempt_id in seen:
                raise _invalid('Cycle in checkpoint attempt ancestry', attempt_id=attempt_id)
            seen.add(attempt_id)
            source = self._attempt_source(root, run_id, attempt_id)
            ancestry.append((attempt_id, bound))
            if source is None:
                break
            bound = source
            attempt_id = source.attempt_id
        for attempt_id, bound in ancestry:
            hint = root / 'attempts' / attempt_id / 'latest.json'
            if hint.exists():
                try:
                    decode_contract(CheckpointRef, _read_json(hint)[0])
                except MahjongError:
                    pass  # A hint does not override immutable published states.
            candidates = []
            checkpoints = root / 'checkpoints'
            if checkpoints.is_dir():
                for directory in checkpoints.iterdir():
                    if not directory.is_dir() or directory.name.startswith('.'):
                        continue
                    try:
                        _, ref = self._manifest(directory)
                    except MahjongError:
                        continue
                    if ref.run_id != run_id or ref.attempt_id != attempt_id:
                        continue
                    if bound is not None:
                        if ref.commit_seq > bound.commit_seq:
                            continue
                        if ref.commit_seq == bound.commit_seq and ref != bound:
                            continue
                    candidates.append(ref)
            for ref in sorted(candidates, key=lambda item: item.commit_seq, reverse=True):
                try:
                    self.load(ref, run_dir=root)
                    return ref
                except MahjongError:
                    continue
        return None
