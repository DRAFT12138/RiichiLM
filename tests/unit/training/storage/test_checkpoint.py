from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest
import torch

from mahjong.contracts import (
    CheckpointRef, MahjongError, RunContext, canonical_json, encode_contract, encode_json_contract,
)
from mahjong.training.checkpoint import FileCheckpointStore, PUBLICATION_STAGES
from tests.fixtures.contracts import make_snapshot


def write_json(path, value):
    path.write_text(canonical_json(value) + '\n')


def make_run(tmp_path):
    root = tmp_path / 'R'
    root.mkdir()
    write_json(root / 'manifest.json', {'run_id': 'R'})
    return root


def make_attempt(root, attempt='A1', source=None):
    directory = root / 'attempts' / attempt
    directory.mkdir(parents=True)
    write_json(directory / 'manifest.json', {'run_id': 'R', 'attempt_id': attempt,
                'source_checkpoint': None if source is None else encode_json_contract(source)})
    return RunContext('R', attempt, root, directory, 'new' if source is None else 'resume', source)


def snapshot_for(context, checkpoint='C1', *, step=0, seq=1, parent=None):
    snapshot = make_snapshot()
    progress = replace(snapshot.progress, global_step=step)
    identity = replace(snapshot.identity, run_id=context.run_id, attempt_id=context.attempt_id,
                       checkpoint_id=checkpoint, global_step=step, commit_seq=seq,
                       parent_checkpoint=parent if parent is not None else context.source_checkpoint)
    observation = replace(snapshot.observation,
                          cutoff=replace(snapshot.observation.cutoff, attempt_id=context.attempt_id))
    return replace(snapshot, identity=identity, progress=progress, observation=observation,
                   target_updates=max(2, step))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_values_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_values_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for item, other in zip(left, right):
            assert_values_equal(item, other)
    else:
        assert left == right


def test_v031_round_trip_and_new_process_restricted_load(tmp_path):
    root = make_run(tmp_path)
    context = make_attempt(root)
    snapshot = snapshot_for(context)
    stages = []
    store = FileCheckpointStore(hook=lambda stage, paths: stages.append(stage))
    ref = store.save(snapshot, context)
    assert stages == list(PUBLICATION_STAGES)
    assert store.resolve(root / ref.path) == ref
    restored = store.load(ref, run_dir=root)
    assert_values_equal(encode_contract(snapshot), encode_contract(restored))
    code = '''
import json, sys
from pathlib import Path
from mahjong.training.checkpoint import FileCheckpointStore
store = FileCheckpointStore()
root = Path(sys.argv[1])
ref = store.resolve(root/'checkpoints'/'C1')
snapshot = store.load(ref, run_dir=root)
print(json.dumps({'checkpoint_id': snapshot.identity.checkpoint_id, 'step': snapshot.progress.global_step,
                  'tensor_device': snapshot.rng.torch_cpu.device.type}))
'''
    result = subprocess.run([sys.executable, '-c', code, str(root)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'checkpoint_id': 'C1', 'step': 0, 'tensor_device': 'cpu'}


@pytest.mark.parametrize('corruption', ['manifest_version', 'state_version', 'contract_version', 'identity',
                                       'cutoff', 'length', 'hash', 'truncated', 'unsupported_pickle'])
def test_v031_corruption_is_rejected_even_when_other_hashes_are_updated(tmp_path, corruption):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    ref = store.save(snapshot_for(context), context)
    directory = root / ref.path
    manifest_path, state_path = directory / 'manifest.json', directory / 'state.pt'
    manifest = json.loads(manifest_path.read_text())
    if corruption == 'manifest_version':
        manifest['schema_version'] = 2
    elif corruption in {'state_version', 'contract_version', 'identity', 'cutoff', 'unsupported_pickle'}:
        state = torch.load(state_path, weights_only=True)
        if corruption == 'state_version':
            state['format_version'] = 2
        elif corruption == 'contract_version':
            state['contract_version'] = 2
        elif corruption == 'identity':
            state['identity']['run_id'] = 'different-run'
        elif corruption == 'cutoff':
            state['observation']['cutoff']['attempt_id'] = 'different-attempt'
        else:
            state = Path('not-a-pure-value')
        torch.save(state, state_path)
        manifest['state_bytes'] = state_path.stat().st_size
        manifest['state_sha256'] = checksum(state_path)
    elif corruption == 'length':
        manifest['state_bytes'] += 1
    elif corruption == 'hash':
        manifest['state_sha256'] = '0' * 64
    else:
        state_path.write_bytes(state_path.read_bytes()[:50])
    write_json(manifest_path, manifest)
    modified_ref = replace(ref, manifest_hash=checksum(manifest_path))
    with pytest.raises(MahjongError) as error:
        store.load(modified_ref, run_dir=root)
    assert error.value.code == 'CHECKPOINT_INVALID'


def test_v031_ref_hash_wrong_path_missing_fields_and_tmp_reject(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    ref = store.save(snapshot_for(context), context)
    for invalid in [replace(ref, manifest_hash='0'*64), replace(ref, path='../checkpoints/C1'),
                    replace(ref, run_id='different-run'), replace(ref, attempt_id='A2')]:
        with pytest.raises(MahjongError) as error:
            store.load(invalid, run_dir=root)
        assert error.value.code == 'CHECKPOINT_INVALID'
    temporary = root / 'checkpoints' / '.tmp-C2'
    temporary.mkdir()
    with pytest.raises(MahjongError):
        store.resolve(temporary)
    manifest_path = root / ref.path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest['state_bytes']
    write_json(manifest_path, manifest)
    with pytest.raises(MahjongError):
        store.resolve(root / ref.path)


def test_v031_same_run_tree_can_be_moved(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    ref = store.save(snapshot_for(context), context)
    moved = tmp_path / 'moved'
    root.rename(moved)
    assert store.load(ref, run_dir=moved).identity.checkpoint_id == 'C1'
    assert store.find_latest(moved, 'A1') == ref


@pytest.mark.parametrize('stage', PUBLICATION_STAGES)
def test_v032_real_sigkill_at_each_publication_boundary_preserves_old_state(tmp_path, stage):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    old = store.save(snapshot_for(context), context)
    old_hashes = {name: checksum(root / old.path / name) for name in ('state.pt', 'manifest.json')}
    code = '''
import json, sys
from pathlib import Path
from dataclasses import replace
from mahjong.contracts import RunContext
from mahjong.training.checkpoint import FileCheckpointStore
from tests.fixtures.contracts import make_snapshot
root = Path(sys.argv[1]); stage = sys.argv[2]
context = RunContext('R','A1',root,root/'attempts'/'A1','new',None)
snapshot = make_snapshot()
snapshot = replace(snapshot, identity=replace(snapshot.identity,checkpoint_id='C2',commit_seq=2))
def barrier(current, paths):
 if current == stage:
  print(json.dumps({'stage': current, 'state_exists': paths['state'].exists(),
                    'checkpoint_exists': paths['checkpoint'].exists()}), flush=True)
  sys.stdin.readline()
FileCheckpointStore(hook=barrier).save(snapshot,context)
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(root), stage],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        readable, _, _ = select.select([child.stdout], [], [], 30)
        assert readable, 'Child did not reach the publication barrier'
        line = child.stdout.readline()
        assert line, child.stderr.read()
        ready = json.loads(line)
        assert ready['stage'] == stage
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    for name, before in old_hashes.items():
        assert checksum(root / old.path / name) == before
    assert store.load(old, run_dir=root).identity.checkpoint_id == 'C1'
    found = store.find_latest(root, 'A1')
    expected = 'C2' if stage in {'after_rename_before_latest', 'after_latest'} else 'C1'
    assert found.checkpoint_id == expected
    assert not found.path.split('/')[-1].startswith('.')


@pytest.mark.parametrize('stage', PUBLICATION_STAGES[:-1])
def test_v032_actual_eacces_at_write_or_rename_edge_preserves_old_state(tmp_path, stage):
    if os.geteuid() == 0:
        pytest.skip('Unix permission errors require a non-root process')
    root = make_run(tmp_path)
    context = make_attempt(root)
    store = FileCheckpointStore()
    old = store.save(snapshot_for(context), context)
    original = checksum(root / old.path / 'state.pt')
    changed = []
    def hook(current, paths):
        if current == stage:
            directory = (paths['temporary'] if stage in {'state_writing', 'before_manifest'} else
                         paths['checkpoint'].parent if stage == 'before_rename' else paths['latest'].parent)
            changed.append((directory, directory.stat().st_mode & 0o777))
            directory.chmod(0o500)
    try:
        with pytest.raises(MahjongError) as error:
            FileCheckpointStore(hook=hook).save(snapshot_for(context, 'C2', seq=2), context)
        assert error.value.code == 'CHECKPOINT_WRITE_FAILED' and error.value.exit_code == 1
        assert isinstance(error.value.__cause__, PermissionError)
        assert error.value.__cause__.errno == errno.EACCES
    finally:
        for directory, mode in changed:
            directory.chmod(mode)
    assert checksum(root / old.path / 'state.pt') == original
    assert store.load(old, run_dir=root).identity.checkpoint_id == 'C1'
    assert store.find_latest(root, 'A1').checkpoint_id == ('C2' if stage == 'after_rename_before_latest' else 'C1')


@pytest.mark.parametrize('stage', ['state_writing', 'before_manifest', 'after_rename_before_latest'])
def test_v032_native_enospc_write_failure_with_dev_full(tmp_path, monkeypatch, stage):
    root = make_run(tmp_path)
    context = make_attempt(root)
    store = FileCheckpointStore()
    old = store.save(snapshot_for(context), context)
    original = checksum(root / old.path / 'state.pt')
    import mahjong.training.checkpoint as module
    real_write = module._write_bytes
    real_state_write = module._StateWriter.write
    armed = False
    def full_write():
        with Path('/dev/full').open('wb', buffering=0) as stream:
            stream.write(b'checkpoint-I/O-fault')  # Real kernel ENOSPC.
    def write_bytes(path, data):
        if armed:
            full_write()
        return real_write(path, data)
    def state_write(self, data):
        if armed:
            full_write()
        return real_state_write(self, data)
    def hook(current, paths):
        nonlocal armed
        if current == stage:
            armed = True
    monkeypatch.setattr(module, '_write_bytes', write_bytes)
    monkeypatch.setattr(module._StateWriter, 'write', state_write)
    with pytest.raises(MahjongError) as error:
        FileCheckpointStore(hook=hook).save(snapshot_for(context, 'C2', seq=2), context)
    assert error.value.code == 'CHECKPOINT_WRITE_FAILED' and error.value.exit_code == 1
    # torch.save may add a ZIP-finalization RuntimeError above the original
    # OSError. Its exception chain must still preserve the real kernel errno.
    nested = error.value
    errors = []
    while nested is not None:
        errors.append(nested)
        nested = nested.__cause__ if nested.__cause__ is not None else nested.__context__
    assert any(isinstance(item, OSError) and item.errno == errno.ENOSPC for item in errors)
    assert checksum(root / old.path / 'state.pt') == original
    assert store.load(old, run_dir=root).identity.checkpoint_id == 'C1'
    assert store.find_latest(root, 'A1').checkpoint_id == ('C2' if stage == 'after_rename_before_latest' else 'C1')


def test_v032_latest_failure_reports_failure_but_published_state_is_found(tmp_path):
    root = make_run(tmp_path)
    context = make_attempt(root)
    store = FileCheckpointStore()
    old = store.save(snapshot_for(context), context)
    before = checksum(root / old.path / 'state.pt')
    def hook(stage, paths):
        if stage == 'after_rename_before_latest':
            raise OSError(errno.ENOSPC, 'injected latest publication failure')
    with pytest.raises(MahjongError) as error:
        FileCheckpointStore(hook=hook).save(snapshot_for(context, 'C2', seq=2), context)
    assert error.value.code == 'CHECKPOINT_WRITE_FAILED' and error.value.exit_code == 1
    assert checksum(root / old.path / 'state.pt') == before
    assert store.find_latest(root, 'A1').checkpoint_id == 'C2'


def test_v033_same_step_sequence_order_old_hint_and_invalid_explicit_ref(tmp_path):
    root = make_run(tmp_path)
    context = make_attempt(root)
    store = FileCheckpointStore()
    first = store.save(snapshot_for(context, 'C1', step=5), context)
    second = store.save(snapshot_for(context, 'C2', step=5, seq=2, parent=first), context)
    write_json(context.attempt_dir / 'latest.json', encode_json_contract(first))
    assert store.find_latest(root, 'A1') == second
    (root / second.path / 'state.pt').write_bytes(b'corrupt')
    with pytest.raises(MahjongError):
        store.load(second, run_dir=root)
    assert store.find_latest(root, 'A1') == first


def test_v033_current_attempt_and_bounded_ancestor_branches(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    a1 = make_attempt(root)
    c1 = store.save(snapshot_for(a1, 'C1', step=40, seq=1), a1)
    c2 = store.save(snapshot_for(a1, 'C2', step=60, seq=2, parent=c1), a1)
    store.save(snapshot_for(a1, 'C3', step=100, seq=3, parent=c2), a1)
    a2 = make_attempt(root, 'A2', c2)
    assert store.find_latest(root, 'A2') == c2
    d1 = store.save(snapshot_for(a2, 'D1', step=80, seq=1), a2)
    store.save(snapshot_for(a2, 'D2', step=90, seq=2, parent=d1), a2)
    make_attempt(root, 'A3', d1)
    assert store.find_latest(root, 'A3') == d1
    (root / d1.path / 'state.pt').write_bytes(b'corrupt')
    # Retention removed older branch states; ancestry metadata remains usable,
    # but it cannot resurrect the pruned files or pick a future sibling D2.
    assert store.find_latest(root, 'A3') is None
    assert not (root / c2.path).exists()
    assert not (root / c1.path).exists()
    make_attempt(root, 'A4', c1)
    assert store.find_latest(root, 'A4') is None


def test_v033_empty_and_only_tmp_return_none_missing_identity_and_cycle_reject(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    (root / 'checkpoints' / '.tmp-C1').mkdir(parents=True)
    assert store.find_latest(root, 'A1') is None
    with pytest.raises(MahjongError):
        store.find_latest(root, 'missing')
    own_source = CheckpointRef('missing', 'R', 'A1', 0, 'checkpoints/missing', '0'*64, 1)
    write_json(context.attempt_dir / 'manifest.json', {'run_id':'R','attempt_id':'A1',
                                                      'source_checkpoint':encode_json_contract(own_source)})
    with pytest.raises(MahjongError) as error:
        store.find_latest(root, 'A1')
    assert 'Cycle' in error.value.message
    write_json(context.attempt_dir / 'manifest.json', {'run_id':'R','attempt_id':'A1'})
    with pytest.raises(MahjongError):
        store.find_latest(root, 'A1')


def test_v033_published_directories_are_never_overwritten(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    first = store.save(snapshot_for(context), context)
    before = checksum(root / first.path / 'state.pt')
    with pytest.raises(MahjongError) as error:
        store.save(snapshot_for(context), context)
    assert error.value.code == 'CHECKPOINT_WRITE_FAILED'
    assert checksum(root / first.path / 'state.pt') == before
    (root / 'checkpoints' / 'empty-existing').mkdir()
    with pytest.raises(MahjongError):
        store.save(snapshot_for(context, 'empty-existing'), context)
    assert list((root / 'checkpoints' / 'empty-existing').iterdir()) == []


def test_v033_ancestry_cycle_is_rejected_even_with_valid_current_checkpoint(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    current = store.save(snapshot_for(context), context)
    write_json(context.attempt_dir / 'manifest.json', {'run_id':'R','attempt_id':'A1',
                                                      'source_checkpoint':encode_json_contract(current)})
    with pytest.raises(MahjongError) as error:
        store.find_latest(root, 'A1')
    assert 'Cycle' in error.value.message


def test_retention_keeps_two_and_loads_without_pruned_parent_files(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    refs = []
    for step in range(1, 5):
        parent = refs[-1] if refs else None
        ref = store.save(snapshot_for(context, f'C{step}', step=step, seq=step, parent=parent), context)
        refs.append(ref)
        published = {p.name for p in (root/'checkpoints').iterdir() if not p.name.startswith('.')}
        assert published == {r.checkpoint_id for r in refs[-2:]}
        assert store.find_latest(root, 'A1') == ref
    # C3 names the deleted C2 as parent; full state loading is self-contained.
    assert store.load(refs[-2], run_dir=root).identity.parent_checkpoint == refs[-3]
    resumed = make_attempt(root, 'A2', refs[-2])
    next_ref = store.save(snapshot_for(resumed, 'D1', step=4), resumed)
    assert {p.name for p in (root/'checkpoints').iterdir()} == {'C3', 'D1'}
    assert store.find_latest(root, 'A2') == next_ref


@pytest.mark.parametrize('failure_stage', ['before_manifest', 'after_rename_before_latest'])
def test_retention_never_deletes_old_states_when_publication_fails(tmp_path, failure_stage):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    first = store.save(snapshot_for(context, 'C1'), context)
    second = store.save(snapshot_for(context, 'C2', seq=2, parent=first), context)
    def hook(stage, paths):
        if stage == failure_stage:
            raise OSError(errno.ENOSPC, 'injected publication failure')
    with pytest.raises(MahjongError, match='CHECKPOINT_WRITE_FAILED'):
        FileCheckpointStore(hook=hook).save(snapshot_for(context, 'C3', seq=3, parent=second), context)
    assert store.load(first, run_dir=root).identity.checkpoint_id == 'C1'
    assert store.load(second, run_dir=root).identity.checkpoint_id == 'C2'


def test_retention_finishes_interrupted_deletion_and_preserves_logs_and_foreign_dirs(tmp_path):
    root, store = make_run(tmp_path), FileCheckpointStore()
    context = make_attempt(root)
    first = store.save(snapshot_for(context, 'C1'), context)
    stale = root/'checkpoints'/'.prune-old'
    stale.mkdir()
    (stale/'state.pt').write_bytes(b'old')
    unknown = root/'checkpoints'/'unknown'
    unknown.mkdir()
    (context.attempt_dir/'events.jsonl').write_text('retained\n')
    second = store.save(snapshot_for(context, 'C2', seq=2, parent=first), context)
    assert not stale.exists()
    assert unknown.is_dir()
    assert (context.attempt_dir/'events.jsonl').read_text() == 'retained\n'
    assert store.load(second, run_dir=root).identity.checkpoint_id == 'C2'
