"""Durable publication and explicit independent repair builds."""
from copy import deepcopy
from collections import OrderedDict
from contextlib import ExitStack
import json
from functools import lru_cache
import os
from pathlib import Path
import time

from .catalog import Source, validate_catalog
from .common import (atomic_json, connect, digest, file_identity, fixed_lock,
                     require, sync_directory, sync_file, TEXT_STORAGE_VERSION,
                     is_text_storage)
from .storage import check_members, inspect_shard, meta
from .common import read_json, json_path, atomic_bytes
from .compression import compressed_storage, product_names

_verified = OrderedDict()


def _snapshot(root, path, task):
    from .archive_source import fingerprint
    from .text_storage import product_paths
    final, work = product_paths(root,task['task_id'])
    require(not (final.exists() and work.exists()), 'ARTIFACT_CONFLICT', 'two product directories')
    directory=final if final.exists() else work
    return {str(p):fingerprint(p.stat()) for p in
            (Path(path), *(directory/name for name in product_names(_build_catalog(str(Path(root).resolve())))))}


def verification_receipt(root, path, catalog, task, info):
    return dict(build_id=catalog['build_id'],task_id=task['task_id'],path=str(path),
                snapshot=_snapshot(root,path,task),info=info)


def accept_verification(root, catalog, task, receipt):
    """Process-local trust from our worker, invalidated by any physical change."""
    require(receipt['build_id']==catalog['build_id'] and receipt['task_id']==task['task_id'],
            'BINDING_MISMATCH','verification receipt')
    if receipt['snapshot']==_snapshot(root,Path(receipt['path']),task):
        key=(str(receipt['path']),catalog['build_id'])
        _verified[key]=(receipt['snapshot'],receipt['info'])
        _verified.move_to_end(key)
        while len(_verified)>32:_verified.popitem(last=False)


def _inspect_task(path,root,catalog,task,codec):
    if catalog['binding'].get('task_state') != 'append-state/1':
        return inspect_shard(path,catalog,task,codec=codec)
    key=(str(path),catalog['build_id'])
    snapshot=_snapshot(root,path,task)
    cached=_verified.get(key)
    if cached is not None and cached[0]==snapshot:
        return cached[1]
    info=inspect_shard(path,catalog,task,codec=codec)
    accept_verification(root,catalog,task,verification_receipt(root,path,catalog,task,info))
    return info


@lru_cache(maxsize=16)
def _build_catalog(root):
    """Read an immutable build catalog once per process and build root."""
    return read_json(Path(root) / 'build.json')


def task_paths(root, task_id):
    root = Path(root)
    name = f'part-{task_id:06d}.sqlite'
    catalog = _build_catalog(str(root.resolve()))
    text = is_text_storage(catalog['binding']['storage'])
    final_name = name + '.zst' if compressed_storage(catalog) else name
    return root / ('tasks' if text else 'shards') / final_name, root / '.work' / name, root / '.locks' / f'part-{task_id:06d}.lock'


def initialize_build(parent, catalog):
    validate_catalog(catalog)
    root = Path(parent) / catalog['build_id']
    root.mkdir(parents=True, exist_ok=True)
    for name in ('.work', 'shards', '.locks', 'tasks'):
        (root / name).mkdir(exist_ok=True)
    with fixed_lock(root / '.locks' / 'coordinator.lock'):
        path = root / 'build.json'
        if json_path(path).exists():
            require(read_json(path) == catalog, 'BINDING_MISMATCH', 'existing build')
        else:
            atomic_json(path, catalog, compressed=compressed_storage(catalog))
    return root


def isolate(root, path, *, verified_residual=False):
    root, path = Path(root), Path(path)
    require(not json_path(root / 'manifest.json').exists() or (verified_residual and path.parent==root/'.work'),
            'REPAIR_REQUIRED', 'published builds are immutable')
    target_dir = root / '.quarantine'
    target_dir.mkdir(exist_ok=True)
    target = target_dir / f'{path.parent.name}-{path.name}-{time.time_ns()}'
    os.rename(path, target)
    sync_directory(path.parent)
    sync_directory(target_dir)
    return target


def reusable_locked(root, catalog, task, *, codec):
    """Call only while holding the fixed task lock, before opening .work."""
    final, work, _ = task_paths(root, task['task_id'])
    final_info = _inspect_task(final, root, catalog, task, codec) if final.exists() else None
    work_info = None
    if work.exists():
        connection = connect(work)
        try:
            marker = meta(connection, 'complete')
            if final_info is not None and marker is None:
                work_info=check_members(connection,catalog,task,codec=codec)
        finally:
            connection.close()
        if marker is None and is_text_storage(catalog['binding']['storage']):
            from .text_storage import inspect_unfinished_text
            inspect_unfinished_text(work, catalog, task)
        if marker is not None:
            work_info = _inspect_task(work, root, catalog, task, codec)
    if final_info is not None and work.exists():
        require(work_info == final_info, 'ARTIFACT_CONFLICT', 'final/work differ')
        isolate(root, work, verified_residual=True)
    if final_info is not None:
        return final, final_info
    if work_info is not None:
        return work, work_info
    return None


def shard_record(root, path, task, info):
    size, sha = file_identity(path)
    return dict(task_id=task['task_id'], path=str(path.relative_to(root)),
                bytes=size, sha256=sha, **info)


def publish_locked(root, path, catalog, task, *, codec):
    """Coordinator records READY before calling; records DONE after return."""
    final, work, _ = task_paths(root, task['task_id'])
    reusable = reusable_locked(root, catalog, task, codec=codec)
    require(reusable is not None, 'INCOMPLETE_SHARD', str(path))
    actual, info = reusable
    require(Path(path) == actual or actual == final, 'ARTIFACT_CONFLICT', 'unexpected source')
    if actual == work:
        if is_text_storage(catalog['binding']['storage']):
            from .text_storage import publish_products
            publish_products(Path(root), task, info,
                             verified=catalog['binding'].get('task_state')=='append-state/1')
        sync_file(work)
        require(not final.exists(), 'ARTIFACT_CONFLICT', 'never overwrite a shard')
        if compressed_storage(catalog):
            from compression import zstd
            atomic_bytes(final, zstd.compress(work.read_bytes(), level=9))
            work.unlink()
        else:
            os.rename(work, final)
        sync_directory(work.parent)
        sync_directory(final.parent)
        if is_text_storage(catalog['binding']['storage']):
            import shutil
            fragments = work.with_suffix('.rows')
            if fragments.exists():
                shutil.rmtree(fragments)
                sync_directory(fragments.parent)
    return shard_record(Path(root), final, task, info)


def finish_build(root, catalog, records, *, hook=None):
    root = Path(root)
    ordered = sorted(records, key=lambda r: r['task_id'])
    require([r['task_id'] for r in ordered] == [t['task_id'] for t in catalog['tasks']],
            'INCOMPLETE_BUILD', 'published task set')
    path = root / 'manifest.json'
    published = ([{k:v for k,v in record.items() if k not in ('path','bytes','sha256')}
                  for record in ordered] if catalog['binding'].get('lean_tasks') else ordered)
    if json_path(path).exists():
        # Must precede any summary write.
        existing = read_json(path)
        require(existing['build_id'] == catalog['build_id'] and existing['binding'] == catalog['binding'] and
                existing['shards'] == published and
                file_identity(root / 'summary.json')[1] == existing['summary_sha256'],
                'REPAIR_REQUIRED', 'existing manifest mismatch')
        if catalog['binding'].get('lean_tasks'):
            cleanup_task_state(root, published)
        return existing
    groups = {}
    source_map = {s['source_key']: s for s in catalog['sources']}
    for task in catalog['tasks']:
        final, _, _ = task_paths(root, task['task_id'])
        connection = connect(final, readonly=True)
        try:
            for row in connection.execute('SELECT source_key,status,counts FROM games'):
                source = source_map[row['source_key']]
                key = f"{source['platform']}/{source['shard']}"
                group = groups.setdefault(key, dict(inputs=0, raw_bytes=0, statuses={}, counts={}))
                group['inputs'] += 1
                group['raw_bytes'] += source['raw_bytes']
                group['statuses'][row['status']] = group['statuses'].get(row['status'], 0) + 1
                for name, count in json.loads(row['counts']).items():
                    group['counts'][name] = group['counts'].get(name, 0) + count
        finally:
            connection.close()
    summary = dict(build_id=catalog['build_id'], inputs=len(catalog['sources']), groups=groups,
                   statuses={name: sum(r['statuses'][name] for r in ordered)
                             for name in ('success', 'partial', 'failed', 'skipped')},
                   counts={name: sum(r[name] for r in ordered) for name in ('games', 'rounds', 'views', 'anchors', 'tokens')},
                   raw_bytes=sum(s['raw_bytes'] for s in catalog['sources']),
                   shard_bytes=sum(sum(p['bytes'] for p in r['products']) if 'products' in r else r['bytes']
                                   for r in ordered),
                   failure_reasons={})
    for name in ('policy_candidate_sets','policy_target_matches','policy_target_misses',
                 'policy_target_unresolved','policy_samples'):
        summary['counts'][name] = sum(group['counts'].get(name,0) for group in groups.values())
    for record in ordered:
        connection = connect(root / record['path'], readonly=True)
        try:
            # One primary error per source in the denominator.
            for row in connection.execute('SELECT source_key,detail FROM errors ORDER BY source_key,error_id'):
                detail = json.loads(row['detail'])
                if detail.get('primary', True):
                    name = detail['code']
                    summary['failure_reasons'][name] = summary['failure_reasons'].get(name, 0) + 1
        finally:
            connection.close()
    atomic_json(root / 'summary.json', summary)
    if hook:
        hook('SUMMARY_PUBLISHED', dict(build_id=catalog['build_id']))
    supervised = (catalog['binding'].get('backend') == 'rust' and
                  catalog['binding']['storage'] == TEXT_STORAGE_VERSION)
    manifest = dict(build_id=catalog['build_id'], binding=catalog['binding'], shards=published,
                    summary_sha256=file_identity(root / 'summary.json')[1],
                    capabilities=dict(policy_prefix=True, legal_actions=supervised, action_labels=supervised,
                                      training_eligibility=False, training_splits=False))
    atomic_json(path, manifest, compressed=compressed_storage(catalog))
    if hook:
        hook('MANIFEST_PUBLISHED', dict(build_id=catalog['build_id']))
    if catalog['binding'].get('lean_tasks'):
        cleanup_task_state(root, published)
    return manifest


def cleanup_task_state(root, records):
    """The manifest and products are the completed task result; SQL is temporary."""
    root = Path(root)
    for record in records:
        task_id = record['task_id']
        (root/'tasks'/f'part-{task_id:06d}.sqlite.zst').unlink(missing_ok=True)
        (root/'shards'/f'part-{task_id:06d}'/'data.append.zst').unlink(missing_ok=True)
    sync_directory(root/'tasks')


def repair_build(parent, damaged):
    parent, damaged = Path(parent), Path(damaged)
    original = read_json(damaged / 'build.json')
    validate_catalog(original)
    # Serialize allocation across all builds with this semantic identity.
    with fixed_lock(parent / f".repair-{original['semantic_id']}.lock"), \
            fixed_lock(damaged / '.locks' / 'coordinator.lock'), ExitStack() as locks:
        state = None
        try:
            if not json_path(damaged / 'manifest.json').exists():
                from .state import BuildState
                state = BuildState(damaged, original)
                # Freeze old authorization before waiting for any worker lock.
                # Unknown attempt endings remain unknown in the damaged build.
                with state.db:
                    state.stop('REPAIR_REQUIRED', dict(reason='explicit independent repair requested'))
                    state.db.execute("UPDATE pools SET status='RECONCILING',failure_id=COALESCE(failure_id,'pool-'||pool_id) WHERE status!='CLOSED'")
                (damaged / '.stop').touch()
            for task in original['tasks']:
                locks.enter_context(fixed_lock(task_paths(damaged, task['task_id'])[2]))
            for item in original['sources']:
                source = Source(**item)
                from .archive_source import verify_source
                verify_source(source)
            return _allocate_repair(parent, original, state)
        finally:
            if state is not None:
                state.close()


def _allocate_repair(parent, original, state):
    """Caller owns semantic allocation, original coordinator and task locks."""
    used = [original['repair_revision']]
    for path in [*parent.glob('*/build.json'), *parent.glob('*/build.json.zst')]:
        candidate = read_json(path)
        if candidate['semantic_id'] == original['semantic_id']:
            used.append(candidate['repair_revision'])
    catalog = deepcopy(original)
    catalog.update(repair_revision=max(used) + 1, repair_of=original['build_id'])
    catalog['build_id'] = digest([catalog['semantic_id'], catalog['repair_revision']])
    root = initialize_build(parent, catalog)
    if state is not None:
        with state.db:
            state.db.execute('INSERT INTO resolutions(action,reason) VALUES (?,?)',
                ('independent-repair', json.dumps(dict(build_id=catalog['build_id'],
                 repair_revision=catalog['repair_revision'],repair_of=catalog['repair_of'],path=str(root)))))
    return root
