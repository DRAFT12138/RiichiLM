"""Publish one training corpus from verified text/3 builds; never delete inputs."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from compression import zstd
from multiprocessing import get_context
import json
import os
from pathlib import Path

from .corpus import CORPUS_VERSION, corpus_identity
from .cleaning.catalog import codec_binding
from .cleaning.common import (
    atomic_bytes, atomic_json, canonical, digest, file_identity, json_path,
    read_json, require, sync_directory, sync_file,
)


def _pack_partition(job):
    source, destination = Path(job['source']), Path(job['destination'])
    destination.mkdir(parents=True, exist_ok=False)
    products = {p['name']: p for p in job['record']['products']}
    body, index_info = products['data.blocks.zst'], products['index.json.zst']
    require(file_identity(source/body['name']) == (body['bytes'], body['sha256']),
            'STORAGE_CORRUPT', 'source body SHA')
    require(file_identity(source/index_info['name']) == (index_info['bytes'], index_info['sha256']),
            'STORAGE_CORRUPT', 'source index SHA')
    index = read_json(source/index_info['name'])
    binding = index['binding']
    games = index['games']; members = [g['source_key'] for g in games]
    require(binding['storage'] == 'mjai-clean-text/3' and binding['build_id'] == job['old_build_id'] and
            binding['task_id'] == job['record']['task_id'] and
            len(members) == len(set(members)) == job['record']['games'] and
            set(members) == set(binding['members']), 'MEMBERSHIP_MISMATCH', 'source index')
    counts = Counter(games=len(games), rounds=len(index['rounds']),
                     views=sum(len(r['views']) for r in index['rounds']),
                     anchors=sum(v[2] for r in index['rounds'] for v in r['views']),
                     tokens=job['record']['tokens'])
    for k in ('games', 'rounds', 'views', 'anchors'):
        require(counts[k] == job['record'][k], 'STORAGE_CORRUPT', 'source index counts')
    rows = []
    for game in games:
        counts.update({k: v for k, v in game.get('counts', {}).items() if k.startswith('policy_')})
        rows.append([game['source_key'], game['game_id'], game['native_id'], job['task_id']])
        # Scheduling, rejection and diagnostic counters are not reader data.
        game.pop('counts', None); game.pop('status', None)
    partition_id = digest([body['sha256'], index_info['sha256']])
    index['binding'] = {'storage': 'mjai-clean-text/3', 'partition_id': partition_id}
    # Hard links retain payload bytes while the old trees still exist, with no
    # second 77 GiB copy. Publication and validation precede all source deletion.
    os.link(source/body['name'], destination/body['name'])
    atomic_bytes(destination/index_info['name'], zstd.compress(canonical(index), level=6))
    size, sha = file_identity(destination/index_info['name'])
    sync_directory(destination)
    record = dict(task_id=job['task_id'], group=job['group'], partition_id=partition_id,
        members_digest=digest(members), products=[body, dict(name=index_info['name'], bytes=size, sha256=sha)],
        **{k: counts[k] for k in ('games', 'rounds', 'views', 'anchors')})
    inventory = b''.join(canonical(row)+b'\n' for row in rows)
    return dict(record=record, inventory=inventory, members=members, counts=dict(counts), build=job['build'])


def pack_corpus(inputs, output, *, workers=4, target_tokens=None, progress=None):
    """Inputs are build paths, or audited {build, manifest_sha256, counts} rows."""
    output = Path(output).resolve(); stage = output.with_name('.'+output.name+'-packing')
    require(not output.exists() and not stage.exists(), 'OUTPUT_EXISTS', 'fresh corpus directory required')
    require(type(workers) is int and workers > 0, 'TYPE_MISMATCH', 'positive packing workers')
    stage.mkdir(parents=True); (stage/'shards').mkdir()
    jobs, groups, rules, inputs_by_group, expected = [], {}, {}, {}, {}
    for item in inputs:
        item = {'build': str(item)} if isinstance(item, (str, Path)) else item
        build = Path(item['build']).resolve()
        manifest_path = json_path(build/'manifest.json')
        manifest = read_json(manifest_path); binding = manifest['binding']
        if 'manifest_sha256' in item:
            require(file_identity(manifest_path)[1] == item['manifest_sha256'],
                    'STORAGE_CORRUPT', 'input manifest changed')
        require(binding['storage'] == 'mjai-clean-text/3' and binding.get('compression') == 'zstd-blocks/1' and
                binding.get('backend') == 'rust' and binding['codec'] == codec_binding() and
                manifest['capabilities'].get('legal_actions') and manifest['capabilities'].get('action_labels'),
                'UNSUPPORTED_STORAGE', 'labelled Rust block text/3 required')
        require(file_identity(build/'summary.json')[1] == manifest['summary_sha256'],
                'STORAGE_CORRUPT', 'input summary SHA')
        summary = read_json(build/'summary.json')
        expected[str(build)] = summary['counts']
        if 'counts' in item:
            require(summary['counts'] == item['counts'], 'STORAGE_CORRUPT', 'audited input counts')
        require(len(summary['groups']) == 1, 'UNSUPPORTED_SOURCE', 'one source group per input build')
        source_group = next(iter(summary['groups']))
        platform, shard = source_group.split('/', 1)
        # Source revision and rule facts survive; TAR locators, batch receipts,
        # SQL, old filesystem paths and worker histories are discarded.
        catalog = read_json(build/'build.json')
        revisions = {s['revision'] for s in catalog['sources']}
        require(len(revisions) == 1, 'BINDING_MISMATCH', 'one frozen source revision')
        group = dict(platform=platform, shard=shard, revision=next(iter(revisions)),
                     rules=binding['rules'], implementation_digest=digest(binding['implementation']))
        group_id = digest(group)
        groups[group_id] = group
        inputs_by_group.setdefault(group_id, []).append(binding['input_digest'])
        for rule_id, entry in binding['rules'].items():
            profile = {'profile': entry['profile']}
            require(rule_id not in rules or rules[rule_id] == profile,
                    'BINDING_MISMATCH', 'conflicting training rule profile')
            rules[rule_id] = profile
        for record in manifest['shards']:
            task_id = len(jobs)
            jobs.append(dict(build=str(build), old_build_id=manifest['build_id'], record=record,
                source=str(build/'shards'/f"part-{record['task_id']:06d}"),
                destination=str(stage/'shards'/f'part-{task_id:06d}'), task_id=task_id, group=group_id))
    require(jobs, 'MISSING_INPUT', 'nonempty published corpus')
    records, counts, by_build, seen = [], Counter(), {}, set()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork')) as pool, \
            zstd.open(stage/'games.jsonl.zst', 'wb', level=6) as inventory:
        for n, result in enumerate(pool.map(_pack_partition, jobs, chunksize=1, buffersize=workers*2), 1):
            require(not seen.intersection(result['members']), 'DUPLICATE_SOURCE', 'input builds overlap')
            seen.update(result['members'])
            inventory.write(result['inventory'])
            records.append(result['record']); counts.update(result['counts'])
            by_build.setdefault(result['build'], Counter()).update(result['counts'])
            if progress and (n % 250 == 0 or n == len(jobs)):
                progress(dict(packed=n, total=len(jobs), games=counts['games']))
    for build, total in expected.items():
        require(by_build[build] == Counter(total), 'STORAGE_CORRUPT', 'packed build totals differ')
    require(len(seen) == counts['games'], 'STORAGE_CORRUPT', 'corpus unique game count')
    for key, group in groups.items():group['input_digest'] = digest(inputs_by_group[key])
    size, sha = file_identity(stage/'games.jsonl.zst')
    manifest = dict(format=CORPUS_VERSION,
        binding=dict(storage='mjai-clean-text/3', compression='zstd-blocks/1', backend='rust',
                     codec=codec_binding(), rules=rules),
        source_groups=groups, shards=records, games=dict(bytes=size, sha256=sha), counts=dict(counts),
        capabilities=dict(legal_actions=True, action_labels=True, policy_prefix=True))
    if target_tokens is not None:manifest['target_tokens'] = target_tokens
    manifest['build_id'] = corpus_identity(manifest)
    sync_file(stage/'games.jsonl.zst'); sync_directory(stage/'shards')
    atomic_json(stage/'dataset.json', manifest)
    stage.rename(output); sync_directory(output.parent)
    return manifest
