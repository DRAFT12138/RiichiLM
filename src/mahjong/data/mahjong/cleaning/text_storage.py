"""Compressed plain-text products; SQLite contains execution state, never tokens.

Each round is one tagged text block; each game is one independent gzip member. An
ordinary gzip reader streams the whole shard; frame/block offsets permit seeks.
Game files become durable before their task-state COMMIT. Orphans are discarded
on recovery; committed files are verified, never silently replaced.
"""
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import shutil

from .catalog import Source
from .common import (TEXT_STORAGE_VERSION, atomic_json, canonical, connect, digest,
                     file_identity, require, sync_directory, sync_file)
from .encoding import encode_view
from .text_codec import encode_round
from .storage import Shard, STATUSES, check_attempt_fact, meta, set_meta
from .metrics import timed,add,fsync


def product_paths(root, task_id):
    name = f'part-{task_id:06d}'
    return Path(root)/'shards'/name, Path(root)/'.work'/(name+'.products')


def compact_facts(prepared):
    # Validation has already consumed the full IR/provenance. Persist source
    # locations and outcomes, not per-field provenance or every response audit.
    results = prepared.facts.get('canonical_results', [])
    return dict(source_refs=[[w['window_id'],w.get('source_refs',[])] for w in prepared.windows],
                outcomes=[r for r in results if r['kind'] in
                          ('EV_WIN','EV_DRAW_END','EV_ABORT_END','EV_NAGASHI')])


def task_schema(db):
    db.executescript('''
      CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE attempts(attempt_id INTEGER PRIMARY KEY,task_id INTEGER NOT NULL,
        pool_id INTEGER NOT NULL,retry_epoch INTEGER NOT NULL,ending TEXT,error TEXT);
      CREATE TABLE games(source_key TEXT PRIMARY KEY,identity TEXT NOT NULL,
        game_id TEXT,native_id TEXT,rule_ref TEXT,status TEXT NOT NULL
        CHECK(status IN ('success','partial','failed','skipped')),counts TEXT NOT NULL,
        content TEXT NOT NULL);
      CREATE TABLE errors(error_id INTEGER PRIMARY KEY,
        source_key TEXT NOT NULL REFERENCES games(source_key),detail TEXT NOT NULL);
    ''')
    db.commit()


class TextShard(Shard):
    def __init__(self, path, catalog, task, *, create=True):
        require(catalog['binding'].get('task_state') is None,
                'BINDING_MISMATCH','fragment writer cannot open append state')
        self.path, self.catalog, self.task = Path(path), catalog, task
        existed = self.path.exists()
        require(existed or create, 'MISSING_SHARD', str(path))
        self.connection = connect(self.path)
        self.fragments = self.path.with_suffix('.rows')
        try:
            if not existed:
                task_schema(self.connection)
                with self.connection:
                    set_meta(self.connection, 'binding', self.binding)
            require(meta(self.connection, 'binding') == self.binding, 'BINDING_MISMATCH', str(path))
            if meta(self.connection, 'complete') is None:
                self.fragments.mkdir(exist_ok=True)
                committed = self.committed()
                for item in self.fragments.iterdir():
                    if item.name.removesuffix('.txt.gz') not in committed:
                        item.unlink()
                for row in self.connection.execute('SELECT source_key,content FROM games'):
                    item = json.loads(row['content'])
                    require(file_identity(self.fragments/(row['source_key']+'.txt.gz')) ==
                            (item['bytes'], item['sha256']), 'STORAGE_CORRUPT', 'committed game text changed')
        except BaseException:
            self.connection.close()
            raise

    @property
    def binding(self):
        return dict(storage=TEXT_STORAGE_VERSION, build_id=self.catalog['build_id'],
                    task_id=self.task['task_id'], members=self.task['members'])

    @contextmanager
    def game_transaction(self, source, raw_reader):
        require(source.source_key in self.task['members'], 'MEMBERSHIP_MISMATCH', source.source_key)
        require(source.source_key not in self.committed(), 'DUPLICATE_SOURCE', source.source_key)
        require(not self.connection.in_transaction, 'TRANSACTION_ERROR', 'nested game transaction')
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            yield
            raw_reader.verify_eof(source)
            with timed('game_sql_commit_seconds'):
                self.connection.commit()
            add('game_sql_commit_calls')
        except BaseException:
            self.connection.rollback()
            for suffix in ('.txt.gz', '.tmp'):
                (self.fragments/(source.source_key+suffix)).unlink(missing_ok=True)
            raise

    def write_game(self, source, result, *, codec):
        require(self.connection.in_transaction, 'TRANSACTION_ERROR', 'use game_transaction')
        require(result.status in STATUSES, 'TYPE_MISMATCH', 'game terminal status')
        require(result.status != 'skipped' or result.errors, 'MISSING_FIELD', 'skip reason')
        profile = self.catalog['binding']['rules'].get((result.rule_ref or {}).get('ruleset_id'), {})
        stock = (profile['profile']['rule_options']['aka_counts'] if 'profile' in profile else
                 profile.get('offline_options', {}).get('aka_counts'))
        path = self.fragments/(source.source_key+'.txt.gz')
        temporary = self.fragments/(source.source_key+'.tmp')
        rounds, tokens, anchors, seen = [], 0, 0, set()
        raw_size = 0
        with temporary.open('wb') as output, gzip.GzipFile(filename='', mode='wb',fileobj=output,
                                                         compresslevel=9,mtime=0) as stream:
            for prepared in result.rounds:
                require(type(prepared.round_id) is int and prepared.round_id >= 0 and prepared.round_id not in seen,
                        'INVALID_REFERENCE', 'round identity')
                seen.add(prepared.round_id)
                require(prepared.aka_counts == stock, 'BINDING_MISMATCH', 'prepared tile stock')
                views = []
                for view in prepared.views:
                    with timed('encoding_seconds'):
                        encoded = encode_view(view, prepared, codec=codec)
                    views.append(dict(viewer=encoded.viewer, valid_through_event_id=encoded.valid_through_event_id,
                                      invalid_reason=encoded.invalid_reason, arrays=encoded.arrays))
                    tokens += sum(len(encoded.arrays[k]) for k in ('header','history','hand_blocks'))
                    anchors += encoded.anchor_count
                require(len({v['viewer'] for v in views}) == len(views), 'INVALID_REFERENCE', 'duplicate viewer')
                record = dict(source_key=source.source_key, round_id=prepared.round_id,
                              event_ids=prepared.event_ids, views=views, facts=compact_facts(prepared))
                with timed('encoding_seconds'):
                    raw = encode_round(record)
                with timed('compression_seconds'):
                    stream.write(raw)
                rounds.append(dict(round_id=prepared.round_id, offset=raw_size, bytes=len(raw),
                                   views=[[v['viewer'],v['valid_through_event_id'],
                                        len(v['arrays']['anchors'])//5] for v in views]))
                raw_size += len(raw)
                require(raw_size<=64*1024*1024, 'RESOURCE_LIMIT', 'text game exceeds64MiB decoded limit')
        sync_file(temporary)
        os.replace(temporary, path)
        sync_directory(path.parent)
        size, sha = file_identity(path)
        content = dict(bytes=size, sha256=sha, raw_bytes=raw_size, rounds=rounds, anchors=anchors, tokens=tokens)
        self.connection.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?)',
            (source.source_key, canonical(source.identity()).decode(), result.game_id, result.native_id,
             canonical(result.rule_ref).decode(), result.status, canonical(result.counts).decode(),
             canonical(content).decode()))
        for i, error in enumerate(result.errors):
            require(all(k in error for k in ('code','stage','reason')), 'MISSING_FIELD', 'error classification')
            self.connection.execute('INSERT INTO errors(source_key,detail) VALUES (?,?)',
                (source.source_key, canonical(dict(error, primary=i==0)).decode()))

    def complete(self, *, attempt_id=None, codec=None):
        games = checked_games(self.connection, self.catalog, self.task)
        root = self.path.parent.parent
        final, work = product_paths(root, self.task['task_id'])
        require(not final.exists(), 'ARTIFACT_CONFLICT', 'published text already exists')
        work.mkdir(exist_ok=True)
        index = dict(binding=self.binding, games=[], rounds=[])
        with (work/'data.txt.gz').open('wb') as stream:
            for row in games:
                content = json.loads(row['content'])
                fragment = self.fragments/(row['source_key']+'.txt.gz')
                require(file_identity(fragment) == (content['bytes'],content['sha256']),
                        'STORAGE_CORRUPT', 'game fragment changed')
                start = stream.tell()
                with fragment.open('rb') as source:
                    shutil.copyfileobj(source, stream)
                index['games'].append(dict({k:json.loads(row[k]) if k in ('rule_ref','counts') else row[k]
                                       for k in ('source_key','game_id','native_id','rule_ref','status','counts')},
                                       offset=start,bytes=content['bytes'],raw_bytes=content['raw_bytes']))
                for item in content['rounds']:
                    index['rounds'].append(dict(item, source_key=row['source_key']))
            stream.flush()
            fsync(stream.fileno(),category='merged_data_fsync')
        atomic_json(work/'index.json', index)
        info = information(games, index)
        info['products'] = [dict(name=name, bytes=file_identity(work/name)[0], sha256=file_identity(work/name)[1])
                            for name in ('data.txt.gz','index.json')]
        with self.connection:
            set_meta(self.connection, 'complete', info)
            if attempt_id is not None:
                self.ended(attempt_id, 'COMPLETED')
        return info


def checked_games(db, catalog, task):
    require(meta(db, 'binding') == dict(storage=TEXT_STORAGE_VERSION, build_id=catalog['build_id'],
            task_id=task['task_id'], members=task['members']), 'BINDING_MISMATCH', 'text task state')
    require(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'STORAGE_CORRUPT', 'task SQLite')
    require(not db.execute('PRAGMA foreign_key_check').fetchall(), 'MEMBERSHIP_MISMATCH', 'orphan task error')
    games = db.execute('SELECT * FROM games ORDER BY source_key').fetchall()
    require(len(games) == len(task['members']) and {g['source_key'] for g in games} == set(task['members']),
            'MEMBERSHIP_MISMATCH', 'text task terminal members')
    sources = {s['source_key']:Source(**s) for s in catalog['sources']}
    for row in games:
        require(json.loads(row['identity']) == sources[row['source_key']].identity() and row['status'] in STATUSES,
                'MEMBERSHIP_MISMATCH', 'text source identity/status')
        require(row['status'] != 'skipped' or db.execute('SELECT 1 FROM errors WHERE source_key=?',
                (row['source_key'],)).fetchone(), 'MISSING_FIELD', 'skip reason')
    return games


def information(games, index):
    return dict(logical_sha256=digest(index), games=len(games), rounds=len(index['rounds']),
                views=sum(len(r['views']) for r in index['rounds']),
                anchors=sum(json.loads(g['content'])['anchors'] for g in games),
                tokens=sum(json.loads(g['content'])['tokens'] for g in games),
                statuses={s:sum(g['status']==s for g in games) for s in STATUSES})


def locate_products(root, task_id, expected):
    final, work = product_paths(root, task_id)
    found = []
    for directory in (final, work):
        if directory.exists():
            for item in expected:
                p = directory/item['name']
                require(p.is_file() and file_identity(p) == (item['bytes'],item['sha256']),
                        'STORAGE_CORRUPT', 'text product missing/changed: '+str(p))
            found.append(directory)
    require(found, 'MISSING_SHARD', 'text products')
    return found[0]


def inspect_text_task(path, catalog, task, *, codec=None, require_complete=True):
    if catalog['binding'].get('task_state') == 'append-state/1':
        from .append_storage import inspect_append
        return inspect_append(path, catalog, task, complete=require_complete, codec=codec)
    db = connect(path, readonly=True)
    try:
        games = checked_games(db, catalog, task)
        marker = meta(db, 'complete')
        require(marker is not None, 'INCOMPLETE_SHARD', 'text completion marker')
        directory = locate_products(Path(path).parent.parent, task['task_id'], marker['products'])
        index = json.loads((directory/'index.json').read_bytes())
        require(index['binding'] == meta(db,'binding'), 'BINDING_MISMATCH', 'text index')
        expected = information(games, index)
        require(dict(marker, products=None) == dict(expected, products=None),
                'STORAGE_CORRUPT', 'text completion facts')
        require([{k:g[k] for k in ('source_key','game_id','native_id','rule_ref','status','counts')}
                 for g in index['games']] == [{k:json.loads(g[k]) if k in ('rule_ref','counts') else g[k]
                                   for k in ('source_key','game_id','native_id','rule_ref','status','counts')}
                                  for g in games], 'MEMBERSHIP_MISMATCH', 'text index game facts')
        for row in db.execute('SELECT * FROM attempts'):
            check_attempt_fact(row)
            require(row['task_id']==task['task_id'], 'ATTEMPT_CONFLICT', 'text task identity')
        return marker
    finally:
        db.close()


def inspect_unfinished_text(path, catalog, task):
    """Check committed files before dispatch, so corruption follows repair policy."""
    if catalog['binding'].get('task_state') == 'append-state/1':
        from .append_storage import inspect_append
        return inspect_append(path, catalog, task, complete=False)
    db=connect(path,readonly=True)
    try:
        require(meta(db,'binding')==dict(storage=TEXT_STORAGE_VERSION,build_id=catalog['build_id'],
                task_id=task['task_id'],members=task['members']), 'BINDING_MISMATCH','unfinished text task')
        require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','STORAGE_CORRUPT','unfinished task state')
        require(not db.execute('PRAGMA foreign_key_check').fetchall(),'MEMBERSHIP_MISMATCH','orphan task error')
        sources={s['source_key']:Source(**s) for s in catalog['sources']}
        for row in db.execute('SELECT source_key,identity,content FROM games'):
            key=row['source_key']
            require(key in task['members'] and json.loads(row['identity'])==sources[key].identity(),
                    'MEMBERSHIP_MISMATCH','unfinished source identity')
            content=json.loads(row['content'])
            file=Path(path).with_suffix('.rows')/(key+'.txt.gz')
            require(file.is_file() and file_identity(file)==(content['bytes'],content['sha256']),
                    'STORAGE_CORRUPT','committed game text missing/changed')
    finally:
        db.close()


def publish_products(root, task, info, *, verified=False):
    final, work = product_paths(root, task['task_id'])
    actual = (final if final.exists() else work) if verified else locate_products(root, task['task_id'], info['products'])
    if actual == work:
        require(not final.exists(), 'ARTIFACT_CONFLICT', 'text product target exists')
        os.rename(work, final)
        sync_directory(final.parent)
        sync_directory(work.parent)


def discard_unpublished_products(root, task_id):
    for p in (*product_paths(root, task_id), Path(root)/'.work'/f'part-{task_id:06d}.rows'):
        if p.exists():
            target = Path(root)/'.quarantine'/(p.name+'-'+str(__import__('time').time_ns()))
            target.parent.mkdir(exist_ok=True)
            os.rename(p, target)
            sync_directory(p.parent)
            sync_directory(target.parent)
