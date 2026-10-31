"""One append file per task, with fsync-before-SQL group checkpoints."""
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import zlib

from .common import (CleaningError, atomic_json, canonical, digest, file_identity, require,
                     connect, sync_directory)
from .encoding import encode_view, EncodedView, check_layout
from .storage import Shard, STATUSES, meta, set_meta, check_attempt_fact
from .text_codec import encode_round, decode_round
from .text_storage import (product_paths, compact_facts, task_schema,
                           checked_games, information)
from .metrics import add,timed,fsync
from .compression import compressed_storage, product_names, ZstdWriter, decompress_frame, BLOCK_STORAGE
from .common import read_json

STATE_VERSION = 'append-state/1'
DEFAULT_COMMIT = dict(commit_max_games=8, commit_max_bytes=8*1024*1024,
                      commit_max_seconds=2)


class HashWriter:
    def __init__(self, stream, budget=None):
        self.stream, self.budget = stream, budget
        self.hasher = hashlib.sha256()
        self.count = 0

    def write(self, data):
        if self.budget is not None:
            self.budget.spend(len(data))
        written = self.stream.write(data)
        require(written == len(data), 'PERSISTENCE_FAILURE', 'short product write')
        self.hasher.update(data)
        self.count += written
        add('product_written_bytes',written)
        return written

    def flush(self):
        self.stream.flush()


class AppendTextShard(Shard):
    """Pending groups retain metadata only, never Replay objects or token arrays."""
    def __init__(self, path, catalog, task, *, create=True, execution=None, budget=None,
                 hook=None):
        from .text_storage import TextShard
        self.path, self.catalog, self.task = Path(path), catalog, task
        self.policy = dict(DEFAULT_COMMIT, **{k:v for k,v in (execution or {}).items()
                                            if k in DEFAULT_COMMIT})
        require(all(type(v) in (int, float) and v > 0 for v in self.policy.values()),
                'TYPE_MISMATCH', 'commit thresholds must be positive')
        self.native_gzip_level=(execution or {}).get("native_gzip_level",1)
        require(type(self.native_gzip_level) is int and 1<=self.native_gzip_level<=9,"INVALID_INPUT","native gzip level1..9")
        self.budget, self.hook = budget, hook
        self.pending = []
        self.pending_started = None
        self.current = None
        self.stream = None
        require(catalog['binding'].get('task_state') == STATE_VERSION,
                'BINDING_MISMATCH', 'append protocol required')
        existed = self.path.exists()
        require(existed or create, 'MISSING_SHARD', str(path))
        self.connection = connect(self.path)
        try:
            if not existed:
                task_schema(self.connection)
                self.connection.execute('CREATE TABLE groups(sequence INTEGER PRIMARY KEY,'
                    'start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,members TEXT NOT NULL,sha256 TEXT NOT NULL)')
                with self.connection:
                    set_meta(self.connection, 'binding', self.binding)
                    set_meta(self.connection, 'task_state', STATE_VERSION)
                    set_meta(self.connection, 'committed_offset', 0)
                    set_meta(self.connection, 'group_sequence', 0)
            if budget is not None:
                page_size=self.connection.execute('PRAGMA page_size').fetchone()[0]
                # Reserve space for both the database and its rollback journal.
                per_task=max(65536,budget.metadata_reserve_bytes//(2*max(1,(execution or {}).get('workers',1))))
                self.connection.execute('PRAGMA max_page_count='+str(max(1,per_task//page_size)))
            require(meta(self.connection, 'binding') == self.binding and
                    meta(self.connection, 'task_state') == STATE_VERSION,
                    'BINDING_MISMATCH', 'append task state')
            if meta(self.connection, 'complete') is None:
                directory = product_paths(self.path.parent.parent, task['task_id'])[1]
                directory.mkdir(exist_ok=True)
                sync_directory(directory.parent)
                self.product = directory/('data.append.zst' if catalog['binding'].get('compression') == BLOCK_STORAGE
                                          else product_names(catalog)[0])
                new_product = not self.product.exists()
                self.stream = self.product.open('w+b' if new_product else 'r+b', buffering=0)
                if new_product:
                    sync_directory(directory)
                # Inspect committed extents before removing any uncommitted tail.
                verify_append(self.connection, self.product, complete=False)
                offset = meta(self.connection, 'committed_offset')
                if self.product.stat().st_size > offset:
                    self.stream.truncate(offset)
                    fsync(self.stream.fileno(),category='recovery_fsync')
                self.stream.seek(offset)
                if budget is not None:
                    from .resources import WriterBudget
                    self.budget = WriterBudget(budget, self.product, self.stream)
        except BaseException:
            self.close()
            raise

    @property
    def binding(self):
        from .text_storage import TextShard
        return TextShard.binding.fget(self)

    def point(self, name, item=None):
        if self.hook:
            self.hook(name, item)

    @contextmanager
    def game_transaction(self, source, raw_reader):
        require(source.source_key in self.task['members'] and
                source.source_key not in self.committed() and
                source.source_key not in {r[0][0] for r in self.pending},
                'MEMBERSHIP_MISMATCH', source.source_key)
        require(self.current is None, 'TRANSACTION_ERROR', 'nested append game')
        start = self.stream.tell()
        try:
            yield
            raw_reader.verify_eof(source)
            require(self.current is not None, 'TRANSACTION_ERROR', 'no game written')
            self.pending.append(self.current)
            self.current = None
            if self.pending_started is None:
                self.pending_started = time.monotonic()
            if (len(self.pending) >= self.policy['commit_max_games'] or
                self.stream.tell()-meta(self.connection, 'committed_offset') >= self.policy['commit_max_bytes'] or
                time.monotonic()-self.pending_started >= self.policy['commit_max_seconds']):
                self.commit_group()
        except BaseException as error:
            self.current = None
            if isinstance(error,CleaningError) and error.code in ('WORK_LIMIT','SPACE_LOW'):
                # The active member is not validated; earlier pending members are.
                # Preserve their progress using the independent metadata reserve.
                self.stream.truncate(start)
                self.stream.seek(start)
                if self.budget is not None:
                    self.budget.close();self.budget.credit=0
                self.commit_group()
            # SQL may have committed even if a notification/fsync wrapper failed.
            # Never truncate here: only reopening under the task lock can decide.
            raise

    def write_game(self, source, result, *, codec):
        require(result.status in STATUSES and (result.status != 'skipped' or result.errors),
                'TYPE_MISMATCH', 'game terminal status/reason')
        profile = self.catalog['binding']['rules'].get((result.rule_ref or {}).get('ruleset_id'), {})
        stock = (profile['profile']['rule_options']['aka_counts'] if 'profile' in profile else
                 profile.get('offline_options', {}).get('aka_counts'))
        start = self.stream.tell()
        output = HashWriter(self.stream, self.budget)
        rounds, seen, raw_size, tokens, anchors = [], set(), 0, 0, 0
        raw_hash = hashlib.sha256()
        with (ZstdWriter(output, level=1 if self.catalog['binding'].get('compression') == BLOCK_STORAGE else 9) if compressed_storage(self.catalog) else
              MeasuredGzip(filename='', mode='wb', fileobj=output,
                           compresslevel=9, mtime=0)) as compressed:
            for prepared in result.rounds:
                require(type(prepared.round_id) is int and prepared.round_id >= 0 and
                        prepared.round_id not in seen, 'INVALID_REFERENCE', 'round identity')
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
                require(len({v['viewer'] for v in views}) == len(views),
                        'INVALID_REFERENCE', 'duplicate viewer')
                with timed('encoding_seconds'):
                    raw = encode_round(dict(source_key=source.source_key, round_id=prepared.round_id,
                        event_ids=prepared.event_ids, views=views, facts=compact_facts(prepared)))
                require(raw_size+len(raw) <= 64*1024*1024, 'RESOURCE_LIMIT', 'text game exceeds64MiB')
                compressed.write(raw)
                raw_hash.update(raw)
                rounds.append(dict(round_id=prepared.round_id, offset=raw_size, bytes=len(raw),
                    views=[[v['viewer'],v['valid_through_event_id'],len(v['arrays']['anchors'])//5] for v in views]))
                raw_size += len(raw)
                self.point('GZIP_WRITTEN', dict(source_key=source.source_key))
        content = dict(offset=start, bytes=output.count, sha256=output.hasher.hexdigest(),
                       raw_bytes=raw_size, rounds=rounds, anchors=anchors, tokens=tokens)
        if self.catalog['binding'].get('compression') == BLOCK_STORAGE:
            content['raw_sha256'] = raw_hash.hexdigest()
        row = (source.source_key, canonical(source.identity()).decode(), result.game_id,
               result.native_id, canonical(result.rule_ref).decode(), result.status,
               canonical(result.counts).decode(), canonical(content).decode())
        errors = []
        for i, error in enumerate(result.errors):
            require(all(k in error for k in ('code','stage','reason')), 'MISSING_FIELD', 'error classification')
            errors.append((source.source_key, canonical(dict(error, primary=i == 0)).decode()))
        self.current = row, errors

    def write_encoded_game(self, source, result):
        """Append Rust's checked text/2 body without reconstructing token objects.

        Publication still parses persisted text and validates every layout.
        """
        from .native_backend import NativeGameResult
        require(type(result) is NativeGameResult, 'TYPE_MISMATCH', 'native result required')
        require(result.status in STATUSES and (result.status != 'skipped' or result.errors),
                'TYPE_MISMATCH', 'game terminal status/reason')
        require(type(result.body) is bytes and len(result.body) <= 64*1024*1024,
                'RESOURCE_LIMIT', 'native body limit')
        end, seen = 0, set()
        for part in result.rounds:
            require(type(part['round_id']) is int and part['round_id'] >= 0 and
                    part['round_id'] not in seen and part['offset'] == end and
                    type(part['bytes']) is int and part['bytes'] > 0,
                    'INVALID_REFERENCE', 'native round extent/identity')
            seen.add(part['round_id']); end += part['bytes']
            require(end <= len(result.body), 'INVALID_REFERENCE', 'native round boundary')
        require(end == len(result.body), 'INVALID_REFERENCE', 'native round coverage')
        start = self.stream.tell()
        output = HashWriter(self.stream, self.budget)
        with (ZstdWriter(output, level=1 if self.catalog['binding'].get('compression') == BLOCK_STORAGE else 9) if compressed_storage(self.catalog) else
              MeasuredGzip(filename='', mode='wb', fileobj=output, compresslevel=self.native_gzip_level, mtime=0)) as compressed:
            compressed.write(result.body)
            self.point('GZIP_WRITTEN', dict(source_key=source.source_key))
        content = dict(offset=start, bytes=output.count, sha256=output.hasher.hexdigest(),
            raw_bytes=len(result.body), rounds=result.rounds, anchors=result.anchors, tokens=result.tokens)
        if self.catalog['binding'].get('compression') == BLOCK_STORAGE:
            content['raw_sha256'] = hashlib.sha256(result.body).hexdigest()
        row = (source.source_key, canonical(source.identity()).decode(), result.game_id,
               result.native_id, canonical(result.rule_ref).decode(), result.status,
               canonical(result.counts).decode(), canonical(content).decode())
        errors = []
        for i, error in enumerate(result.errors):
            require(all(k in error for k in ('code','stage','reason')), 'MISSING_FIELD', 'error classification')
            errors.append((source.source_key, canonical(dict(error, primary=i == 0)).decode()))
        self.current = row, errors

    def commit_group(self):
        if not self.pending:
            return
        self.stream.flush()
        fsync(self.stream.fileno(),category='group_data_fsync')
        self.point('GROUP_SYNCED', dict(games=len(self.pending)))
        with group_transaction(self.connection):
            self.connection.executemany('INSERT INTO games VALUES (?,?,?,?,?,?,?,?)',
                                        [r for r,e in self.pending])
            self.connection.executemany('INSERT INTO errors(source_key,detail) VALUES (?,?)',
                                        [e for r,errors in self.pending for e in errors])
            start=meta(self.connection,'committed_offset')
            sequence=meta(self.connection,'group_sequence')+1
            members=[r[0] for r,e in self.pending]
            group_sha=digest([json.loads(r[-1]) for r,e in self.pending])
            self.connection.execute('INSERT INTO groups VALUES (?,?,?,?,?)',
                (sequence,start,self.stream.tell(),canonical(members).decode(),group_sha))
            set_meta(self.connection, 'committed_offset', self.stream.tell())
            set_meta(self.connection, 'group_sequence', sequence)
            set_meta(self.connection, 'last_group', dict(
                members=[r[0] for r,e in self.pending],
                sha256=digest([json.loads(r[-1]) for r,e in self.pending])))
        self.pending.clear()
        add('committed_groups')
        self.pending_started = None
        self.point('GROUP_COMMITTED', dict(offset=self.stream.tell()))

    def close(self):
        if self.budget is not None and hasattr(self.budget, 'close'):
            self.budget.close()
        if self.stream is not None:
            self.stream.close()
        if getattr(self, 'connection', None) is not None:
            self.connection.close()

    def complete(self, *, attempt_id=None, codec=None):
        self.commit_group()
        if self.budget is not None:
            self.budget.close();self.budget.credit=0
        games = checked_games(self.connection, self.catalog, self.task)
        directory = self.product.parent
        index = make_index(self.binding, games)
        with timed('product_verification_seconds'):
            data_info = verify_append(self.connection, self.product, complete=True, codec=codec,
                native=self.catalog['binding'].get('backend') == 'rust')
        add('product_verification_calls')
        if self.catalog['binding'].get('compression') == BLOCK_STORAGE:
            from .block_storage import repack_blocks
            index, data_info = repack_blocks(self.product, games, index,
                self.budget.ledger if self.budget is not None else None)
        from .resources import write_index
        body_name, index_name = product_names(self.catalog)
        write_index(directory/index_name, index,
                    self.budget.ledger if self.budget is not None else None,
                    compressed=compressed_storage(self.catalog))
        size, sha = file_identity(directory/index_name)
        info = information(games, index)
        info['products'] = [dict(name=body_name, **data_info),
                            dict(name=index_name, bytes=size, sha256=sha)]
        with self.connection:
            set_meta(self.connection, 'complete', info)
            if attempt_id is not None:
                self.ended(attempt_id, 'COMPLETED')
        if self.catalog['binding'].get('compression') == BLOCK_STORAGE:
            self.product.unlink()
            from .common import sync_directory
            sync_directory(directory)
        return info


def make_index(binding, games):
    index = dict(binding=binding, games=[], rounds=[])
    for row in games:
        content = json.loads(row['content'])
        index['games'].append(dict({k:json.loads(row[k]) if k in ('rule_ref','counts') else row[k]
            for k in ('source_key','game_id','native_id','rule_ref','status','counts')},
            **{k:content[k] for k in ('offset','bytes','raw_bytes')}))
        index['rounds'].extend(dict(r, source_key=row['source_key']) for r in content['rounds'])
    return index


def verify_append(db, path, *, complete, codec=None, native=False):
    """One sequential read for physical SHA, per-member SHA, gzip and layouts."""
    require(meta(db, 'task_state') == STATE_VERSION, 'BINDING_MISMATCH', 'append state version')
    boundary = meta(db, 'committed_offset')
    require(type(boundary) is int and boundary >= 0 and path.exists() and
            path.stat().st_size >= boundary, 'STORAGE_CORRUPT', 'product shorter than committed boundary')
    full, offset = hashlib.sha256(), 0
    contents={r['source_key']:json.loads(r['content']) for r in db.execute('SELECT source_key,content FROM games')}
    group_end=0;group_members=[];sequence=0
    for group in db.execute('SELECT * FROM groups ORDER BY sequence'):
        members=json.loads(group['members']);sequence+=1
        require(group['sequence']==sequence and group['start_offset']==group_end and members and
                all(k in contents for k in members) and group['sha256']==digest([contents[k] for k in members]),
                'STORAGE_CORRUPT','group checkpoint facts')
        group_end=group['end_offset'];group_members.extend(members)
    require(group_end==boundary and sequence==meta(db,'group_sequence') and
            group_members==sorted(contents), 'STORAGE_CORRUPT','group coverage')
    with path.open('rb') as stream:
        for row in db.execute('SELECT source_key,content FROM games ORDER BY source_key'):
            item = json.loads(row['content'])
            require(type(item['raw_bytes']) is int and 0<=item['raw_bytes']<=64*1024*1024 and
                    item['offset'] == offset and item['bytes'] > 0 and
                    offset+item['bytes'] <= boundary, 'STORAGE_CORRUPT', 'committed extent')
            frame, chunks = hashlib.sha256(), []
            remaining = item['bytes']
            while remaining:
                chunk = stream.read(min(1024*1024, remaining))
                require(bool(chunk), 'STORAGE_CORRUPT', 'committed extent truncated')
                remaining -= len(chunk)
                full.update(chunk); frame.update(chunk)
                add('product_verification_read_bytes',len(chunk))
                chunks.append(chunk)
            require(frame.hexdigest() == item['sha256'], 'STORAGE_CORRUPT', 'committed member changed')
            decoded = decompress_frame(b''.join(chunks), zstandard=path.suffix == '.zst',
                max_bytes=item['raw_bytes'], expected_size=item['raw_bytes'])
            end = 0
            for part in item['rounds']:
                require(part['offset'] == end and part['bytes'] > 0,
                        'STORAGE_CORRUPT', 'round extent')
                end += part['bytes']
                if complete:
                    raw_round = bytes(decoded[part['offset']:end])
                    if native:
                        from .native_backend import native_module
                        try:
                            native_module().verify_text_round(raw_round,row['source_key'],
                                part['round_id'],canonical(part['views']).decode())
                        except ValueError as error:
                            require(False, 'STORAGE_CORRUPT', str(error))
                    else:
                        record = decode_round(raw_round)
                        require(record['source_key'] == row['source_key'] and
                                record['round_id'] == part['round_id'], 'STORAGE_CORRUPT', 'round identity')
                        require([[v['viewer'],v['valid_through_event_id'],len(v['arrays']['anchors'])//5]
                                 for v in record['views']]==part['views'], 'STORAGE_CORRUPT','round/view index')
                        for view in record['views']:
                            # Validate persisted structure against the written-byte SHA.
                            check_layout(EncodedView(view['viewer'], view['valid_through_event_id'],
                                view['invalid_reason'], view['arrays']), record['event_ids'])
            require(end == item['raw_bytes'], 'STORAGE_CORRUPT', 'round coverage')
            offset += item['bytes']
    require(offset == boundary and (not complete or path.stat().st_size == boundary),
            'STORAGE_CORRUPT', 'committed coverage/tail')
    return dict(bytes=boundary, sha256=full.hexdigest())


def inspect_append(path, catalog, task, *, complete=True, codec=None):
    db = connect(path, readonly=True)
    try:
        from .text_storage import TextShard
        binding = dict(storage=catalog['binding']['storage'], build_id=catalog['build_id'],
                       task_id=task['task_id'], members=task['members'])
        require(meta(db, 'binding') == binding, 'BINDING_MISMATCH', 'append binding')
        require(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'STORAGE_CORRUPT', 'task SQL')
        for row in db.execute('SELECT * FROM attempts'):
            check_attempt_fact(row)
        final, work = product_paths(Path(path).parent.parent, task['task_id'])
        directory = final if final.exists() else work
        body_name, index_name = product_names(catalog)
        marker = meta(db, 'complete')
        if not complete:
            verify_append(db, directory/('data.append.zst' if catalog['binding'].get('compression') == BLOCK_STORAGE else body_name), complete=False)
            return None
        require(marker is not None, 'INCOMPLETE_SHARD', str(path))
        games = checked_games(db, catalog, task)
        if catalog['binding'].get('compression') == BLOCK_STORAGE:
            from .block_storage import inspect_blocks
            return inspect_blocks(db, directory, catalog, games, marker, codec=codec)
        actual = verify_append(db, directory/body_name, complete=True, codec=codec,
            native=catalog['binding'].get('backend') == 'rust')
        index = read_json(directory/index_name)
        require(index == make_index(binding, games) and
                {p['name']:(p['bytes'],p['sha256']) for p in marker['products']} ==
                {body_name:(actual['bytes'],actual['sha256']),
                 index_name:file_identity(directory/index_name)} and
                dict(marker, products=None) == dict(information(games,index), products=None),
                'STORAGE_CORRUPT', 'append products/index')
        return marker
    finally:
        db.close()


class MeasuredGzip(gzip.GzipFile):
    def write(self,data):
        with timed('compression_seconds'):
            return super().write(data)

    def close(self):
        with timed('compression_seconds'):
            return super().close()


@contextmanager
def group_transaction(db):
    try:
        yield
        with timed('group_sql_commit_seconds'):
            db.commit()
        add('group_sql_commit_calls')
    except BaseException:
        db.rollback()
        raise
