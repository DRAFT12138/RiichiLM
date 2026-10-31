"""Private task SQLite, EOF-gated game transactions and complete membership."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import zlib

from .catalog import Source
from .common import CleaningError, STORAGE_VERSION, canonical, connect, require
from .encoding import ARRAY_WIDTHS, EncodedView, check_layout, encode_view, pack, unpack

STATUSES = ('success', 'partial', 'failed', 'skipped')


def json_block(value):
    raw = canonical(value)
    return zlib.compress(raw, 1), len(raw)


def read_json_block(blob, byte_count, *, max_bytes=64 * 1024 * 1024):
    # Reuse bounded decompression; uint16/64 is not needed for UTF8 bytes.
    require(type(byte_count) is int and 0 <= byte_count <= max_bytes,
            'STORAGE_CORRUPT', 'JSON block size')
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(blob, byte_count + 1)
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
                len(raw) == byte_count, 'STORAGE_CORRUPT', 'JSON block length')
        return json.loads(raw)
    except (zlib.error, UnicodeError, json.JSONDecodeError) as error:
        require(False, 'STORAGE_CORRUPT', str(error))


def shard_schema(connection):
    columns = ','.join(f'{name} BLOB NOT NULL,{name}_bytes INTEGER NOT NULL,{name}_count INTEGER NOT NULL'
                       for name in ARRAY_WIDTHS)
    connection.executescript(f'''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(
          attempt_id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, pool_id INTEGER NOT NULL,
          retry_epoch INTEGER NOT NULL, ending TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS games(
          source_key TEXT PRIMARY KEY, identity TEXT NOT NULL,
          game_id TEXT, native_id TEXT, rule_ref TEXT,
          status TEXT NOT NULL CHECK(status IN ('success','partial','failed','skipped')),
          counts TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rounds(
          source_key TEXT NOT NULL REFERENCES games(source_key), round_id INTEGER NOT NULL,
          complete INTEGER NOT NULL, event_ids BLOB NOT NULL,
          event_ids_bytes INTEGER NOT NULL, event_ids_count INTEGER NOT NULL,
          facts BLOB NOT NULL, facts_bytes INTEGER NOT NULL,
          PRIMARY KEY(source_key,round_id));
        CREATE TABLE IF NOT EXISTS views(
          source_key TEXT NOT NULL, round_id INTEGER NOT NULL, viewer INTEGER NOT NULL,
          valid_through_event_id INTEGER NOT NULL, invalid_reason TEXT,
          {columns}, PRIMARY KEY(source_key,round_id,viewer),
          FOREIGN KEY(source_key,round_id) REFERENCES rounds(source_key,round_id));
        CREATE TABLE IF NOT EXISTS errors(
          error_id INTEGER PRIMARY KEY, source_key TEXT NOT NULL REFERENCES games(source_key),
          detail TEXT NOT NULL);
    ''')
    connection.commit()


def meta(connection, key):
    row = connection.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def set_meta(connection, key, value):
    connection.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
                       (key, canonical(value).decode()))


def check_attempt_fact(row):
    """Unreadable execution facts must never become a zero-cost retry."""
    for name in ('attempt_id', 'task_id', 'pool_id', 'retry_epoch'):
        require(type(row[name]) is int and row[name] >= (1 if name in ('attempt_id','pool_id') else 0),
                'ATTEMPT_CONFLICT', 'invalid attempt ' + name)
    ending=row['ending']
    require(ending in (None,'COMPLETED','RETRYABLE_FAILURE','REQUIRES_RESUME','STOPPED'),
            'ATTEMPT_CONFLICT', 'unknown worker ending')
    try:
        error=json.loads(row['error']) if row['error'] is not None else None
    except (TypeError,json.JSONDecodeError) as failure:
        raise CleaningError('ATTEMPT_CONFLICT','unreadable worker error') from failure
    if ending in (None,'COMPLETED'):
        require(error is None,'ATTEMPT_CONFLICT','error without a failed/stopped ending')
    else:
        require(type(error) is dict and all(type(error.get(k)) is str and error[k] for k in ('code','reason')),
                'ATTEMPT_CONFLICT','missing worker failure detail')


class Shard:
    def __new__(cls, path, catalog, task, **kwargs):
        if cls is Shard and catalog['binding']['storage'] != STORAGE_VERSION:
            if catalog['binding'].get('task_state') == 'append-state/1':
                from .append_storage import AppendTextShard
                return object.__new__(AppendTextShard)
            from .text_storage import TextShard
            return object.__new__(TextShard)
        return object.__new__(cls)

    def __init__(self, path, catalog, task, *, create=True):
        self.path, self.catalog, self.task = Path(path), catalog, task
        existed = self.path.exists()
        require(existed or create, 'MISSING_SHARD', str(path))
        self.connection = connect(self.path)
        try:
            if not existed:
                shard_schema(self.connection)
                with self.connection:
                    set_meta(self.connection, 'binding', self.binding)
            require(meta(self.connection, 'binding') == self.binding,
                    'BINDING_MISMATCH', str(path))
        except BaseException:
            self.connection.close()
            raise

    @property
    def binding(self):
        return dict(storage=STORAGE_VERSION, build_id=self.catalog['build_id'],
                    task_id=self.task['task_id'], members=self.task['members'])

    def close(self):
        self.connection.close()

    def committed(self):
        return {r[0] for r in self.connection.execute('SELECT source_key FROM games')}

    @contextmanager
    def game_transaction(self, source, raw_reader):
        """All four terminal statuses require verified raw EOF before COMMIT."""
        require(source.source_key in self.task['members'], 'MEMBERSHIP_MISMATCH', source.source_key)
        require(source.source_key not in self.committed(), 'DUPLICATE_SOURCE', source.source_key)
        require(not self.connection.in_transaction, 'TRANSACTION_ERROR', 'nested game transaction')
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            yield
            raw_reader.verify_eof(source)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def write_game(self, source, result, *, codec):
        require(self.connection.in_transaction, 'TRANSACTION_ERROR', 'use game_transaction')
        require(result.status in STATUSES, 'TYPE_MISMATCH', 'game terminal status')
        require(result.status != 'skipped' or result.errors, 'MISSING_FIELD', 'skip reason')
        self.connection.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?)',
            (source.source_key, canonical(source.identity()).decode(), result.game_id,
             result.native_id, canonical(result.rule_ref).decode(), result.status,
             canonical(result.counts).decode()))
        ruleset_id=(result.rule_ref or {}).get('ruleset_id')
        rule=self.catalog['binding']['rules'].get(ruleset_id,{})
        expected_stock=(rule['profile']['rule_options']['aka_counts'] if 'profile' in rule else
                        rule.get('offline_options',{}).get('aka_counts'))
        for prepared in result.rounds:
            require(prepared.aka_counts==expected_stock,'BINDING_MISMATCH',
                    'prepared tile stock differs from bound profile')
            require(type(prepared.round_id) is int and prepared.round_id >= 0,
                    'TYPE_MISMATCH', 'round_id')
            shared = pack(prepared.event_ids, 8)
            facts = json_block(dict(ledger=prepared.ledger, windows=prepared.windows,
                                   settlement=prepared.facts))
            self.connection.execute('INSERT INTO rounds VALUES (?,?,?,?,?,?,?,?)',
                (source.source_key, prepared.round_id, int(prepared.complete), *shared, *facts))
            viewers = [v.viewer for v in prepared.views]
            require(len(viewers) == len(set(viewers)), 'INVALID_REFERENCE', 'duplicate viewer')
            for view in prepared.views:
                encoded = encode_view(view, prepared, codec=codec)
                values = [source.source_key, prepared.round_id, view.viewer,
                          view.valid_through_event_id, canonical(view.invalid_reason).decode()]
                for name, width in ARRAY_WIDTHS.items():
                    values.extend(pack(encoded.arrays[name], width))
                placeholders = ','.join('?' for _ in values)
                self.connection.execute(f'INSERT INTO views VALUES ({placeholders})', values)
        for index, error in enumerate(result.errors):
            require(all(k in error for k in ('code', 'stage', 'reason')),
                    'MISSING_FIELD', 'error classification')
            self.connection.execute('INSERT INTO errors(source_key,detail) VALUES (?,?)',
                                    (source.source_key, canonical(dict(error, primary=index == 0)).decode()))

    def started(self, attempt):
        with self.connection:
            self.connection.execute('INSERT INTO attempts(attempt_id,task_id,pool_id,retry_epoch) VALUES (?,?,?,?)',
                tuple(attempt[k] for k in ('attempt_id', 'task_id', 'pool_id', 'retry_epoch')))

    def ended(self, attempt_id, ending, error=None):
        require(ending in ('COMPLETED', 'RETRYABLE_FAILURE', 'REQUIRES_RESUME', 'STOPPED'),
                'TYPE_MISMATCH', 'attempt ending')
        cursor = self.connection.execute('UPDATE attempts SET ending=?,error=? WHERE attempt_id=? AND ending IS NULL',
                (ending, canonical(error).decode(), attempt_id))
        require(cursor.rowcount == 1, 'ATTEMPT_CONFLICT', str(attempt_id))

    def complete(self, *, attempt_id=None, codec=None):
        info = check_members(self.connection, self.catalog, self.task, codec=codec)
        with self.connection:
            set_meta(self.connection, 'complete', info)
            if attempt_id is not None:
                self.ended(attempt_id, 'COMPLETED')
        return info


def logical_digest(connection):
    """Business digest excludes execution ledger and physical SQLite page order."""
    hasher = hashlib.sha256()
    hasher.update(canonical(meta(connection, 'binding')))
    orders = dict(games='source_key', rounds='source_key,round_id',
                  views='source_key,round_id,viewer', errors='source_key,detail')
    for table, order in orders.items():
        for row in connection.execute(f'SELECT * FROM {table} ORDER BY {order}'):
            values = dict(row)
            if table == 'errors':
                values.pop('error_id')
            # Compression bytes are deterministic for identical blocks in this version.
            values = {k: v.hex() if isinstance(v, bytes) else v for k, v in values.items()}
            hasher.update(canonical([table, values]))
    return hasher.hexdigest()


def read_encoded_view(row, *, max_bytes=64 * 1024 * 1024):
    arrays = {name: unpack(row[name], row[f'{name}_bytes'], row[f'{name}_count'], width,
                           max_bytes=max_bytes) for name, width in ARRAY_WIDTHS.items()}
    return EncodedView(row['viewer'], row['valid_through_event_id'],
                       json.loads(row['invalid_reason']), arrays)


def check_members(connection, catalog, task, *, codec=None):
    if catalog['binding']['storage'] != STORAGE_VERSION:
        from .text_storage import inspect_text_task
        path = connection.execute('PRAGMA database_list').fetchone()[2]
        return inspect_text_task(path, catalog, task, codec=codec)
    binding = dict(storage=STORAGE_VERSION, build_id=catalog['build_id'],
                   task_id=task['task_id'], members=task['members'])
    require(meta(connection, 'binding') == binding, 'BINDING_MISMATCH', 'shard binding')
    require(connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok',
            'STORAGE_CORRUPT', 'SQLite quick_check')
    require(not connection.execute('PRAGMA foreign_key_check').fetchall(),
            'MEMBERSHIP_MISMATCH', 'orphan records')
    sources = {s['source_key']: Source(**s) for s in catalog['sources']}
    games = connection.execute('SELECT * FROM games ORDER BY source_key').fetchall()
    require({r['source_key'] for r in games} == set(task['members']) and
            len(games) == len(task['members']), 'MEMBERSHIP_MISMATCH', 'frozen member set')
    for row in games:
        require(row['status'] in STATUSES and json.loads(row['identity']) == sources[row['source_key']].identity(),
                'MEMBERSHIP_MISMATCH', 'source identity/terminal status')
        if row['status'] == 'skipped':
            require(connection.execute('SELECT 1 FROM errors WHERE source_key=?', (row['source_key'],)).fetchone(),
                    'MISSING_FIELD', 'skip reason')
    anchors, tokens = 0, 0
    for row in connection.execute('SELECT * FROM rounds'):
        ids = unpack(row['event_ids'], row['event_ids_bytes'], row['event_ids_count'], 8)
        read_json_block(row['facts'], row['facts_bytes'])
        for view_row in connection.execute('SELECT * FROM views WHERE source_key=? AND round_id=?',
                                           (row['source_key'], row['round_id'])):
            view = read_encoded_view(view_row)
            check_layout(view, ids, codec=codec)
            anchors += view.anchor_count
            tokens += sum(len(view.arrays[k]) for k in ('header', 'history', 'hand_blocks'))
    return dict(logical_sha256=logical_digest(connection), games=len(games),
                rounds=connection.execute('SELECT COUNT(*) FROM rounds').fetchone()[0],
                views=connection.execute('SELECT COUNT(*) FROM views').fetchone()[0],
                anchors=anchors, tokens=tokens,
                statuses={s: sum(r['status'] == s for r in games) for s in STATUSES})


def inspect_shard(path, catalog, task, *, codec=None, require_complete=True):
    if catalog['binding']['storage'] != STORAGE_VERSION:
        from .text_storage import inspect_text_task
        return inspect_text_task(path, catalog, task, codec=codec, require_complete=require_complete)
    connection = connect(path, readonly=True)
    try:
        info = check_members(connection, catalog, task, codec=codec)
        if require_complete:
            require(meta(connection, 'complete') == info, 'STORAGE_CORRUPT', 'completion marker/digest')
        for row in connection.execute('SELECT * FROM attempts'):
            check_attempt_fact(row)
            require(row['task_id'] == task['task_id'], 'ATTEMPT_CONFLICT', 'task identity')
            require(row['ending'] != 'COMPLETED' or meta(connection, 'complete') is not None,
                    'ATTEMPT_CONFLICT', 'completed attempt without marker')
        return info
    finally:
        connection.close()
