"""Read only published prefix blocks; never import or call the rules engine."""
from collections import OrderedDict
import json
from pathlib import Path

from mahjong.tokenizer import MahjongTokenizer
from .catalog import validate_catalog
from .common import read_json, json_path
from .common import connect, file_identity, require
from .encoding import assemble_prefix, check_layout, unpack
from .storage import meta, read_encoded_view, read_json_block


class CleaningReader:
    def __new__(cls, build_dir, **kwargs):
        from .common import is_text_storage
        path = Path(build_dir)/'build.json'
        if cls is CleaningReader and json_path(path).exists() and is_text_storage(read_json(path)['binding']['storage']):
            from .text_reader import TextCleaningReader
            return TextCleaningReader(build_dir, **kwargs)
        return object.__new__(cls)

    def __init__(self, build_dir, *, cache_bytes=64 * 1024 * 1024, open_shards=8,
                 max_block_bytes=64 * 1024 * 1024):
        self.root = Path(build_dir)
        require(json_path((self.root / 'manifest.json')).is_file(), 'UNPUBLISHED_BUILD', str(self.root))
        self.catalog = read_json((self.root / 'build.json'))
        validate_catalog(self.catalog, check_implementation=False)
        self.codec = MahjongTokenizer()
        self.manifest = read_json((self.root / 'manifest.json'))
        require(self.manifest['build_id'] == self.catalog['build_id'] and
                self.manifest['binding'] == self.catalog['binding'], 'BINDING_MISMATCH', 'manifest')
        require(file_identity(self.root / 'summary.json')[1] == self.manifest['summary_sha256'],
                'STORAGE_CORRUPT', 'summary digest')
        require([s['task_id'] for s in self.manifest['shards']] ==
                [t['task_id'] for t in self.catalog['tasks']], 'STORAGE_CORRUPT', 'manifest task set/order')
        self.sources = {s['source_key']: s for s in self.catalog['sources']}
        self.locations = {source: task['task_id'] for task in self.catalog['tasks'] for source in task['members']}
        self.shards = {s['task_id']: s for s in self.manifest['shards']}
        self._connections, self._cache = OrderedDict(), OrderedDict()
        self.cache_bytes, self.max_block_bytes = cache_bytes, max_block_bytes
        self.open_shards, self._cache_size = open_shards, 0
        require(cache_bytes >= 0 and open_shards > 0 and max_block_bytes > 0, 'TYPE_MISMATCH', 'reader limits')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._cache.clear()
        self._cache_size = 0

    def _connection(self, task_id):
        if task_id in self._connections:
            self._connections.move_to_end(task_id)
            return self._connections[task_id]
        item = self.shards[task_id]
        expected_path = f'shards/part-{task_id:06d}.sqlite'
        require(item['path'] == expected_path, 'STORAGE_CORRUPT', 'shard path')
        path = self.root / expected_path
        require(path.is_file() and file_identity(path) == (item['bytes'], item['sha256']),
                'STORAGE_CORRUPT', 'published shard missing/changed')
        connection = connect(path, readonly=True)
        try:
            task = self.catalog['tasks'][task_id]
            from .common import STORAGE_VERSION
            require(meta(connection, 'binding') == dict(storage=STORAGE_VERSION,
                    build_id=self.catalog['build_id'], task_id=task_id, members=task['members']),
                    'BINDING_MISMATCH', 'shard binding')
        except BaseException:
            connection.close()
            raise
        self._connections[task_id] = connection
        while len(self._connections) > self.open_shards:
            self._connections.popitem(last=False)[1].close()
        return connection

    def iter_views(self):
        for task_id in self.shards:
            connection = self._connection(task_id)
            for row in connection.execute('SELECT source_key,round_id,viewer,valid_through_event_id,anchors_count FROM views ORDER BY source_key,round_id,viewer'):
                yield dict(source_key=row['source_key'], round_id=row['round_id'], viewer=row['viewer'],
                           valid_through_event_id=row['valid_through_event_id'], anchors=row['anchors_count'] // 5)

    def read_view(self, source_key, round_id, viewer):
        key = (source_key, round_id, viewer)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        connection = self._connection(self.locations[source_key])
        row = connection.execute('SELECT * FROM views WHERE source_key=? AND round_id=? AND viewer=?', key).fetchone()
        if row is None:
            raise KeyError(key)
        shared = connection.execute('SELECT * FROM rounds WHERE source_key=? AND round_id=?', key[:2]).fetchone()
        ids = unpack(shared['event_ids'], shared['event_ids_bytes'], shared['event_ids_count'], 8,
                     max_bytes=self.max_block_bytes)
        view = read_encoded_view(row, max_bytes=self.max_block_bytes)
        check_layout(view, ids, codec=self.codec)
        size = sum(len(v) * (2 if k in ('header', 'history', 'hand_blocks') else 8)
                   for k, v in view.arrays.items())
        if size <= self.cache_bytes:
            while self._cache_size + size > self.cache_bytes:
                _, (_, removed) = self._cache.popitem(last=False)
                self._cache_size -= removed
            self._cache[key] = (view, size)
            self._cache_size += size
        return view

    def read_prefix(self, source_key, round_id, viewer, window_id):
        view = self.read_view(source_key, round_id, viewer)
        ids, identity = assemble_prefix(view, window_id, self.codec)
        connection = self._connection(self.locations[source_key])
        row = connection.execute('SELECT facts,facts_bytes FROM rounds WHERE source_key=? AND round_id=?',
                                 (source_key, round_id)).fetchone()
        facts = read_json_block(row['facts'], row['facts_bytes'], max_bytes=self.max_block_bytes)
        window = next(w for w in facts['windows'] if w['window_id'] == window_id)
        game = connection.execute('SELECT game_id,rule_ref FROM games WHERE source_key=?', (source_key,)).fetchone()
        return ids, dict(**identity, source_key=source_key, round_id=round_id, viewer=viewer,
                         game_id=game['game_id'], rule_ref=json.loads(game['rule_ref']),
                         source=self.sources[source_key], source_refs=window.get('source_refs', []))
