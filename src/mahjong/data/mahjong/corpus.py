"""Unified, ledger-free text/3 training data with bounded shard caches."""
from collections import OrderedDict
from compression import zstd
import json
from pathlib import Path

from mahjong.tokenizer import MahjongTokenizer
from .cleaning.catalog import codec_binding
from .cleaning.common import digest, file_identity, read_json, require
from .cleaning.text_reader import TextCleaningReader

CORPUS_VERSION = 'mahjong-policy-corpus/1'


def corpus_identity(manifest):
    return digest({k: v for k, v in manifest.items() if k != 'build_id'})


def open_policy_reader(directory, **kwargs):
    if (Path(directory) / 'dataset.json').is_file():
        return PolicyCorpusReader(directory, **kwargs)
    from .cleaning.reader import CleaningReader
    return CleaningReader(directory, **kwargs)


class _Sources:
    def __init__(self, reader): self.reader = reader

    def __getitem__(self, source):
        reader = self.reader
        task = reader.locations[source]
        game = reader._index(task)['game_map'][source]
        group = reader.manifest['source_groups'][reader.shards[task]['group']]
        return dict(source_key=source, platform=group['platform'], shard=group['shard'],
                    revision=group['revision'], relative_path=game['game_id'].split(':', 3)[-1])


class PolicyCorpusReader(TextCleaningReader):
    """Reuse the verified round decoder; cleaning SQL/catalogs are unnecessary."""
    def __init__(self, directory, *, cache_bytes=64*1024*1024, open_shards=8,
                 max_block_bytes=64*1024*1024):
        self.root = Path(directory)
        self.manifest = read_json(self.root/'dataset.json')
        m = self.manifest
        require(m.get('format') == CORPUS_VERSION and m['build_id'] == corpus_identity(m),
                'BINDING_MISMATCH', 'corpus identity')
        require(m['binding']['codec'] == codec_binding() and
                m['binding']['storage'] == 'mjai-clean-text/3' and
                m['binding']['compression'] == 'zstd-blocks/1',
                'BINDING_MISMATCH', 'corpus codec/storage')
        require(cache_bytes >= 0 and open_shards > 0 and max_block_bytes > 0,
                'TYPE_MISMATCH', 'corpus reader limits')
        self.catalog = {'build_id': m['build_id'], 'binding': m['binding']}
        self.codec = MahjongTokenizer()
        self.body_name, self.index_name, self.blocked = 'data.blocks.zst', 'index.json.zst', True
        self.shards = {s['task_id']: s for s in m['shards']}
        require(list(self.shards) == list(range(len(m['shards']))),
                'MEMBERSHIP_MISMATCH', 'corpus shard order')
        self.locations, self.sources = {}, _Sources(self)
        self.cache_bytes, self.open_shards, self.max_block_bytes = cache_bytes, open_shards, max_block_bytes
        self._files, self._indices, self._cache = OrderedDict(), OrderedDict(), OrderedDict()
        self._cache_size, self._verified = 0, {}
        self._inventory_verified = False

    def close(self):
        super().close()
        self._indices.clear()
        self.locations.clear()
        self._verified.clear()

    def iter_games(self):
        info = self.manifest['games']
        path = self.root/'games.jsonl.zst'
        if not self._inventory_verified:
            require(file_identity(path) == (info['bytes'], info['sha256']),
                    'STORAGE_CORRUPT', 'corpus game inventory')
            self._inventory_verified = True
        count = 0
        with zstd.open(path, 'rb') as stream:
            for line in stream:
                source, game_id, native_id, task = json.loads(line)
                require(task in self.shards, 'MEMBERSHIP_MISMATCH', 'game shard')
                task = self.shards[task]['task_id']
                previous = self.locations.setdefault(source, task)
                require(previous == task, 'MEMBERSHIP_MISMATCH', 'duplicate game across shards')
                count += 1
                yield dict(source_key=source, game_id=game_id, native_id=native_id)
        require(count == self.manifest['counts']['games'], 'STORAGE_CORRUPT', 'corpus game count')

    def _index(self, task_id):
        if task_id in self._indices:
            self._indices.move_to_end(task_id)
            return self._indices[task_id]
        item = self.shards[task_id]
        directory = self.root/'shards'/f'part-{task_id:06d}'
        require({p['name'] for p in item['products']} == {self.body_name, self.index_name},
                'STORAGE_CORRUPT', 'corpus product names')
        signature = []
        for product in item['products']:
            path = directory/product['name']
            stat = path.stat()
            require(stat.st_size == product['bytes'], 'STORAGE_CORRUPT', 'corpus product size')
            signature.append((stat.st_size, stat.st_mtime_ns))
        if self._verified.get(task_id) != signature:
            for product in item['products']:
                require(file_identity(directory/product['name']) == (product['bytes'], product['sha256']),
                        'STORAGE_CORRUPT', 'corpus product SHA')
            self._verified[task_id] = signature
        index = read_json(directory/self.index_name)
        require(index['binding'] == {'storage': 'mjai-clean-text/3', 'partition_id': item['partition_id']},
                'BINDING_MISMATCH', 'corpus index')
        games = index['games']
        game_map = {g['source_key']: g for g in games}
        require(len(games) == len(game_map) == item['games'] and
                digest([g['source_key'] for g in games]) == item['members_digest'],
                'MEMBERSHIP_MISMATCH', 'corpus index games')
        end, frame, game_end = 0, None, 0
        for game in games:
            signature = (game['offset'], game['bytes'], game['frame_raw_bytes'])
            if signature != frame:
                require((frame is None or game_end == frame[2]) and game['offset'] == end and
                        0 <= game['frame_raw_bytes'] <= self.max_block_bytes,
                        'STORAGE_CORRUPT', 'corpus block coverage')
                frame, game_end = signature, 0
                end += game['bytes']
            require(game['frame_game_offset'] == game_end and
                    0 <= game['raw_bytes'] <= self.max_block_bytes and
                    game_end + game['raw_bytes'] <= frame[2], 'STORAGE_CORRUPT', 'corpus game extent')
            game_end += game['raw_bytes']
            previous = self.locations.setdefault(game['source_key'], task_id)
            require(previous == task_id, 'MEMBERSHIP_MISMATCH', 'game in multiple shards')
        require((frame is None or game_end == frame[2]) and end == (directory/self.body_name).stat().st_size,
                'STORAGE_CORRUPT', 'corpus body extent')
        ends, round_map, views, anchors = dict.fromkeys(game_map, 0), {}, 0, 0
        for row in index['rounds']:
            key = row['source_key'], row['round_id']
            require(key not in round_map and key[0] in game_map and type(key[1]) is int and key[1] >= 0 and
                    row['offset'] == ends[key[0]] and type(row['bytes']) is int and row['bytes'] > 0,
                    'STORAGE_CORRUPT', 'corpus round extent')
            require([v[0] for v in row['views']] == [0, 1, 2, 3] and all(
                len(v) == 3 and type(v[1]) is int and v[1] >= 0 and type(v[2]) is int and v[2] >= 0
                for v in row['views']), 'STORAGE_CORRUPT', 'corpus round views')
            round_map[key] = row
            ends[key[0]] += row['bytes']
            views += len(row['views']); anchors += sum(v[2] for v in row['views'])
        require(all(ends[k] == g['raw_bytes'] for k, g in game_map.items()) and
                len(round_map) == item['rounds'] and views == item['views'] and anchors == item['anchors'],
                'STORAGE_CORRUPT', 'corpus index counts')
        index.update(game_map=game_map, round_map=round_map)
        self._indices[task_id] = index
        while len(self._indices) > self.open_shards:
            self._indices.popitem(last=False)
        return index
