"""Read plain text by game frame and round offset; no SQLite queries or rule Replay."""
from collections import OrderedDict
import json
from pathlib import Path
import zlib

from mahjong.tokenizer import MahjongTokenizer
from .catalog import validate_catalog
from .common import read_json, json_path
from .common import TEXT_STORAGE_VERSION, file_identity, require
from .encoding import EncodedView, assemble_prefix, check_layout
from .text_codec import decode_round
from .compression import compressed_storage, product_names, decompress_frame, BLOCK_STORAGE


class TextCleaningReader:
    def __init__(self, build_dir, *, cache_bytes=64*1024*1024, open_shards=8,
                 max_block_bytes=64*1024*1024):
        self.root = Path(build_dir)
        require(json_path((self.root/'manifest.json')).is_file(), 'UNPUBLISHED_BUILD', str(self.root))
        self.catalog = read_json((self.root/'build.json'))
        validate_catalog(self.catalog, check_implementation=False)
        self.manifest = read_json((self.root/'manifest.json'))
        self.body_name, self.index_name = product_names(self.catalog)
        self.blocked = self.catalog['binding'].get('compression') == BLOCK_STORAGE
        require(self.manifest['build_id']==self.catalog['build_id'] and
                self.manifest['binding']==self.catalog['binding'], 'BINDING_MISMATCH', 'text manifest')
        require(file_identity(self.root/'summary.json')[1]==self.manifest['summary_sha256'],
                'STORAGE_CORRUPT', 'summary digest')
        require([s['task_id'] for s in self.manifest['shards']] ==
                [t['task_id'] for t in self.catalog['tasks']], 'STORAGE_CORRUPT', 'text task set')
        require(cache_bytes>=0 and open_shards>0 and max_block_bytes>0, 'TYPE_MISMATCH', 'reader limits')
        self.codec = MahjongTokenizer()
        self.sources = {s['source_key']:s for s in self.catalog['sources']}
        self.locations = {s:t['task_id'] for t in self.catalog['tasks'] for s in t['members']}
        self.shards = {s['task_id']:s for s in self.manifest['shards']}
        self.cache_bytes, self.max_block_bytes, self.open_shards = cache_bytes, max_block_bytes, open_shards
        self._files, self._indices, self._cache = OrderedDict(), {}, OrderedDict()
        self._cache_size = 0

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def close(self):
        for stream in self._files.values(): stream.close()
        self._files.clear()
        self._cache.clear()
        self._cache_size = 0

    def _index(self, task_id):
        if task_id in self._indices: return self._indices[task_id]
        item = self.shards[task_id]
        directory = self.root/'shards'/f'part-{task_id:06d}'
        require({p['name'] for p in item['products']} == {self.body_name, self.index_name},
                'STORAGE_CORRUPT', 'text product names')
        for p in item['products']:
            path = directory/p['name']
            require(path.is_file() and file_identity(path)==(p['bytes'],p['sha256']),
                    'STORAGE_CORRUPT', 'text product missing/changed: '+str(path))
        index = read_json(directory/self.index_name)
        task = self.catalog['tasks'][task_id]
        require(index['binding']==dict(storage=self.catalog['binding']['storage'], build_id=self.catalog['build_id'],
                task_id=task_id,members=task['members']), 'BINDING_MISMATCH', 'text index binding')
        require(len(index['games'])==len(task['members']) and
                {g['source_key'] for g in index['games']}==set(task['members']),
                'MEMBERSHIP_MISMATCH', 'text index members')
        end, frame, game_end = 0, None, 0
        for row in index['games']:
            if self.blocked:
                signature = (row['offset'], row['bytes'], row['frame_raw_bytes'])
                if signature != frame:
                    require(frame is None or game_end == frame[2], 'STORAGE_CORRUPT', 'block coverage')
                    require(row['offset'] == end and 0<=row['frame_raw_bytes']<=self.max_block_bytes,
                            'STORAGE_CORRUPT', 'block offset/size')
                    frame, game_end = signature, 0
                    end += row['bytes']
                require(row['frame_game_offset'] == game_end and game_end+row['raw_bytes'] <= frame[2],
                        'STORAGE_CORRUPT', 'game offset in block')
                game_end += row['raw_bytes']
            require(type(row['offset']) is int and (self.blocked or row['offset']==end) and
                    type(row['bytes']) is int and row['bytes']>0 and
                    type(row['raw_bytes']) is int and 0<=row['raw_bytes']<=self.max_block_bytes,
                    'STORAGE_CORRUPT','text game extent')
            if not self.blocked:
                end+=row['bytes']
        require(not self.blocked or frame is None or game_end == frame[2], 'STORAGE_CORRUPT', 'last block coverage')
        require(end==(directory/self.body_name).stat().st_size, 'STORAGE_CORRUPT', 'text size/extent')
        index['game_map'] = {g['source_key']:g for g in index['games']}
        index['round_map'] = {}
        ends = dict.fromkeys(index['game_map'],0)
        for row in index['rounds']:
            key = (row['source_key'], row['round_id'])
            require(key not in index['round_map'] and key[0] in index['game_map'] and
                    type(key[1]) is int and key[1]>=0 and
                    type(row['offset']) is int and row['offset']==ends[key[0]] and
                    type(row['bytes']) is int and row['bytes']>0,
                    'STORAGE_CORRUPT', 'text round extent/identity')
            require(len({v[0] for v in row['views']})==len(row['views']) and
                    all(len(v)==3 and type(v[0]) is int and 0<=v[0]<4 and
                        type(v[1]) is int and v[1]>=0 and type(v[2]) is int and v[2]>=0 for v in row['views']),
                    'STORAGE_CORRUPT', 'view index')
            index['round_map'][key] = row
            ends[key[0]] += row['bytes']
        require(all(ends[k]==g['raw_bytes'] for k,g in index['game_map'].items()),
                'STORAGE_CORRUPT','text game decoded extents')
        self._indices[task_id] = index
        return index

    def _stream(self, task_id):
        self._index(task_id)
        if task_id not in self._files:
            self._files[task_id] = (self.root/'shards'/f'part-{task_id:06d}'/self.body_name).open('rb')
        self._files.move_to_end(task_id)
        while len(self._files)>self.open_shards: self._files.popitem(last=False)[1].close()
        return self._files[task_id]

    def iter_games(self):
        for task in self.shards:
            yield from self._index(task)['games']

    def iter_round_refs(self, source_key=None):
        if source_key is not None:
            task = self.locations[source_key]
            for row in self._index(task)['rounds']:
                if row['source_key'] == source_key:
                    yield row['source_key'], row['round_id']
            return
        for task in self.shards:
            for row in self._index(task)['rounds']:
                yield row['source_key'], row['round_id']

    def iter_views(self):
        for task in self.shards:
            for row in sorted(self._index(task)['rounds'], key=lambda r:(r['source_key'],r['round_id'])):
                for viewer, valid, anchors in sorted(row['views']):
                    yield dict(source_key=row['source_key'],round_id=row['round_id'],viewer=viewer,
                               valid_through_event_id=valid,anchors=anchors)

    def read_round(self, source_key, round_id):
        key = (source_key,round_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        task = self.locations[source_key]
        item = self._index(task)['round_map'][key]
        game = self._index(task)['game_map'][source_key]
        frame_key=(task, 'frame', game['offset']) if self.blocked else (source_key,None)
        try:
            if frame_key in self._cache:
                self._cache.move_to_end(frame_key)
                decoded=self._cache[frame_key][0]
            else:
                stream=self._stream(task)
                stream.seek(game['offset'])
                data=stream.read(game['bytes'])
                decoded=decompress_frame(data,zstandard=compressed_storage(self.catalog),
                    max_bytes=game.get('frame_raw_bytes',game['raw_bytes']),
                    expected_size=game.get('frame_raw_bytes',game['raw_bytes']))
                self._remember(frame_key,decoded,len(decoded))
            start=game.get('frame_game_offset',0)+item['offset']
            raw=decoded[start:start+item['bytes']]
            require(raw.endswith(b'\n') and len(raw)==item['bytes'], 'STORAGE_CORRUPT','text round block')
            record = decode_round(raw)
        except (zlib.error,UnicodeError,json.JSONDecodeError) as error:
            require(False,'STORAGE_CORRUPT',str(error))
        require((record['source_key'],record['round_id'])==key, 'STORAGE_CORRUPT', 'round identity')
        require([[v['viewer'],v['valid_through_event_id'],len(v['arrays']['anchors'])//5]
                  for v in record['views']]==item['views'], 'STORAGE_CORRUPT', 'view summaries')
        for v in record['views']:
            check_layout(EncodedView(v['viewer'],v['valid_through_event_id'],v['invalid_reason'],v['arrays']),
                         record['event_ids'],codec=self.codec)
        from .audit import check_cleaning_audits
        check_cleaning_audits(record['facts']['outcomes'])
        self._check_policy_decisions(record)
        record['_window_map']={w:refs for w,refs in record['facts']['source_refs']}
        self._remember(key,record,item['bytes'])
        return record

    def _check_policy_decisions(self, record):
        decisions = record['facts'].get('policy_decisions', [])
        require(type(decisions) is list, 'STORAGE_CORRUPT', 'policy decision list')
        view_map = {view['viewer']: view for view in record['views']}
        anchors_by_window = {}
        for viewer, view in view_map.items():
            anchors = view['arrays']['anchors']
            for index in range(0, len(anchors), 5):
                anchors_by_window[(anchors[index], viewer)] = (anchors[index+1], anchors[index+2])
        seen = set()
        for decision in decisions:
            require(type(decision) is dict and set(decision) == {
                'window_id','trigger_id','viewer','response','candidates','target_index','target_event_id'
            }, 'STORAGE_CORRUPT', 'policy decision fields')
            viewer = decision['viewer']
            require(type(viewer) is int and viewer in view_map and type(decision['response']) is bool,
                    'STORAGE_CORRUPT', 'policy decision identity')
            require(type(decision['window_id']) is int and type(decision['trigger_id']) is int and
                    decision['window_id'] >= 0 and decision['trigger_id'] >= 0 and
                    decision['window_id'] // 4 == decision['trigger_id'] and
                    (decision['window_id'], viewer) not in seen,
                    'STORAGE_CORRUPT', 'policy decision window')
            seen.add((decision['window_id'], viewer))
            view = view_map[viewer]
            anchors = view['arrays']['anchors']
            anchor = anchors_by_window.get((decision['window_id'], viewer))
            require(anchor is not None and anchor[0] == decision['trigger_id'],
                    'STORAGE_CORRUPT', 'policy decision has no prefix anchor')
            candidates = decision['candidates']
            require(type(candidates) is list and len(candidates) >= 2 and
                    all(type(action) is list and action and all(type(token) is int and 0 <= token < self.codec.vocab_size
                        for token in action) for action in candidates) and
                    len({tuple(action) for action in candidates}) == len(candidates),
                    'STORAGE_CORRUPT', 'policy candidate set')
            for action in candidates:
                self.codec.decode_action_suffix(action)
            target = decision['target_index']
            event_id = decision['target_event_id']
            require(type(target) is int and 0 <= target < len(candidates) and
                    type(event_id) is int and event_id > anchor[1] and
                    event_id <= view['valid_through_event_id'],
                    'STORAGE_CORRUPT', 'policy target identity')

    def read_policy_decisions(self, source_key, round_id):
        return self.read_round(source_key, round_id)['facts'].get('policy_decisions', [])

    def _remember(self,key,value,size):
        if size<=self.cache_bytes:
            while self._cache_size+size>self.cache_bytes:
                _, (_,removed)=self._cache.popitem(last=False)
                self._cache_size-=removed
            self._cache[key]=(value,size)
            self._cache_size+=size

    def read_view(self, source_key, round_id, viewer):
        record = self.read_round(source_key,round_id)
        v = next((v for v in record['views'] if v['viewer']==viewer),None)
        if v is None: raise KeyError((source_key,round_id,viewer))
        return EncodedView(v['viewer'],v['valid_through_event_id'],v['invalid_reason'],v['arrays'])

    def read_prefix(self, source_key, round_id, viewer, window_id):
        view = self.read_view(source_key,round_id,viewer)
        ids, identity = assemble_prefix(view,window_id,self.codec)
        record = self.read_round(source_key,round_id)
        refs=record['_window_map'][window_id]
        game = self._index(self.locations[source_key])['game_map'][source_key]
        return ids, dict(**identity,source_key=source_key,round_id=round_id,viewer=viewer,
                         game_id=game['game_id'],rule_ref=game['rule_ref'],source=self.sources[source_key],
                         source_refs=refs)
