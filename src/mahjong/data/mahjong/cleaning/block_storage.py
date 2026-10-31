"""Repack a completed task into bounded Zstd blocks; no new recovery ledger.

The original append file stays authoritative until the ordinary completion SQL
commit. An interrupted repack is simply overwritten on the next completion.
"""
from compression import zstd
import hashlib
import json

from .common import file_identity, read_json, require
from .compression import decompress_frame
from .metrics import fsync, timed

MAX_BLOCK_BYTES = 64*1024*1024
# Chosen from the bounded end-to-end speed/size comparison, not size alone.
BLOCK_COMPRESSION_LEVEL = 6
ROUND_KEYS = ('source_key', 'round_id', 'offset', 'bytes', 'views')


def compact_rounds(index):
    return [{key: row[key] for key in ROUND_KEYS} for row in index['rounds']]


def repack_blocks(append_path, games, index, ledger=None):
    from .append_storage import HashWriter
    from .resources import WriterBudget
    path = append_path.with_name('data.blocks.zst')
    hashes = {row['source_key']: json.loads(row['content'])['raw_sha256'] for row in games}
    records = []
    with append_path.open('rb') as original, path.open('w+b', buffering=0) as stream:
        budget = WriterBudget(ledger, path, stream) if ledger is not None else None
        output = HashWriter(stream, budget)
        pending, size = [], 0

        def flush():
            if not pending:
                return
            raw = b''.join(body for _, body in pending)
            with timed('compression_seconds'):
                compressed = zstd.compress(raw, options={
                    zstd.CompressionParameter.compression_level: BLOCK_COMPRESSION_LEVEL,
                    zstd.CompressionParameter.checksum_flag: 1})
            offset = output.count
            output.write(compressed)
            start = 0
            for record, body in pending:
                records.append(dict(record, offset=offset, bytes=len(compressed),
                    frame_raw_bytes=len(raw), frame_game_offset=start))
                start += len(body)
            pending.clear()

        try:
            for record in index['games']:
                original.seek(record['offset'])
                body = decompress_frame(original.read(record['bytes']), zstandard=True,
                    max_bytes=record['raw_bytes'], expected_size=record['raw_bytes'])
                if pending and size+len(body) > MAX_BLOCK_BYTES:
                    flush(); size = 0
                pending.append((record, body)); size += len(body)
            flush()
            stream.flush(); fsync(stream.fileno(), category='block_data_fsync')
        finally:
            if budget is not None:
                budget.close()
    index = dict(index, games=records, rounds=compact_rounds(index))
    info = verify_blocks(path, index, hashes)
    return index, info


def verify_blocks(path, index, hashes):
    """Verify exact original game bytes, shared extents and full block coverage."""
    physical = hashlib.sha256()
    end, current, game_end, decoded = 0, None, 0, b''
    with path.open('rb') as stream:
        for row in index['games']:
            signature = (row['offset'], row['bytes'], row['frame_raw_bytes'])
            if signature != current:
                require(current is None or game_end == len(decoded), 'STORAGE_CORRUPT', 'block game coverage')
                require(row['offset'] == end and row['bytes'] > 0 and
                        0 <= row['frame_raw_bytes'] <= MAX_BLOCK_BYTES,
                        'STORAGE_CORRUPT', 'block extent')
                compressed = stream.read(row['bytes'])
                require(len(compressed) == row['bytes'], 'STORAGE_CORRUPT', 'truncated block')
                physical.update(compressed)
                decoded = decompress_frame(compressed, zstandard=True,
                    max_bytes=row['frame_raw_bytes'], expected_size=row['frame_raw_bytes'])
                current, game_end = signature, 0
                end += row['bytes']
            require(row['frame_game_offset'] == game_end and row['raw_bytes'] >= 0 and
                    game_end+row['raw_bytes'] <= len(decoded), 'STORAGE_CORRUPT', 'game within block')
            body = decoded[game_end:game_end+row['raw_bytes']]
            require(hashlib.sha256(body).hexdigest() == hashes[row['source_key']],
                    'STORAGE_CORRUPT', 'repacked game differs from validated append bytes')
            game_end += row['raw_bytes']
    require((current is None or game_end == len(decoded)) and end == path.stat().st_size,
            'STORAGE_CORRUPT', 'block file coverage')
    return dict(bytes=end, sha256=physical.hexdigest())


def inspect_blocks(db, directory, catalog, games, marker, *, codec=None):
    from .append_storage import make_index
    from .storage import meta
    from .text_storage import information
    index = read_json(directory/'index.json.zst')
    expected = make_index(meta(db, 'binding'), games)
    strip = lambda row: {k:v for k,v in row.items()
        if k not in ('offset', 'bytes', 'frame_raw_bytes', 'frame_game_offset')}
    require(index['binding'] == expected['binding'] and
            [strip(r) for r in index['games']] == [strip(r) for r in expected['games']] and
            index['rounds'] == compact_rounds(expected), 'STORAGE_CORRUPT', 'block index changed')
    hashes = {r['source_key']:json.loads(r['content'])['raw_sha256'] for r in games}
    actual = verify_blocks(directory/'data.blocks.zst', index, hashes)
    require({p['name']:(p['bytes'],p['sha256']) for p in marker['products']} ==
            {'data.blocks.zst':(actual['bytes'],actual['sha256']),
             'index.json.zst':file_identity(directory/'index.json.zst')} and
            dict(marker, products=None) == dict(information(games,index), products=None),
            'STORAGE_CORRUPT', 'block completion receipt')
    return marker
