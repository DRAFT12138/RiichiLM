"""Immutable bounded JSONL episodes; offsets exist only in the reader."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from mahjong.contracts import MahjongError, canonical_json, encode_json_contract
from mahjong.inference.policy import digest
from mahjong.training.checkpoint import _fsync_directory, _write_bytes, _stream_digest

SCHEMA = 'mahjong-rl-trajectory/v1'


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise MahjongError('CHECKPOINT_INVALID', 'artifact path escapes run')
    return path


class EpisodeWriter:
    def __init__(self, root, episode_id, limits, header):
        self.root, self.limits = Path(root), limits
        self.root.mkdir(parents=True, exist_ok=True)
        if not episode_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in episode_id):
            raise ValueError('invalid episode ID')
        self.final = self.root / episode_id
        self.staging = self.root / ('.tmp-' + uuid.uuid4().hex)
        self.staging.mkdir()
        self.stream = (self.staging / 'records.jsonl').open('xb')
        self.size = self.count = 0
        self.sha = hashlib.sha256()
        self.write(dict(type='header', key='header', schema=SCHEMA, **header))

    def write(self, value):
        encoded = (canonical_json(encode_json_contract(value)) + '\n').encode('utf8')
        if (len(encoded) > self.limits['max_record_bytes'] or self.size + len(encoded) > self.limits['max_bytes']
                or self.count >= self.limits['max_records']):
            raise MahjongError('DATA_READ_FAILED', 'trajectory storage budget exceeded')
        self.stream.write(encoded)
        self.sha.update(encoded)
        self.size += len(encoded)
        self.count += 1

    def publish(self, final_scores, ranks):
        self.write(dict(type='trailer', key='trailer', status='complete', final_scores=list(final_scores), ranks=ranks))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        manifest = dict(schema=SCHEMA, status='complete', bytes=self.size, records=self.count, sha256=self.sha.hexdigest())
        _write_bytes(self.staging / 'manifest.json', (canonical_json(manifest) + '\n').encode())
        _fsync_directory(self.staging)
        if self.final.exists():
            raise MahjongError('CHECKPOINT_INVALID', 'immutable episode already exists')
        os.rename(self.staging, self.final)
        _fsync_directory(self.root)
        return dict(path='trajectories/' + self.final.name, manifest_hash=digest(manifest), **manifest)

    def abort(self):
        self.stream.close()
        if self.staging.exists():
            failures = self.root.parent / 'failures'
            failures.mkdir(exist_ok=True)
            destination = failures / (self.final.name + '-' + uuid.uuid4().hex)
            os.rename(self.staging, destination)
            entries = sorted((p for p in failures.iterdir() if p.is_dir()), key=lambda p:p.stat().st_mtime_ns)
            sizes = {p:sum(f.stat().st_size for f in p.iterdir() if f.is_file()) for p in entries}
            total = sum(sizes.values())
            for entry in entries:
                if total <= self.limits['max_failure_bytes']: break
                shutil.rmtree(entry); total -= sizes[entry]



class EpisodeReader:
    def __init__(self, run_dir, ref, limits):
        self.directory = checked_path(run_dir, ref['path'])
        self.path = self.directory / 'records.jsonl'
        self.ref = ref
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        if (digest(manifest) != ref['manifest_hash'] or manifest['schema'] != SCHEMA or manifest['status'] != 'complete'
                or any(ref[k] != manifest[k] for k in manifest)):
            raise MahjongError('CHECKPOINT_INVALID', 'trajectory manifest mismatch')
        self.offsets = {}
        sha, size, count, offset_bytes = hashlib.sha256(), 0, 0, 0
        with self.path.open('rb') as stream:
            while True:
                raw = stream.readline(limits['max_record_bytes'] + 1)
                if not raw:
                    break
                if not raw.endswith(b'\n') or len(raw) > limits['max_record_bytes']:
                    raise MahjongError('CHECKPOINT_INVALID', 'trajectory record budget exceeded')
                record = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                key = record['key']
                if not isinstance(key, str) or key in self.offsets:
                    raise MahjongError('CHECKPOINT_INVALID', 'duplicate/invalid trajectory key')
                self.offsets[key] = (size, len(raw))
                sha.update(raw)
                size += len(raw)
                count += 1
                offset_bytes += len(key.encode()) + 96
                if (size > limits['max_bytes'] or count > limits['max_records']
                        or offset_bytes > limits['max_offset_bytes']):
                    raise MahjongError('CHECKPOINT_INVALID', 'trajectory scan budget exceeded')
        if (size, count, sha.hexdigest()) != (ref['bytes'], ref['records'], ref['sha256']):
            raise MahjongError('CHECKPOINT_INVALID', 'trajectory content hash mismatch')
        self.header, self.trailer = self.read('header'), self.read('trailer')
        if list(self.offsets)[0] != 'header' or list(self.offsets)[-1] != 'trailer' or self.trailer['status'] != 'complete':
            raise MahjongError('CHECKPOINT_INVALID', 'incomplete episode')

    def read(self, key):
        offset, length = self.offsets[key]
        with self.path.open('rb') as stream:
            stream.seek(offset)
            return json.loads(stream.read(length))

    def records(self):
        for key in self.offsets:
            yield self.read(key)
