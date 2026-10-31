"""Canonical identities, durable files, and fixed local-process locks."""
from mahjong.envs.mjai_errors import CleaningError, require
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from .metrics import fsync as measured_fsync

STORAGE_VERSION = 'mjai-clean-storage/1'
TEXT_STORAGE_VERSION = 'mjai-clean-text/3'
LEGACY_TEXT_STORAGE_VERSIONS = frozenset({'mjai-clean-text/2'})
TEXT_STORAGE_VERSIONS = LEGACY_TEXT_STORAGE_VERSIONS | {TEXT_STORAGE_VERSION}


def is_text_storage(version):
    return version in TEXT_STORAGE_VERSIONS


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_identity(path):
    size, hasher = 0, hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            hasher.update(chunk)
    return size, hasher.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        measured_fsync(fd,category='directory_fsync')
    finally:
        os.close(fd)


def sync_file(path):
    with Path(path).open('rb') as stream:
        measured_fsync(stream.fileno(),category='file_fsync')


def json_path(path):
    path = Path(path)
    compressed = Path(str(path) + '.zst')
    require(not (path.is_file() and compressed.is_file()), 'ARTIFACT_CONFLICT', 'two metadata representations')
    return compressed if compressed.is_file() else path


def read_json(path):
    path = json_path(path)
    raw = path.read_bytes()
    if path.suffix == '.zst':
        from .compression import decompress_frame
        raw = decompress_frame(raw, zstandard=True, max_bytes=128*1024*1024)
    return json.loads(raw)


def atomic_bytes(path, raw):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            measured_fsync(stream.fileno(),category='metadata_fsync')
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value, *, compressed=False):
    path = Path(path)
    raw = canonical(value) + b'\n'
    if compressed:
        from compression import zstd
        if path.suffix != '.zst':
            path = Path(str(path) + '.zst')
        raw = zstd.compress(raw, level=9)
    atomic_bytes(path, raw)


@contextmanager
def fixed_lock(path):
    """Never unlink this file: old workers must lock the same inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def connect(path, *, readonly=False):
    path = Path(path)
    if path.suffix == '.zst':
        require(readonly, 'READ_ONLY', 'completed compressed task state')
        from .compression import decompress_frame
        raw = decompress_frame(path.read_bytes(), zstandard=True, max_bytes=64*1024*1024)
        connection = sqlite3.connect(':memory:')
        connection.deserialize(raw)
        connection.execute('PRAGMA query_only=ON')
    elif readonly:
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        connection.execute('PRAGMA query_only=ON')
    else:
        connection = sqlite3.connect(path)
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute('PRAGMA synchronous=FULL')
    connection.execute('PRAGMA foreign_keys=ON')
    connection.row_factory = sqlite3.Row
    return connection
