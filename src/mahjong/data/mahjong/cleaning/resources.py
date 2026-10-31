"""Coarse byte credits shared by writers, with reserved metadata headroom.

Manager state is ephemeral. Durable files are the accounting authority after all
old workers have been reaped. No lock or statvfs is taken for ordinary small writes.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import time

from .common import CleaningError, require, fixed_lock


def tree_bytes(path):
    total = 0
    if not Path(path).exists():
        return total
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            file = Path(root)/name
            if not file.is_symlink():
                try:
                    total += file.stat().st_size
                except FileNotFoundError:
                    pass
    return total


@dataclass
class BudgetLedger:
    entries: object
    lock: object
    state: object
    max_work_bytes: int
    min_free_bytes: int
    metadata_reserve_bytes: int
    write_credit_bytes: int

    @classmethod
    def create(cls, manager, config, lock_path):
        values = {k:config.get(k,v) for k,v in dict(max_work_bytes=512*1024*1024,
            min_free_bytes=2*1024**3, metadata_reserve_bytes=64*1024*1024,
            write_credit_bytes=8*1024*1024).items()}
        require(all(type(v) is int and v >= 0 for v in values.values()) and
                values['write_credit_bytes'] > 0 and values['max_work_bytes'] > 0,
                'TYPE_MISMATCH', 'disk budget')
        state = manager.Namespace()
        state.root = ''; state.peak_work_bytes = 0; state.peak_reserved_bytes = 0
        state.host_disk_path = config.get('host_disk_path')
        require(state.host_disk_path is None or Path(state.host_disk_path).is_dir(),
                'INVALID_INPUT', 'host disk path must be mounted')
        state.quarantines={str(p.resolve()):tree_bytes(p)
            for p in Path(lock_path).parent.rglob('.quarantine')}
        state.extra_quarantine_bytes=sum(state.quarantines.values())
        return cls(manager.dict(), str(lock_path), state, **values)

    def activate(self, root):
        with fixed_lock(self.lock):
            require(not self.entries, 'RESOURCE_LIMIT', 'previous writers still own credits')
            self.state.root = str(Path(root).resolve())
            self.state.extra_quarantine_bytes=(sum(self.state.quarantines.values())-
                self.state.quarantines.get(str((Path(root)/'.quarantine').resolve()),0))

    def usage(self):
        root = Path(self.state.root)
        actual = tree_bytes(root/'.work') + tree_bytes(root/'.quarantine') + self.state.extra_quarantine_bytes
        for name in ('state.sqlite','state.sqlite-journal','build.json','build.json.zst'):
            p = root/name
            if p.exists():
                actual += p.stat().st_size
        unused = 0
        for path, end in self.entries.items():
            try:
                size = Path(path).stat().st_size
            except FileNotFoundError:
                size = 0
            unused += max(0, end-size)
        return actual, unused

    def _check(self, extra=0):
        actual, unused = self.usage()
        free = shutil.disk_usage(self.state.root).free
        require(actual+unused+extra+self.metadata_reserve_bytes <= self.max_work_bytes,
                'WORK_LIMIT', 'unpublished bytes, credits and metadata exceed work budget')
        # free already excludes bytes on disk: subtract only unspent reservations.
        require(free-unused-extra >= self.min_free_bytes+self.metadata_reserve_bytes,
                'SPACE_LOW', 'insufficient disk free for data and metadata reserve')
        if self.state.host_disk_path is not None:
            require(shutil.disk_usage(self.state.host_disk_path).free-unused-extra >=
                    self.min_free_bytes+self.metadata_reserve_bytes,
                    'SPACE_LOW', 'insufficient host disk free: '+self.state.host_disk_path)
        self.state.peak_work_bytes = max(self.state.peak_work_bytes, actual)
        self.state.peak_reserved_bytes = max(self.state.peak_reserved_bytes, actual+unused+extra)

    def check(self):
        with fixed_lock(self.lock):
            self._check()

    def grant(self, path, position, needed):
        path = str(path)
        with fixed_lock(self.lock):
            prior = self.entries.pop(path, None)
            try:
                actual,unused=self.usage()
                available=min(self.max_work_bytes-actual-unused-self.metadata_reserve_bytes,
                    shutil.disk_usage(self.state.root).free-unused-self.min_free_bytes-self.metadata_reserve_bytes)
                amount=min(max(self.write_credit_bytes,needed),available)
                if amount<needed:
                    self._check(needed)
                self._check(amount)
                self.entries[path] = position+amount
                return amount
            except BaseException:
                if prior is not None:
                    self.entries[path] = prior
                raise

    def release(self, path):
        with fixed_lock(self.lock):
            self.entries.pop(str(path), None)

    def reclaim_after_join(self):
        with fixed_lock(self.lock):
            self.entries.clear()


class WriterBudget:
    def __init__(self, ledger, path, stream):
        self.ledger, self.path, self.stream = ledger, Path(path), stream
        self.credit = 0
        self.last_check = time.monotonic()

    def spend(self, count):
        if count > self.credit:
            self.credit = self.ledger.grant(self.path, self.stream.tell(), count)
            self.last_check = time.monotonic()
        elif time.monotonic()-self.last_check >= 2:
            self.ledger.check()
            self.last_check = time.monotonic()
        self.credit -= count

    def close(self):
        self.ledger.release(self.path)


def write_index(path, value, ledger=None, *, compressed=False):
    from .common import atomic_json, canonical, sync_file, sync_directory
    if ledger is None:
        return atomic_json(path, value, compressed=compressed)
    # Reserve the complete bounded index before atomic_json creates its temp file.
    raw=canonical(value)+b'\n'
    require(len(raw) <= 32*1024*1024, 'RESOURCE_LIMIT', 'task index exceeds32MiB')
    if compressed:
        from compression import zstd
        raw = zstd.compress(raw, level=9)
    temporary=Path(str(path)+'.append-index.tmp')
    ledger.grant(str(temporary), temporary.stat().st_size if temporary.exists() else 0, len(raw))
    try:
        with temporary.open('wb') as stream:
            stream.write(raw)
            stream.flush()
            from .metrics import fsync
            fsync(stream.fileno(),category='index_fsync')
        os.replace(temporary,path)
        sync_directory(Path(path).parent)
    finally:
        ledger.release(str(temporary))
