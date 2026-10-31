"""Necessary synchronous local records and the optional TensorBoard backend."""
import os
from pathlib import Path
import tempfile

from mahjong.contracts import canonical_json


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'wb') as handle:
            handle.write((canonical_json(value) + '\n').encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class JsonlSink:
    def __init__(self, path):
        # A new identity never overwrites an existing attempt's raw records.
        self.handle = Path(path).open('xb', buffering=0)

    def append(self, value):
        data = (canonical_json(value) + '\n').encode()
        if self.handle.write(data) != len(data):
            raise OSError('Short local JSONL write')

    def flush(self, durable=False):
        self.handle.flush()
        if durable:
            os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


def default_writer_factory(**kwargs):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(**kwargs)


class TensorBoardSink:
    def __init__(self, path, *, enabled, flush_seconds, factory, degraded):
        self.writer = None
        self.disabled = not enabled
        self.failed = False
        self.closed = False
        self.degraded = degraded
        if enabled:
            try:
                self.writer = factory(log_dir=str(path), max_queue=256, flush_secs=flush_seconds)
            except Exception as exc:
                self._fail('initialize', exc)

    def _fail(self, operation, exc):
        self.disabled = True
        if not self.failed:
            self.failed = True
            self.degraded(operation, str(exc))

    def scalar(self, tag, value, step):
        if self.disabled or value is None:
            return
        try:
            self.writer.add_scalar(tag, value, step, new_style=True, double_precision=True)
        except Exception as exc:
            self._fail('add_scalar', exc)

    def flush(self):
        if not self.disabled and not self.closed:
            try:
                self.writer.flush()
            except Exception as exc:
                self._fail('flush', exc)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as exc:
                self._fail('close', exc)
