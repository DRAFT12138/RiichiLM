"""Manual and object-payload providers for P10 contract/decoupling tests."""
from dataclasses import dataclass
import hashlib
import uuid

import torch

from mahjong.contracts import Batch, BatchLease, ComponentDescriptor, MahjongError, clone_state


@dataclass(frozen=True)
class ObjectPayload:
    values: torch.Tensor


class FixedSource:
    """Ordered manual rows; implements both finite eval and committed training."""
    def __init__(self, *, values=None, batch_size=2, object_payload=False,
                 input_schema='synthetic-regression/v1'):
        self.values = (torch.tensor([[1., 2.], [3., 4.], [-1., 0.], [0.5, 1.]], dtype=torch.float32)
                       if values is None else values.detach().cpu().clone())
        self.batch_size = batch_size
        self.object_payload = object_payload
        self.input_schema = input_schema
        self.cursor = 0
        self.epoch = 0
        self.lease = None
        self.fingerprint = hashlib.sha256(self.values.numpy().tobytes()).hexdigest()

    def descriptor(self):
        return ComponentDescriptor('data', 'test.manual-source/v1', 1, self.input_schema,
                                   f'manual/{self.batch_size}/{self.object_payload}', self.fingerprint,
                                   None, 'committed_batch', ('cpu/32-true', 'cuda:0/32-true'))

    def _batch(self, start):
        end = min(start + self.batch_size, len(self.values))
        values = self.values[start:end].clone()
        payload = ObjectPayload(values) if self.object_payload else {'x': values}
        return Batch(payload, tuple(f'{self.fingerprint}/manual/{i}' for i in range(start, end)),
                     end-start, self.input_schema)

    def acquire(self):
        if self.lease is not None:
            raise MahjongError('CONTRACT_MISMATCH', 'manual source lease already open', 'training')
        if len(self.values) == 0:
            raise MahjongError('EMPTY_TRAIN_DATA', 'manual source is empty', 'training')
        self.lease = BatchLease(self._batch(self.cursor), uuid.uuid4().hex)
        return self.lease

    def commit(self, token):
        if self.lease is None or token != self.lease.token:
            raise MahjongError('CONTRACT_MISMATCH', 'manual source lease token differs', 'training')
        self.cursor += self.lease.batch.sample_count
        if self.cursor == len(self.values):
            self.cursor = 0
            self.epoch += 1
        self.lease = None

    def has_open_lease(self):
        return self.lease is not None

    def state_dict(self):
        return clone_state({'version': 1, 'fingerprint': self.fingerprint,
                            'cursor': self.cursor, 'epoch': self.epoch})

    def load_state_dict(self, state):
        if self.lease is not None or state['version'] != 1 or state['fingerprint'] != self.fingerprint:
            raise MahjongError('CHECKPOINT_INVALID', 'manual source state differs', 'restoring')
        if type(state['cursor']) is not int or not 0 <= state['cursor'] < len(self.values):
            raise MahjongError('CHECKPOINT_INVALID', 'manual cursor invalid', 'restoring')
        owned = clone_state(state)
        self.cursor, self.epoch = owned['cursor'], owned['epoch']

    def iter_batches(self):
        for start in range(0, len(self.values), self.batch_size):
            yield self._batch(start)


class ReadFailureSource(FixedSource):
    """Transient fault control is outside the provider's persisted math state."""
    def __init__(self, *, fail_at=1, **options):
        super().__init__(**options)
        self.fail_at = fail_at
        self.calls = 0

    def acquire(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise MahjongError('DATA_READ_FAILED', f'injected read failure at acquire {self.calls}', 'training')
        return super().acquire()
