"""Stateless per-identity samples and independent train/evaluation providers."""
import hashlib
import uuid

import torch

from mahjong.contracts import (BatchLease, ComponentDescriptor, MahjongError, State,
                              canonical_json, derive_seed)
from .batching import make_batch
from .sampling import CommittedSampler

INPUT_SCHEMA = 'synthetic-regression/v1'
SUPPORTED_EXECUTION = ('cpu/32-true', 'cuda:0/32-true')


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _integer(field: str, value: int, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value >= maximum):
        raise MahjongError('CONFIG_INVALID', f'{field}: unsupported integer value', 'preparing_data')


class SyntheticDataset:
    def __init__(self, *, seed: int, train_size: int, val_size: int, input_dim: int,
                 input_schema: str = INPUT_SCHEMA):
        _integer('seed (derived data_seed)', seed, 0, 2**63)
        _integer('data.train_size', train_size, 1)
        _integer('data.val_size', val_size, 0)
        _integer('data.input_dim', input_dim, 1)
        if input_schema != INPUT_SCHEMA:
            raise MahjongError('CONFIG_INVALID', 'data.input_schema: unsupported schema', 'preparing_data')
        self.seed = seed
        self.train_size = train_size
        self.val_size = val_size
        self.input_dim = input_dim
        self.input_schema = input_schema
        self.data_fingerprint = _fingerprint({
            'generator_version': 1, 'seed': seed, 'train_size': train_size,
            'val_size': val_size, 'input_dim': input_dim, 'distribution': 'normal',
        })

    def sample(self, split: str, index: int) -> tuple[str, torch.Tensor]:
        if split not in ('train', 'validation'):
            raise MahjongError('CONTRACT_MISMATCH', 'Unknown synthetic split', 'preparing_data')
        size = self.train_size if split == 'train' else self.val_size
        if type(index) is not int or not 0 <= index < size:
            raise MahjongError('CONTRACT_MISMATCH', 'Synthetic sample index out of range', 'preparing_data')
        seed = derive_seed(self.seed, f'sample/{split}/{index}/v1')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        value = torch.randn(self.input_dim, generator=generator, dtype=torch.float32)
        return f'{self.data_fingerprint}/{split}/{index}', value


def _batch_configuration(batch_size: int, num_workers: int) -> None:
    _integer('data.batch_size', batch_size, 1)
    if type(num_workers) is not int or num_workers != 0:
        raise MahjongError('CONFIG_INVALID', 'data.num_workers: only 0 is supported', 'preparing_data')


class SyntheticTrainSource:
    def __init__(self, *, seed: int, sampler_seed: int, train_size: int = 1024,
                 val_size: int = 256, input_dim: int = 8, batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False, num_workers: int = 0,
                 input_schema: str = INPUT_SCHEMA):
        _batch_configuration(batch_size, num_workers)
        _integer('sampler_seed', sampler_seed, 0, 2**63)
        if type(shuffle) is not bool or type(drop_last) is not bool:
            raise MahjongError('CONFIG_INVALID', 'data.shuffle/drop_last: expected bool', 'preparing_data')
        self.dataset = SyntheticDataset(seed=seed, train_size=train_size, val_size=val_size,
                                        input_dim=input_dim, input_schema=input_schema)
        if drop_last and train_size < batch_size:
            raise MahjongError('CONFIG_INVALID', 'data.drop_last: would create zero training batches', 'preparing_data')
        self.batch_size = batch_size
        sampling_fingerprint = _fingerprint({'sampler_seed': sampler_seed,
                                             'batch_size': batch_size,
                                             'shuffle': shuffle, 'drop_last': drop_last})
        self.sampler = CommittedSampler(size=train_size, batch_size=batch_size, shuffle=shuffle,
                                        drop_last=drop_last, seed=sampler_seed,
                                        data_fingerprint=self.dataset.data_fingerprint,
                                        sampling_fingerprint=sampling_fingerprint)
        self._descriptor = ComponentDescriptor(
            'data', 'majang.synthetic_train/v1', 1, input_schema,
            _fingerprint({'data': self.dataset.data_fingerprint, 'sampling': sampling_fingerprint,
                          'input_schema': input_schema, 'num_workers': num_workers}),
            self.dataset.data_fingerprint, None, 'committed_batch', SUPPORTED_EXECUTION)
        self._lease: BatchLease | None = None

    def descriptor(self) -> ComponentDescriptor:
        return self._descriptor

    def acquire(self) -> BatchLease:
        if self._lease is not None:
            raise MahjongError('CONTRACT_MISMATCH', 'A synthetic training lease is already open', 'training')
        batch = make_batch(self.sampler.indices(), lambda index: self.dataset.sample('train', index),
                           self.dataset.input_schema)
        self._lease = BatchLease(batch, uuid.uuid4().hex)
        return self._lease

    def commit(self, token: str) -> None:
        if self._lease is None or token != self._lease.token:
            raise MahjongError('CONTRACT_MISMATCH', 'Wrong or repeated synthetic lease token', 'training')
        self.sampler.commit()
        self._lease = None

    def has_open_lease(self) -> bool:
        return self._lease is not None

    def state_dict(self) -> State:
        return self.sampler.state_dict()

    def load_state_dict(self, state: State) -> None:
        if self._lease is not None:
            raise MahjongError('CONTRACT_MISMATCH', 'Cannot import with an open training lease', 'restoring')
        self.sampler.load_state_dict(state)


class SyntheticEvalSource:
    def __init__(self, *, seed: int, train_size: int = 1024, val_size: int = 256,
                 input_dim: int = 8, batch_size: int = 32, num_workers: int = 0,
                 input_schema: str = INPUT_SCHEMA):
        _batch_configuration(batch_size, num_workers)
        self.dataset = SyntheticDataset(seed=seed, train_size=train_size, val_size=val_size,
                                        input_dim=input_dim, input_schema=input_schema)
        self.batch_size = batch_size
        self._descriptor = ComponentDescriptor(
            'data', 'majang.synthetic_eval/v1', 1, input_schema,
            _fingerprint({'data': self.dataset.data_fingerprint, 'batch_size': batch_size,
                          'input_schema': input_schema, 'num_workers': num_workers}),
            self.dataset.data_fingerprint, None, 'committed_batch', SUPPORTED_EXECUTION)

    def descriptor(self) -> ComponentDescriptor:
        return self._descriptor

    def iter_batches(self):
        for start in range(0, self.dataset.val_size, self.batch_size):
            indices = range(start, min(start + self.batch_size, self.dataset.val_size))
            yield make_batch(indices, lambda index: self.dataset.sample('validation', index),
                             self.dataset.input_schema)
