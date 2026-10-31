"""A committed cursor: fetching does not advance sampling or private RNG."""
import torch

from mahjong.contracts import MahjongError, State, clone_state


class CommittedSampler:
    def __init__(self, *, size: int, batch_size: int, shuffle: bool,
                 drop_last: bool, seed: int, data_fingerprint: str,
                 sampling_fingerprint: str):
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data_fingerprint = data_fingerprint
        self.sampling_fingerprint = sampling_fingerprint
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.epoch = 0
        self.next_index = 0
        self.dropped_samples = 0
        self.permutation = self._new_permutation()

    def _new_permutation(self) -> torch.Tensor:
        return (torch.randperm(self.size, generator=self.generator)
                if self.shuffle else torch.arange(self.size, dtype=torch.int64))

    def indices(self) -> list[int]:
        end = min(self.next_index + self.batch_size, self.size)
        return self.permutation[self.next_index:end].tolist()

    def commit(self) -> None:
        self.next_index += min(self.batch_size, self.size - self.next_index)
        remaining = self.size - self.next_index
        if remaining == 0 or (self.drop_last and remaining < self.batch_size):
            self.dropped_samples += remaining
            self.epoch += 1
            self.next_index = 0
            self.permutation = self._new_permutation()

    def state_dict(self) -> State:
        return clone_state({
            'version': 1,
            'data_fingerprint': self.data_fingerprint,
            'sampling_fingerprint': self.sampling_fingerprint,
            'epoch': self.epoch,
            'next_index': self.next_index,
            'permutation': self.permutation,
            'sampler_rng': self.generator.get_state(),
            'dropped_samples': self.dropped_samples,
        })

    def load_state_dict(self, state: State) -> None:
        required = {'version', 'data_fingerprint', 'sampling_fingerprint', 'epoch',
                    'next_index', 'permutation', 'sampler_rng', 'dropped_samples'}
        def reject(message: str) -> None:
            raise MahjongError('CHECKPOINT_INVALID', 'Synthetic sampler: ' + message, 'restoring')

        if type(state) is not dict or set(state) != required:
            reject('missing or unknown state fields')
        if type(state['version']) is not int or state['version'] != 1:
            reject('unsupported version')
        if state['data_fingerprint'] != self.data_fingerprint:
            reject('data_fingerprint differs')
        if state['sampling_fingerprint'] != self.sampling_fingerprint:
            reject('sampling_fingerprint differs')
        for key in ('epoch', 'next_index', 'dropped_samples'):
            if type(state[key]) is not int or state[key] < 0:
                reject(key + ' must be a nonnegative integer')
        limit = ((self.size // self.batch_size) * self.batch_size
                 if self.drop_last else self.size)
        if state['next_index'] >= limit or state['next_index'] % self.batch_size:
            reject('next_index is not a normalized batch boundary')
        permutation = state['permutation']
        if (not isinstance(permutation, torch.Tensor) or permutation.device.type != 'cpu'
                or permutation.dtype != torch.int64 or permutation.shape != (self.size,)
                or not torch.equal(permutation.sort().values, torch.arange(self.size))):
            reject('permutation must contain every sample index once as CPU int64')
        rng = state['sampler_rng']
        if (not isinstance(rng, torch.Tensor) or rng.device.type != 'cpu'
                or rng.dtype != torch.uint8 or rng.ndim != 1):
            reject('sampler_rng must be a CPU byte vector')
        owned = clone_state(state)
        generator = torch.Generator(device='cpu')
        try:
            generator.set_state(owned['sampler_rng'])
        except RuntimeError as exc:
            raise MahjongError('CHECKPOINT_INVALID', 'Synthetic sampler: invalid RNG state',
                              'restoring') from exc
        self.epoch = owned['epoch']
        self.next_index = owned['next_index']
        self.dropped_samples = owned['dropped_samples']
        self.permutation = owned['permutation']
        self.generator = generator
