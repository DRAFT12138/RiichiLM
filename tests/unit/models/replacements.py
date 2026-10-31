"""Small real adapters and private-RNG/fault models for P10/P06 reuse."""
from dataclasses import replace
import torch
from torch import nn

from mahjong.contracts import Batch, ComponentDescriptor, MahjongError, MetricStat, StepResult, clone_state
from mahjong.models import MockMLP


class ScalarModel(nn.Module):
    def __init__(self, *, object_payload=False, input_schema='synthetic-regression/v1'):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.object_payload = object_payload
        self.input_schema = input_schema

    def descriptor(self):
        return ComponentDescriptor('model', 'test.scalar-model/v1', 1, self.input_schema,
                                   f'scalar/{self.object_payload}', None, 1, 'state_dict',
                                   ('cpu/32-true', 'cuda:0/32-true'))

    def _values(self, batch):
        return batch.payload.values if self.object_payload else batch.payload['x']

    def prepare_batch(self, batch, device):
        values = self._values(batch).to(device)
        payload = type(batch.payload)(values) if self.object_payload else {'x': values}
        return Batch(payload, batch.sample_ids, batch.sample_count, batch.input_schema)

    def forward(self, batch, *, phase):
        inputs = self._values(batch).sum(dim=-1)
        squared = (self.weight * inputs - (inputs + 0.25)).square()
        return StepResult(squared.mean(), {'loss': MetricStat(float(squared.detach().double().sum()),
                                                            batch.sample_count, 'squared_target')})

    def auxiliary_state_dict(self):
        return {}

    def load_auxiliary_state_dict(self, state):
        if state:
            raise MahjongError('CHECKPOINT_INVALID', 'scalar auxiliary state must be empty', 'restoring')

    def reset_auxiliary_rng(self, seed):
        pass


class PrivateGeneratorModel(MockMLP):
    """Consumes its own Generator in both train/eval, with nested auxiliary state."""
    def __init__(self, **options):
        super().__init__(**options)
        self.private = torch.Generator(device='cpu').manual_seed(self.private_seed)
        self.owned = {'nested': [1, {7: torch.tensor([3.])}]}

    def descriptor(self):
        return replace(super().descriptor(), implementation_id='test.private-generator-model/v1',
                       config_fingerprint=super().descriptor().config_fingerprint + '/private-noise-v1')

    def predict(self, batch):
        result = super().predict(batch)
        noise = torch.randn(result.shape, generator=self.private, device='cpu') * 0.05
        return result + noise.to(result.device)

    def auxiliary_state_dict(self):
        return clone_state({'version': 1, 'private_rng': self.private.get_state(), 'owned': self.owned})

    def load_auxiliary_state_dict(self, state):
        owned = clone_state(state)
        self.private.set_state(owned['private_rng'])
        self.owned = owned['owned']

    def reset_auxiliary_rng(self, seed):
        self.private.manual_seed(seed)


class AllSkipModel(ScalarModel):
    def forward(self, batch, *, phase):
        if phase == 'train':
            return StepResult(None, {}, 'no_valid_targets')
        return StepResult(None, {'loss': MetricStat(0., 0, 'squared_target')})


class NonfiniteLossModel(ScalarModel):
    def __init__(self, *, bad_value=float('nan'), fail_at=1, **options):
        super().__init__(**options)
        self.bad_value = bad_value
        self.fail_at = fail_at
        self.calls = 0

    def forward(self, batch, *, phase):
        self.calls += 1
        if self.calls == self.fail_at:
            return StepResult(self.weight * self.bad_value,
                              {'loss': MetricStat(0., batch.sample_count, 'squared_target')})
        return super().forward(batch, phase=phase)
