"""A small real MLP and its complete regression learning adapter."""
import hashlib
import math

import torch
from torch import nn

from mahjong.contracts import (Batch, ComponentDescriptor, MahjongError, MetricStat,
                              State, StepResult, canonical_json)

INPUT_SCHEMA = 'synthetic-regression/v1'


class MockMLP(nn.Module):
    def __init__(self, *, input_dim: int = 8, hidden_dim: int = 32, dropout: float = 0.0,
                 target_rule: str = 'linear-teacher/v1', init_seed: int, private_seed: int,
                 input_schema: str = INPUT_SCHEMA):
        super().__init__()
        for name, value in [('input_dim', input_dim), ('hidden_dim', hidden_dim)]:
            if type(value) is not int or value <= 0:
                raise MahjongError('CONFIG_INVALID', f'model.{name}: expected positive integer')
        if type(dropout) not in (float, int) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise MahjongError('CONFIG_INVALID', 'model.dropout: expected finite number in [0,1)')
        if target_rule != 'linear-teacher/v1' or input_schema != INPUT_SCHEMA:
            raise MahjongError('CONFIG_INVALID', 'model.target_rule/input_schema: unsupported semantics')
        for name, seed in [('init_seed', init_seed), ('private_seed', private_seed)]:
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise MahjongError('CONFIG_INVALID', f'model.{name}: expected seed in [0,2**63)')
        self.input_dim = input_dim
        self.input_schema = input_schema
        self.private_seed = private_seed
        # Only the CPU default stream is temporarily seeded; caller RNG is restored.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(init_seed)
            self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                         nn.Dropout(float(dropout)), nn.Linear(hidden_dim, 1))
        fingerprint = hashlib.sha256(canonical_json({
            'input_dim': input_dim, 'hidden_dim': hidden_dim, 'dropout': float(dropout),
            'input_schema': input_schema, 'target_rule': target_rule,
        }).encode('utf-8')).hexdigest()
        self._descriptor = ComponentDescriptor('model', 'majang.mock_mlp/v1', 1, input_schema,
                                                fingerprint, None, 1, 'state_dict',
                                                ('cpu/32-true', 'cuda:0/32-true'))

    def descriptor(self) -> ComponentDescriptor:
        return self._descriptor

    def _input(self, batch: Batch) -> torch.Tensor:
        if batch.input_schema != self.input_schema:
            raise MahjongError('CONTRACT_MISMATCH', 'MockMLP input_schema differs', 'training')
        if type(batch.payload) is not dict or set(batch.payload) != {'x'}:
            raise MahjongError('CONTRACT_MISMATCH', 'MockMLP payload must contain only x', 'training')
        x = batch.payload['x']
        if (not isinstance(x, torch.Tensor) or x.ndim != 2
                or x.shape != (batch.sample_count, self.input_dim) or x.dtype != torch.float32):
            raise MahjongError('CONTRACT_MISMATCH', 'MockMLP x must be float32 [batch,input_dim]', 'training')
        if not bool(torch.isfinite(x).all().item()):
            raise MahjongError('CONTRACT_MISMATCH', 'MockMLP x contains nonfinite values', 'training')
        return x

    def prepare_batch(self, batch: Batch, device: torch.device) -> Batch:
        x = self._input(batch)
        return Batch({'x': x.to(device=device, dtype=torch.float32)}, batch.sample_ids,
                     batch.sample_count, batch.input_schema)

    def predict(self, batch: Batch) -> torch.Tensor:
        return self.network(self._input(batch)).squeeze(-1)

    def targets(self, batch: Batch) -> torch.Tensor:
        x = self._input(batch)
        weights = torch.arange(1, self.input_dim + 1, dtype=x.dtype, device=x.device) / self.input_dim
        return (x * weights).sum(dim=-1) + 0.25

    def forward(self, batch: Batch, *, phase: str) -> StepResult:
        if phase not in ('train', 'eval'):
            raise MahjongError('CONTRACT_MISMATCH', 'MockMLP phase must be train/eval', 'training')
        prediction = self.predict(batch)
        error = prediction - self.targets(batch)
        squared = error.square()
        squared_sum = float(squared.detach().double().sum().item())
        absolute_sum = float(error.detach().abs().double().sum().item())
        metrics = {
            'loss': MetricStat(squared_sum, batch.sample_count, 'squared_target'),
            'mse': MetricStat(squared_sum, batch.sample_count, 'squared_target'),
            'mae': MetricStat(absolute_sum, batch.sample_count, 'target'),
        }
        return StepResult(squared.mean(), metrics)

    def auxiliary_state_dict(self) -> State:
        return {}

    def load_auxiliary_state_dict(self, state: State) -> None:
        if type(state) is not dict or state:
            raise MahjongError('CHECKPOINT_INVALID', 'MockMLP auxiliary state must be empty', 'restoring')

    def reset_auxiliary_rng(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise MahjongError('CONTRACT_MISMATCH', 'MockMLP reset seed must be in [0,2**63)', 'evaluating')
        # The baseline MLP has no private mutable random stream.
