"""Batch construction owned by the synthetic data provider."""
from collections.abc import Callable, Sequence

import torch

from mahjong.contracts import Batch


def make_batch(indices: Sequence[int], sample: Callable[[int], tuple[str, torch.Tensor]],
               input_schema: str) -> Batch:
    samples = [sample(index) for index in indices]
    return Batch({'x': torch.stack([value for _, value in samples])},
                 tuple(identity for identity, _ in samples), len(samples), input_schema)
