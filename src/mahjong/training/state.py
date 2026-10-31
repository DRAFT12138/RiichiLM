"""Independent progress and RNG values shared by the runtime and loop."""

from dataclasses import replace
import random

import numpy as np
import torch

from mahjong.contracts import RngState, TrainProgress, clone_state


def copy_progress(progress: TrainProgress) -> TrainProgress:
    # All nested values are immutable PendingValidation or scalar values.
    return replace(progress)


def encode_numpy_rng(state: tuple) -> tuple:
    algorithm, keys, position, has_gauss, cached = state
    return algorithm, [int(value) for value in keys], int(position), int(has_gauss), float(cached)


def decode_numpy_rng(state: tuple) -> tuple:
    algorithm, keys, position, has_gauss, cached = state
    return algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached


def seeded_rng(seed: int, template: RngState) -> RngState:
    """Create independent seeded streams without advancing the global RNG."""
    numpy = encode_numpy_rng(np.random.RandomState(seed % 2**32).get_state()) if template.numpy is not None else None
    device = torch.Generator(device="cuda:0").manual_seed(seed).get_state().cpu().clone() if template.torch_device is not None else None
    return RngState(clone_state(random.Random(seed).getstate()), numpy,
                    torch.Generator(device="cpu").manual_seed(seed).get_state().clone(), device, {})
