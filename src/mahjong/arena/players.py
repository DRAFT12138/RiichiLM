"""Players choose against immutable, viewer-specific decisions."""

from dataclasses import dataclass
import random
import time
from typing import Any, Protocol

from mahjong.contracts import MahjongError


def check_deadline(deadline, cancel=None):
    if cancel is not None and cancel():
        raise MahjongError("CANCELLED", "玩家决策已取消")
    if time.monotonic() >= deadline:
        raise MahjongError("DEADLINE_EXCEEDED", "玩家决策超时")


@dataclass(frozen=True)
class PlayerChoice:
    key: Any
    execution_ref: str | None = None
    generation: Any = None
    summary: str | None = None


class Player(Protocol):
    def choose(self, view, *, deadline, cancel=None) -> PlayerChoice: ...


class RandomPlayer:
    def __init__(self, seed):
        self.random = random.Random(seed)

    def choose(self, view, *, deadline, cancel=None):
        check_deadline(deadline, cancel)
        return PlayerChoice(view.key, self.random.choice(view.legal_moves).execution_ref)


class ModelPlayer:
    """Preserve the checkpoint policy's generated-action binding and probabilities."""

    def __init__(self, policy, *, seed=0):
        import torch

        self.policy = policy
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def choose(self, view, *, deadline, cancel=None):
        from mahjong.inference.policy import DecodeOptions

        result = self.policy.generate(
            self.policy.prepare(view), options=DecodeOptions(strategy="categorical"),
            generator=self.generator, deadline_monotonic=deadline, cancel=cancel
        )
        return PlayerChoice(view.key, generation=result)


def load_model_policy(path, *, kind='checkpoint', device='cpu'):
    """Use one model execution policy for Web and headless model players."""
    from mahjong.inference.policy import PolicyInference

    precision = 'bf16-mixed' if device == 'cuda:0' else '32-true'
    if kind == 'weights':
        loader = PolicyInference.from_weights
    elif kind == 'checkpoint':
        loader = PolicyInference.from_checkpoint
    else:
        raise MahjongError('CONFIG_INVALID', 'unknown model player source')
    return loader(path, device=device, precision=precision)
