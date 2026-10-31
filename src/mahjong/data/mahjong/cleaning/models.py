"""Preparation objects. Freeze owned copies before later engine mutations."""
from mahjong.envs.round_types import VISIBLE, FrozenAnchor, PreparedView, PreparedRound
from dataclasses import dataclass, field
from copy import deepcopy

from mahjong.tokenizer import Event, HandMeldState


@dataclass
class GameResult:
    game_id: str | None
    native_id: str | None
    rule_ref: dict | None
    status: str
    rounds: list[PreparedRound] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
