"""Owned round snapshots shared by replay and preprocessing."""
from dataclasses import dataclass, field
from copy import deepcopy

from mahjong.tokenizer import Event, HandMeldState


VISIBLE = frozenset(('EV_DRAW', 'EV_DISCARD', 'EV_CHI', 'EV_PON',
    'EV_DAIMINKAN', 'EV_ANKAN_ATTEMPT', 'EV_KAKAN_ATTEMPT', 'EV_KAN_COMMIT',
    'EV_DORA_REVEAL', 'EV_RIICHI_DECLARE', 'EV_RIICHI_ACCEPT'))


@dataclass(frozen=True)
class FrozenAnchor:
    window_id: int
    trigger_id: int
    cutoff_event_id: int
    state: HandMeldState

    @classmethod
    def capture(cls, window_id, trigger_id, cutoff_event_id, state):
        return cls(window_id, trigger_id, cutoff_event_id, deepcopy(state))


@dataclass
class PreparedView:
    viewer: int
    header: dict
    history: list[Event]
    anchors: list[FrozenAnchor]
    valid_through_event_id: int
    invalid_reason: dict | None = None


@dataclass
class PreparedRound:
    round_id: int
    event_ids: list[int]
    ledger: list[dict]
    windows: list[dict]
    views: list[PreparedView]
    complete: bool
    facts: dict = field(default_factory=dict)
    aka_counts: list | None = None
