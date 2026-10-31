"""Freeze all responses before applying any of them; shared by Web and headless."""

from dataclasses import dataclass
import math
import time

from mahjong.contracts import MahjongError
from .players import check_deadline


@dataclass
class PendingWindow:
    views: tuple
    decisions: dict
    humans: tuple
    response: bool


class DecisionLoop:
    def __init__(self, bridge, players, *, decision_timeout=120, on_thinking=lambda _: None,
                 on_decision=lambda view, choice, result: None):
        if len(players) != 4 or not math.isfinite(decision_timeout) or decision_timeout <= 0:
            raise MahjongError("CONFIG_INVALID", "需要四席玩家及有限正数决策时限")
        self.bridge, self.players = bridge, tuple(players)
        self.decision_timeout, self.on_thinking = decision_timeout, on_thinking
        self.on_decision = on_decision
        self.last_decision_seconds = None
        self.decisions = [0] * 4

    def select(self, view, *, cancel=None):
        seat = view.key.viewer
        player = self.players[seat]
        if player is None:
            return self.bridge.select_legal(view.key, view.legal_moves[0].execution_ref)
        self.on_thinking(seat)
        start = time.monotonic()
        deadline = start + self.decision_timeout
        choice = player.choose(view, deadline=deadline, cancel=cancel)
        check_deadline(deadline, cancel)
        if choice.key != view.key or (choice.generation is None) == (choice.execution_ref is None):
            raise MahjongError("CONTRACT_MISMATCH", "玩家结果与当前决策不一致")
        result = (
            self.bridge.select_legal(view.key, choice.execution_ref)
            if choice.generation is None
            else self.bridge.bind_result(view, choice.generation)
        )
        self.last_decision_seconds = time.monotonic() - start
        self.decisions[seat] += 1
        self.on_decision(view, choice, result)
        return result

    def collect(self, *, cancel=None):
        views = self.bridge.decision_views()
        response = self.bridge.is_response_window()
        decisions, humans = {}, []
        for view in views:
            if cancel is not None and cancel():
                raise MahjongError("CANCELLED", "对局已取消")
            seat = view.key.viewer
            if self.players[seat] is None and len(view.legal_moves) > 1:
                humans.append(view)
            elif response and len(view.legal_moves) == 1 and view.legal_moves[0].action.kind == "A_PASS":
                continue
            else:
                decisions[seat] = self.select(view, cancel=cancel)
        return PendingWindow(views, decisions, tuple(humans), response)

    def apply(self, pending):
        if not pending.views:
            raise MahjongError("STALE_DECISION", "当前没有待提交窗口")
        if pending.response:
            return self.bridge.apply_responses(pending.views[0].key.window, pending.decisions)
        return self.bridge.apply(pending.decisions[pending.views[0].key.viewer])
