"""One spawned process owns the whole table and all loaded policies."""

from dataclasses import asdict
from collections import deque
import hashlib
import math
import resource
import time

from mahjong.contracts import MahjongError
from mahjong.inference.policy import PolicyInference
from mahjong.arena.bridge import ArenaDecisionBridge as WebDecisionBridge, profile_for
from mahjong.arena.llm import LLMConfig, LLMPlayer, plain
from mahjong.arena.loop import DecisionLoop
from mahjong.arena.players import ModelPlayer, RandomPlayer


class TableEngine:
    def __init__(
        self, config, *, device="cpu", decision_timeout=120, emit=lambda _: None
    ):
        self.config, self.emit = config, emit
        self.mode = config["mode"]
        self.spectator = config.get("spectator", False)
        self.viewer, self.paused, self.playback_interval = 0, False, 0.75
        self.decision_notes, self.pending_notes = deque(maxlen=64), []
        self.decision_timeout = decision_timeout
        self.timings = dict(
            load_seconds=0.0, last_decision_seconds=None, peak_rss_mib=0.0
        )
        start = time.monotonic()
        cache = {}
        controllers = [] if self.spectator else [("human", None)]
        players = [] if self.spectator else [None]
        self.controller_items = config["opponents"] if self.spectator else [None, *config["opponents"]]
        training_profile = profile_for("hanchan")
        for item in config["opponents"]:
            if item is None:
                controllers.append(("random", None))
                players.append(None)  # initialized below with the existing seat seed
                continue
            if item.get("kind") == "llm":
                controllers.append(("llm", None))
                players.append(LLMPlayer(LLMConfig(**item["config"]), explain=self.spectator))
                continue
            key = (item["path"], item["manifest_hash"])
            if key not in cache:
                if item.get("kind") == "weights":
                    policy = PolicyInference.from_weights(item["path"], device=device)
                else:
                    policy = PolicyInference.from_checkpoint(item["path"], device=device)
                    actual = policy.source
                    if (
                        any(
                            actual[k] != item[k]
                            for k in (
                                "run_id",
                                "checkpoint_id",
                                "attempt_id",
                                "manifest_hash",
                            )
                        )
                        or actual["global_step"] != item["step"]
                    ):
                        raise MahjongError(
                            "CHECKPOINT_INVALID", "实际加载发布物与选择不一致，请刷新重选"
                        )
                    policy.verify_training_profile(training_profile)
                cache[key] = policy
            controllers.append(("model", cache[key]))
            seat = len(controllers) - 1
            seed = int.from_bytes(hashlib.sha256(f"web-play/model-sampling/v1:{config['seed']}:{seat}".encode()).digest()[:8], "big")
            players.append(ModelPlayer(cache[key], seed=seed))
        self.controllers = controllers
        self.bridge = WebDecisionBridge(
            profile_for(self.mode), model_binding_digest="web-non-model"
        )
        self.bridge.bind_controllers(controllers)
        self.bridge.reset(seed=config["seed"])
        self.timings["load_seconds"] = time.monotonic() - start
        for p in range(4):
            if controllers[p][0] != "random":
                continue
            players[p] = RandomPlayer(
                int.from_bytes(
                    hashlib.sha256(
                        f"web-play/random/v1:{config['seed']}:{p}".encode()
                    ).digest(),
                    "big",
                )
            )
        self.loop = DecisionLoop(self.bridge, players, decision_timeout=decision_timeout,
                                 on_thinking=self._thinking, on_decision=self._record_decision)
        self.waiting = None
        self.status = "playing"
        self.publish()

    def publish(self, *, thinking=None):
        table = self.bridge.public_table(self.viewer)
        self.timings["peak_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        controllers = [
                dict(
                    kind=kind,
                    checkpoint_id=None
                    if model is None
                    else model.source["checkpoint_id"],
                    identity=None
                    if item is None or kind == "llm"
                    else {
                        k: item[k]
                        for k in (
                            "run_id",
                            "attempt_id",
                            "checkpoint_id",
                            "step",
                            "manifest_hash",
                            "state_sha256",
                        )
                    },
                    name=item["config"]["name"] if kind == "llm" else None,
                    model=item["config"]["model"] if kind == "llm" else None,
                    sampling="categorical" if kind == "model" else "uniform" if kind == "random" else None,
                )
                for (kind, model), item in zip(
                    self.controllers, self.controller_items
                )
            ]
        table.update(
            state=self.status, mode=self.mode, thinking=thinking,
            timings=dict(self.timings), controllers=controllers, opponents=controllers[1:],
            spectator=self.spectator, viewer=self.viewer, paused=self.paused,
            playback_interval=self.playback_interval,
            decision_notes=[dict(n) for n in self.decision_notes if n["seat"] == self.viewer] if self.spectator else [],
            cross_mode=self.mode == "tonpuu"
            and any(m is not None for _, m in self.controllers),
        )
        if thinking is not None or self.spectator:
            table["candidates"] = []
            table["window"] = None
        self.emit(
            dict(
                type="snapshot",
                table=table,
                timeout=self.decision_timeout if thinking is not None else None,
            )
        )
        return table

    def _thinking(self, seat):
        if self.controllers[seat][0] in ("model", "llm"):
            self.publish(thinking=seat)

    def _record_decision(self, view, choice, result):
        if not self.spectator:
            return
        kind = self.controllers[view.key.viewer][0]
        summary = choice.summary
        if kind == "random":
            summary = f"从{len(view.legal_moves)}个合法候选中等概率采样。"
        probability = (math.exp(result.generation.behavior_action_logprob) if kind == "model"
                       else 1 / len(view.legal_moves) if kind == "random" else None)
        self.pending_notes.append(dict(
            seat=view.key.viewer, round_id=view.key.window.round_id,
            revision=view.key.window.state_revision, candidate_id=result.execution_ref,
            action=plain(result.action), candidate_count=len(view.legal_moves),
            summary=summary, summary_source="model_summary" if kind == "llm" and summary else "random_policy" if kind == "random" else None,
            sampling="categorical" if kind == "model" else "uniform" if kind == "random" else "llm",
            selected_probability=probability, seconds=self.loop.last_decision_seconds,
        ))

    def drive(self, *, force=False):
        """Freeze a complete window, compute AI choices, then wait for the human."""
        while self.status == "playing":
            if self.spectator and self.paused and not force:
                return
            views = self.bridge.decision_views()
            if not views:
                self.status = "finished" if self.mode == "single" else "round_result"
                self.publish()
                return
            self.pending_notes = []
            pending = self.loop.collect()
            self.timings["last_decision_seconds"] = self.loop.last_decision_seconds
            if pending.humans:
                self.waiting = pending
                self.publish()
                return
            self.loop.apply(pending)
            self.decision_notes.extend(self.pending_notes)
            self.pending_notes = []
            if self.spectator and not self.bridge.decision_views():
                self.status = "finished" if self.mode == "single" else "round_result"
            self.publish()
            if self.spectator:
                return

    def action(self, window, candidate):
        if self.waiting is None or self.status != "playing":
            raise MahjongError("STALE_DECISION", "当前没有待提交的真人选择")
        pending = self.waiting
        view = pending.humans[0]
        if asdict(view.key.window) != window:
            raise MahjongError("STALE_DECISION", "动作窗口已过期")
        decision = self.bridge.select_legal(view.key, candidate)
        pending.decisions[0] = decision
        self.loop.apply(pending)
        self.waiting = None
        self.publish()
        self.drive()

    def continue_round(self, round_id, revision):
        current = self.bridge.public_table(0)
        if self.status != "round_result" or (round_id, revision) != (
            current["round_id"],
            current["revision"],
        ):
            raise MahjongError("STALE_DECISION", "局末结果已过期")
        result = self.bridge.next_round()
        self.status = "finished" if result.status == "game_ended" else "playing"
        self.publish()
        self.drive()

    def control(self, control, value=None):
        if not self.spectator or self.status not in ("playing", "round_result"):
            raise MahjongError("STALE_DECISION", "当前不能控制观战")
        if control in ("pause", "resume", "step"):
            if self.status != "playing":
                raise MahjongError("STALE_DECISION", "请先进入下一局")
            self.paused = control != "resume"
            if control == "step":
                self.drive(force=True)
                return
        elif control == "view" and type(value) is int and 0 <= value < 4:
            self.viewer = value
        elif control == "speed" and type(value) in (int, float) and math.isfinite(value) and 0.1 <= value <= 5:
            self.playback_interval = value
        else:
            raise MahjongError("CONTRACT_MISMATCH", "观战操作或参数无效")
        self.publish()


def run_worker(connection, config, device, decision_timeout):
    try:
        import torch

        torch.set_num_threads(1)
        engine = TableEngine(
            config,
            device=device,
            decision_timeout=decision_timeout,
            emit=connection.send,
        )
        engine.drive()
        next_tick = time.monotonic() + engine.playback_interval
        while engine.status != "finished":
            automatic = engine.spectator and not engine.paused and engine.status == "playing"
            wait = max(0, next_tick - time.monotonic()) if automatic else None
            if not connection.poll(wait):
                engine.drive()
                next_tick = time.monotonic() + engine.playback_interval
                continue
            message = connection.recv()
            try:
                if message["operation"] == "action":
                    engine.action(message["window"], message["candidate"])
                elif message["operation"] == "continue":
                    engine.continue_round(message["round_id"], message["revision"])
                elif message["operation"] == "control":
                    engine.control(message["control"], message.get("value"))
                else:
                    raise MahjongError("STALE_DECISION", "未知牌桌操作")
                connection.send(dict(type="ack", request_id=message["request_id"]))
                next_tick = time.monotonic() + engine.playback_interval
            except MahjongError as exc:
                if exc.code not in ("STALE_DECISION", "CONTRACT_MISMATCH"):
                    raise
                connection.send(
                    dict(
                        type="rejected",
                        request_id=message["request_id"],
                        error=str(exc),
                    )
                )
                engine.publish()
    except EOFError:
        pass
    except Exception as exc:
        # No omniscient state or traceback in IPC/browser errors.
        connection.send(dict(type="failed", error=str(exc)))
    finally:
        connection.close()
