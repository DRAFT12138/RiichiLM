"""Bounded headless matches using exactly the same players and loop as Web."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from mahjong.contracts import MahjongError
from .llm import LLMPlayer, PROMPT_VERSION, SYSTEM_PROMPT, load_llm_configs
from .loop import DecisionLoop
from .players import ModelPlayer, RandomPlayer


def run_match(bridge, players, *, seed, single_round=False, decision_timeout=120,
              max_windows=20000, cancel=None):
    """The caller binds four controllers before reset. No Web/session/IPC required."""
    if any(p is None for p in players) or type(max_windows) is not int or max_windows <= 0:
        raise MahjongError("CONFIG_INVALID", "离线评估需要四席自动玩家及正数窗口上限")
    loop = DecisionLoop(bridge, players, decision_timeout=decision_timeout)
    started = time.monotonic()
    rounds, windows = 0, 0
    status, error, scores, ranks, pool = "failed", None, None, None, None
    try:
        bridge.reset(seed=seed)
        while True:
            if cancel is not None and cancel():
                raise MahjongError("CANCELLED", "评估已取消")
            pending = loop.collect(cancel=cancel)
            if not pending.views:
                rounds += 1
                result = bridge.round_result()
                if single_round:
                    scores, pool = result["scores"], result["kyotaku"]
                    status = "complete"
                    break
                if bridge.next_round().status == "game_ended":
                    final = bridge.public_table(0)["final"]
                    scores, ranks, pool = final["scores"], final["ranks"], 0
                    status = "complete"
                    break
                continue
            if windows >= max_windows:
                raise MahjongError("EVALUATION_FAILED", "对局超过窗口预算")
            loop.apply(pending)
            windows += 1
    except MahjongError as exc:
        status = "cancelled" if exc.code == "CANCELLED" else "failed"
        error = dict(code=exc.code, message=exc.message)
    except Exception as exc:
        error = dict(code="EVALUATION_FAILED", message=f"评估执行失败（{type(exc).__name__}）")
    return dict(status=status, error=error, seed=seed, rounds=rounds, windows=windows,
                seconds=time.monotonic() - started, scores=scores, ranks=ranks,
                unclaimed_kyotaku=pool, decisions=loop.decisions,
                llm_stats=[deepcopy(getattr(p, "stats", None)) for p in players])


def evaluate_cli(options, *, cancel=None):
    from .bridge import ArenaDecisionBridge, profile_for
    from mahjong.inference.policy import PolicyInference, digest

    if (not math.isfinite(options.decision_timeout) or options.decision_timeout <= 0
            or options.max_windows <= 0 or len(set(options.seeds)) != len(options.seeds)
            or any(not 0 <= seed < 2**32 for seed in options.seeds)):
        raise MahjongError("CONFIG_INVALID", "评估需要有限正数时限、窗口预算及不重复的32位无符号种子")
    output = Path(options.output)
    if output.exists():
        raise MahjongError("CONFIG_INVALID", "评估报告已存在，请使用新路径")
    configs, policies, identities = load_llm_configs(options.llm_config), {}, []
    for spec in options.players:
        if spec == "random":
            identities.append(dict(kind="random"))
        elif spec in configs:
            identities.append(dict(kind="llm", **configs[spec].public(),
                                   config=asdict(configs[spec])))
        elif spec.startswith("checkpoint:"):
            path = spec.removeprefix("checkpoint:")
            if spec not in policies:
                policy = PolicyInference.from_checkpoint(path, device=options.device)
                policy.verify_training_profile(profile_for("hanchan"))
                policies[spec] = policy
            source = policies[spec].source
            identities.append(dict(kind="model", **{k: source[k] for k in (
                "run_id", "attempt_id", "checkpoint_id", "manifest_hash", "global_step")}))
        else:
            raise MahjongError("CONFIG_INVALID", "玩家必须为 random、已配置的 llm:<id> 或 checkpoint:<目录>")
    outcomes = []
    for seed in options.seeds:
        for rotation in range(4 if options.rotate else 1):
            order = [(seat - rotation) % 4 for seat in range(4)]
            players, controllers = [], []
            for seat, index in enumerate(order):
                spec = options.players[index]
                if spec == "random":
                    derived = int.from_bytes(hashlib.sha256(f"arena/random/v1:{seed}:{seat}".encode()).digest(), "big")
                    players.append(RandomPlayer(derived))
                    controllers.append(("random", None))
                elif spec in configs:
                    players.append(LLMPlayer(configs[spec]))
                    controllers.append(("llm", None))
                else:
                    derived = int.from_bytes(hashlib.sha256(f"arena/model-sampling/v1:{seed}:{seat}".encode()).digest()[:8], "big")
                    players.append(ModelPlayer(policies[spec], seed=derived))
                    controllers.append(("model", policies[spec]))
            bridge = ArenaDecisionBridge(profile_for(options.mode), model_binding_digest="arena-non-model")
            bridge.bind_controllers(controllers)
            outcome = run_match(bridge, players, seed=seed, single_round=options.mode == "single",
                                decision_timeout=options.decision_timeout,
                                max_windows=options.max_windows, cancel=cancel)
            outcome.update(rotation=rotation, player_indices=order,
                           model_sampling_seeds=[p.generator.initial_seed() if isinstance(p, ModelPlayer) else None for p in players])
            outcomes.append(outcome)
            print(f"seed={seed} rotation={rotation} {outcome['status']} windows={outcome['windows']}", flush=True)
            if outcome["status"] == "cancelled":
                break
        if outcomes[-1]["status"] == "cancelled":
            break
    # Only complete seed groups enter summaries. Four rotations of one seed do
    # not become four independent observations, and failed matches do not vanish.
    groups = []
    for seed in options.seeds:
        matches = [o for o in outcomes if o["seed"] == seed]
        if len(matches) != (4 if options.rotate else 1) or any(o["status"] != "complete" for o in matches):
            continue
        scores, ranks = [[] for _ in range(4)], [[] for _ in range(4)]
        for match in matches:
            for seat, index in enumerate(match["player_indices"]):
                scores[index].append(match["scores"][seat])
                if match["ranks"] is not None:
                    ranks[index].append(match["ranks"][seat])
        groups.append(dict(seed=seed, scores=[sum(v) / len(v) for v in scores],
                           ranks=None if options.mode == "single" else [sum(v) / len(v) for v in ranks]))
    report = dict(schema="mahjong-arena-evaluation/1", mode=options.mode,
                  profile_digest=digest(profile_for(options.mode)), players=identities,
                  prompt_version=PROMPT_VERSION, prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                  seeds=options.seeds, rotate=options.rotate, decision_timeout=options.decision_timeout,
                  max_windows=options.max_windows, outcomes=outcomes, seed_groups=groups,
                  completed_seed_groups=len(groups), failed_matches=sum(o["status"] != "complete" for o in outcomes),
                  model_sampling="categorical", strength_claim=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"评估报告：{output}", flush=True)
    return 0 if report["failed_matches"] == 0 else 1
