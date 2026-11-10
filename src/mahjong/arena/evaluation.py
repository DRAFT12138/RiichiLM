"""Bounded headless matches using exactly the same players and loop as Web."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from mahjong.contracts import MahjongError
from .llm import LLMPlayer, PROMPT_VERSION, SYSTEM_PROMPT, load_llm_configs
from .loop import DecisionLoop
from .players import ModelPlayer, RandomPlayer, load_model_policy


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
        error = dict(code=exc.code, message=exc.message, phase=exc.phase,
                     details=deepcopy(exc.details), traceback=traceback.format_exc())
    except Exception as exc:
        error = dict(code="EVALUATION_FAILED", message=f"评估执行失败（{type(exc).__name__}）",
                     exception_type=type(exc).__name__, traceback=traceback.format_exc())
    return dict(status=status, error=error, seed=seed, rounds=rounds, windows=windows,
                seconds=time.monotonic() - started, scores=scores, ranks=ranks,
                unclaimed_kyotaku=pool, decisions=loop.decisions,
                llm_stats=[deepcopy(getattr(p, "stats", None)) for p in players])


def evaluate_cli(options, *, cancel=None, on_match=None):
    from .bridge import ArenaDecisionBridge, profile_for
    from mahjong.inference.policy import digest
    from mahjong.policy.execution import policy_precision
    from .memory import ModelMemoryBudget

    if (not math.isfinite(options.decision_timeout) or options.decision_timeout <= 0
            or options.max_windows <= 0 or len(set(options.seeds)) != len(options.seeds)
            or any(not 0 <= seed < 2**32 for seed in options.seeds)):
        raise MahjongError("CONFIG_INVALID", "评估需要有限正数时限、窗口预算及不重复的32位无符号种子")
    output = Path(options.output)
    if output.exists():
        raise MahjongError("CONFIG_INVALID", "评估报告已存在，请使用新路径")
    memory_budget = ModelMemoryBudget(options.device, getattr(options, 'gpu_memory_gib', 16.))
    configs, policies, identities = load_llm_configs(options.llm_config), {}, []
    sources, source_identities = {}, {}
    for spec in options.players:
        if spec == "random":
            identities.append(dict(kind="random"))
        elif spec in configs:
            identities.append(dict(kind="llm", **configs[spec].public(),
                                   config=asdict(configs[spec])))
        elif spec.startswith(("checkpoint:", "weights:")):
            prefix, path = spec.split(':', 1)
            kind = 'weights' if prefix == 'weights' else 'checkpoint'
            if not path:
                raise MahjongError('CONFIG_INVALID', 'model player path is empty')
            path = Path(path).resolve()
            key = (kind, str(path))
            if key not in sources:
                policy = load_model_policy(path, kind=kind, device=options.device, memory_budget=memory_budget)
                if kind == 'checkpoint':
                    policy.verify_training_profile(profile_for("hanchan"))
                    source = policy.source
                    identity = {k: source[k] for k in (
                        'run_id', 'attempt_id', 'checkpoint_id', 'manifest_hash', 'global_step')}
                else:
                    with path.open('rb') as stream:
                        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                    identity = dict(checkpoint_id=path.name, state_sha256=sha)
                sources[key] = policy
                source_identities[key] = dict(kind='model', source_kind=kind, **identity,
                                             precision=policy_precision(policy.model),
                                             model_binding_digest=policy.binding_digest)
            policies[spec] = sources[key]
            identities.append(deepcopy(source_identities[key]))
        else:
            raise MahjongError("CONFIG_INVALID", "玩家必须为 random、已配置的 llm:<id>、checkpoint:<目录> 或 weights:<文件>")
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
                    players.append(ModelPlayer(policies[spec], seed=derived, memory_budget=memory_budget))
                    controllers.append(("model", policies[spec]))
            bridge = ArenaDecisionBridge(profile_for(options.mode), model_binding_digest="arena-non-model")
            bridge.bind_controllers(controllers)
            print(f"seed={seed} rotation={rotation} started", flush=True)
            outcome = run_match(bridge, players, seed=seed, single_round=options.mode == "single",
                                decision_timeout=options.decision_timeout,
                                max_windows=options.max_windows, cancel=cancel)
            outcome.update(rotation=rotation, player_indices=order,
                           prefix_cache=[p.prefix_cache.snapshot() if isinstance(p, ModelPlayer) else None for p in players],
                           model_sampling_seeds=[p.generator.initial_seed() if isinstance(p, ModelPlayer) else None for p in players])
            outcomes.append(outcome)
            print(f"seed={seed} rotation={rotation} {outcome['status']} windows={outcome['windows']} scores={outcome['scores']}", flush=True)
            if outcome['error'] is not None:
                print(json.dumps(dict(seed=seed, rotation=rotation, error=outcome['error']),
                                 ensure_ascii=False), flush=True)
            if on_match is not None:
                on_match(deepcopy(outcome), deepcopy(identities))
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
                  model_sampling="categorical", strength_claim=False,
                  gpu_memory=memory_budget.snapshot(),
                  model_summaries=model_summaries(identities, outcomes, groups,
                                                 profile_for(options.mode)['initial_scores']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"评估报告：{output}", flush=True)
    return 0 if report["failed_matches"] == 0 else 1


def model_summaries(identities, outcomes, groups, initial_scores):
    """Group repeated model seats; only fully completed seed groups count."""
    from mahjong.inference.policy import digest

    models = {}
    for index, identity in enumerate(identities):
        if identity['kind'] != 'model':
            continue
        key = digest(identity)
        if key not in models:
            models[key] = dict(identity=deepcopy(identity), player_indices=[],
                               appearances=0, total_score=0, net_points=0, ranks=[])
        models[key]['player_indices'].append(index)
    complete_seeds = {g['seed'] for g in groups}
    for match in outcomes:
        if match['seed'] not in complete_seeds or match['status'] != 'complete':
            continue
        for seat, index in enumerate(match['player_indices']):
            identity = identities[index]
            if identity['kind'] != 'model':
                continue
            row = models[digest(identity)]
            row['appearances'] += 1
            row['total_score'] += match['scores'][seat]
            row['net_points'] += match['scores'][seat] - initial_scores[seat]
            if match['ranks'] is not None:
                row['ranks'].append(match['ranks'][seat])
    result = []
    for row in models.values():
        ranks = row.pop('ranks')
        count = row['appearances']
        row.update(mean_score=row['total_score'] / count if count else None,
                   mean_net_points=row['net_points'] / count if count else None,
                   mean_rank=sum(ranks) / len(ranks) if ranks else None)
        result.append(row)
    return result
