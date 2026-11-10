#!/usr/bin/env python3
"""Export a new checkpoint and evaluate it against the latest three 3090 weights.

Uses the existing Arena players, categorical RNG domains and four-seat rotations.
Every attempted match is persisted; incomplete seed groups never enter statistics.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def latest_opponents(root, candidate):
    choices = []
    for path in root.glob('3090_*.pt'):
        match = re.fullmatch(r'3090_(\d+)w(\d{0,4})', path.stem)
        if match and path.resolve() != candidate.resolve():
            step = int(match[1]) * 10000 + int(match[2].ljust(4, '0'))
            choices.append((step, path))
    return [path for _, path in sorted(choices, reverse=True)[:3]]


def save_statistics(directory, outcomes, identities):
    from mahjong.arena.evaluation import model_summaries

    seeds = sorted({o['seed'] for o in outcomes})
    complete = []
    for seed in seeds:
        matches = [o for o in outcomes if o['seed'] == seed]
        if (len(matches) == 4 and {o['rotation'] for o in matches} == set(range(4))
                and all(o['status'] == 'complete' for o in matches)):
            complete.append(dict(seed=seed))
    summaries = model_summaries(identities, outcomes, complete, [25000] * 4)
    included = {g['seed'] for g in complete}
    for row in summaries:
        ranks = [o['ranks'][seat] for o in outcomes if o['seed'] in included
                 for seat, index in enumerate(o['player_indices']) if index in row['player_indices']]
        row['rank_counts'] = [ranks.count(rank) for rank in range(1, 5)]
    stats = dict(attempted_matches=len(outcomes),
                 failed_matches=sum(o['status'] != 'complete' for o in outcomes),
                 completed_seed_groups=len(complete), counted_matches=4 * len(complete),
                 players=identities, outcomes=outcomes, model_summaries=summaries,
                 strength_claim=False)
    atomic_json(directory / 'progress.json', stats)
    columns = ['model', 'appearances', 'mean_rank', 'mean_score', 'mean_net_points',
               'first', 'second', 'third', 'fourth']
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in summaries:
            writer.writerow([row['identity']['checkpoint_id'], row['appearances'], row['mean_rank'],
                             row['mean_score'], row['mean_net_points'], *row['rank_counts']])
    lines = [f"已尝试 {len(outcomes)} 场；失败／取消 {stats['failed_matches']} 场；"
             f"纳入统计 {stats['counted_matches']} 场（{len(complete)} 个完整种子组）。", '',
             '| 模型 | 场数 | 平均顺位↓ | 场均净得点 | 一／二／三／四位 |',
             '| --- | ---: | ---: | ---: | --- |']
    for row in sorted(summaries, key=lambda r: r['mean_rank'] if r['mean_rank'] is not None else 99):
        rank = f"{row['mean_rank']:.2f}" if row['mean_rank'] is not None else '—'
        net = f"{row['mean_net_points']:+,.0f}" if row['mean_net_points'] is not None else '—'
        counts = '／'.join(map(str, row['rank_counts']))
        lines.append(f"| {row['identity']['checkpoint_id']} | {row['appearances']} | {rank} | {net} | {counts} |")
    lines += ['', '净得点不含顺位马／头名奖励。少量种子组不能确定稳定棋力差异。',
              '失败场保留在 matches.jsonl 和 progress.json；不完整的四场换座组全部排除，不自动换种子。']
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(lines[0], flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'state.pt')
    parser.add_argument('--weights-root', type=Path, default=ROOT / 'weights')
    parser.add_argument('--opponents', type=Path, nargs=3, help='不指定时按训练步数选择最新三个3090权重')
    parser.add_argument('--games', type=int, default=20, help='完整半庄数，必须为4的倍数')
    parser.add_argument('--seeds', type=int, nargs='+', help='默认随机生成新种子；数量须为 games/4')
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default=None)
    parser.add_argument('--gpu-memory-gib', type=float, default=16.)
    parser.add_argument('--decision-timeout', type=float, default=120.)
    parser.add_argument('--max-windows', type=int, default=20000)
    parser.add_argument('--output-dir', type=Path, help='新的输出目录，禁止覆盖既有评测')
    args = parser.parse_args()
    if args.games <= 0 or args.games % 4:
        parser.error('--games 必须为正数且为4的倍数')
    seeds = args.seeds if args.seeds is not None else secrets.SystemRandom().sample(range(2**32), args.games // 4)
    if len(seeds) != args.games // 4 or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        parser.error('--seeds 需为不重复的32位无符号数，数量为 games/4')

    import torch
    from mahjong.arena.evaluation import evaluate_cli
    from mahjong.inference.weights import export_policy_weights, read_weights

    source_hash = file_hash(args.checkpoint)
    snapshot = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    step = snapshot.get('progress', {}).get('global_step')
    if type(step) is not int or step < 0:
        parser.error('--checkpoint 需要含 global_step 的训练 state.pt')
    del snapshot
    whole, remainder = divmod(step, 10000)
    label = f"{whole}w" + (f"{remainder:04d}".rstrip('0') if remainder else '')
    candidate = args.weights_root / f'3090_{label}.pt'
    opponents = args.opponents or latest_opponents(args.weights_root, candidate)
    if len(opponents) != 3 or any(not p.is_file() for p in opponents):
        parser.error('需要三份存在的对手权重，可显式指定 --opponents')
    if len({p.resolve() for p in [candidate, *opponents]}) != 4:
        parser.error('候选和三个对手必须为不同文件')
    directory = args.output_dir or ROOT / 'outputs' / 'arena' / f'3090-{label}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    directory.mkdir(parents=True, exist_ok=False)
    if not candidate.exists():
        export_policy_weights(args.checkpoint, candidate)
    settings, state = read_weights(args.checkpoint)
    exported_settings, exported = read_weights(candidate)
    if (settings != exported_settings or state.keys() != exported.keys()
            or any(state[k].dtype != exported[k].dtype or not torch.equal(state[k], exported[k]) for k in state)):
        raise ValueError(f'已有导出 {candidate} 与源checkpoint不一致；未覆盖，请检查或指定其他 weights-root')
    if file_hash(args.checkpoint) != source_hash:
        raise ValueError('state.pt 在导出期间改变，请使用固定的checkpoint文件重试')
    del state, exported
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    files = [dict(path=str(p.resolve()), sha256=file_hash(p), bytes=p.stat().st_size)
             for p in [args.checkpoint, candidate, *opponents]]
    provenance = dict(created=datetime.now().astimezone().isoformat(), source_step=step, files=files,
                      seeds=seeds, rotate=True, games=args.games, device=device,
                      device_name=torch.cuda.get_device_name(0) if device == 'cuda:0' else 'cpu',
                      torch_version=torch.__version__, export_verified=True,
                      argv=sys.argv, git_head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    atomic_json(directory / 'provenance.json', provenance)
    options = SimpleNamespace(players=[f'weights:{p.resolve()}' for p in [candidate, *opponents]],
                              llm_config=None, mode='hanchan', device=device,
                              gpu_memory_gib=args.gpu_memory_gib, seeds=seeds, rotate=True,
                              decision_timeout=args.decision_timeout, max_windows=args.max_windows,
                              output=str(directory / 'report.json'))
    print(f'候选={candidate} 对手={opponents}\n种子={seeds}\n输出={directory}', flush=True)
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    outcomes = []

    def on_match(outcome, identities):
        with (directory / 'matches.jsonl').open('a') as stream:
            stream.write(json.dumps(outcome, ensure_ascii=False, allow_nan=False) + '\n')
        outcomes.append(outcome)
        save_statistics(directory, outcomes, identities)

    return evaluate_cli(options, cancel=stopped.is_set, on_match=on_match)


if __name__ == '__main__':
    raise SystemExit(main())
