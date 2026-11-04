"""Public single-run command grammar."""

import argparse
from pathlib import Path
import sys

from mahjong.contracts import MahjongError, canonical_json


def parser():
    result = argparse.ArgumentParser(prog='python -m mahjong', description='阶段 1 训练、恢复和独立评测')
    commands = result.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='创建新实验；key=value 覆盖配置')
    training.add_argument('--init-weights', help='仅加载指定 checkpoint 模型，创建新 run')
    training.add_argument('overrides', nargs='*', metavar='key=value')
    pretraining = commands.add_parser('pretrain', help='使用已清洗的麻将合法动作样本进行 Qwen3 监督训练')
    pretraining.add_argument('overrides', nargs='*', metavar='key=value')
    recovery = commands.add_parser('resume', help='同 run 新 attempt 完整续训')
    recovery.add_argument('--run', help='run 目录；latest 时必需')
    recovery.add_argument('--checkpoint', required=True, help='checkpoint 目录或 latest')
    recovery.add_argument('overrides', nargs='*', metavar='key=value')
    evaluation = commands.add_parser('evaluate', help='独立评测 checkpoint，不更新训练状态')
    evaluation.add_argument('--checkpoint', required=True, help='显式 checkpoint 目录')
    evaluation.add_argument('overrides', nargs='*', metavar='key=value')
    inspection = commands.add_parser('inspect', help='只读查看当前 attempt、终态和可恢复状态')
    inspection.add_argument('--run', required=True, help='run 目录')
    export = commands.add_parser('export-weights', help='从训练 .pt 导出精简推理权重，保留原精度')
    export.add_argument('--checkpoint', required=True, help='源 state.pt 文件')
    export.add_argument('--output', required=True, help='新推理 .pt 文件，不覆盖已有文件')
    rl = commands.add_parser('rl', help='有界 RL 训练、恢复及纯 Actor 配对评测')
    modes = rl.add_subparsers(dest='rl_command', required=True)
    new = modes.add_parser('train', help='从兼容 NTP checkpoint 初始化独立 RL 实验')
    new.add_argument('--init-checkpoint', required=True)
    new.add_argument('--output', default='outputs/rl')
    recovery_rl = modes.add_parser('resume', help='从最新完整 READY/PREPARED 恢复')
    recovery_rl.add_argument('--checkpoint', required=True)
    recovery_rl.add_argument('--run')
    evaluation_rl = modes.add_parser('evaluate', help='纯 Actor 快照，四席配对评测')
    evaluation_rl.add_argument('--actor', required=True)
    evaluation_rl.add_argument('--baseline', required=True)
    evaluation_rl.add_argument('--opponent')
    evaluation_rl.add_argument('--output', required=True)
    evaluation_rl.add_argument('--strategy', choices=('categorical','greedy'), default='categorical')
    evaluation_rl.add_argument('--seed-plan', choices=('monitor','heldout'), default='heldout')
    for entry in (new, evaluation_rl):
        entry.add_argument('--config', help='RL YAML 配置；省略时使用 configs/rl.yaml')
        entry.add_argument('--device', choices=('cpu','cuda:0'), default='cpu')
        entry.add_argument('overrides', nargs='*', metavar='rl.key=value')
    web = commands.add_parser('web-play', help='本机 React 真人对战；退出后重新开局')
    web.add_argument('--models-root', default='outputs')
    web.add_argument('--weights-root', default='weights', help='自定义 .pt/.pth 权重目录，无需训练 run 元数据')
    web.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    web.add_argument('--gpu-memory-gib', type=float, default=16., help='模型玩家进程显存预算，默认16GiB，按空闲显存动态下调')
    web.add_argument('--port', type=int, default=8765)
    web.add_argument('--lease-seconds', type=float, default=30)
    web.add_argument('--load-timeout', type=float, default=300)
    web.add_argument('--decision-timeout', type=float, default=120)
    web.add_argument('--dev-origin', help='本机 Vite 开发页面 origin，如 http://localhost:5173')
    web.add_argument('--llm-config', help='服务端 LLM 玩家 JSON 配置；密钥只从环境变量读取')
    arena = commands.add_parser('arena-evaluate', help='通用玩家对战评估：LLM、本地 NTP 和随机策略')
    arena.add_argument('--players', nargs=4, required=True, metavar='PLAYER',
                       help='每席 random、llm:<id>、checkpoint:<目录> 或 weights:<文件>')
    arena.add_argument('--llm-config', help='与 web-play 共用的 LLM 玩家 JSON 配置')
    arena.add_argument('--mode', choices=('single', 'tonpuu', 'hanchan'), default='hanchan')
    arena.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    arena.add_argument('--gpu-memory-gib', type=float, default=16., help='模型玩家进程显存预算，默认16GiB，按空闲显存动态下调')
    arena.add_argument('--seeds', nargs='+', type=int, default=[42])
    arena.add_argument('--rotate', action='store_true', help='每个种子轮换四席；按种子聚合')
    arena.add_argument('--decision-timeout', type=float, default=120)
    arena.add_argument('--max-windows', type=int, default=20000)
    arena.add_argument('--output', required=True, help='评估 JSON 报告；不改写已有文件')
    return result


def main(argv=None):
    from mahjong.training.signals import SignalStopToken, signal_handlers
    arguments = list(sys.argv[1:] if argv is None else argv)
    stop = SignalStopToken()
    with signal_handlers(stop):
        command_parser = parser()
        options, trailing = command_parser.parse_known_args(arguments)
        if trailing:
            if options.command in ('inspect', 'web-play', 'arena-evaluate', 'export-weights') or (options.command == 'rl' and options.rl_command == 'resume') or any(token.startswith('-') or '=' not in token for token in trailing):
                command_parser.error('unrecognized arguments: ' + ' '.join(trailing))
            options.overrides.extend(trailing)
        try:
            if options.command == 'export-weights':
                from mahjong.inference.weights import export_policy_weights
                path = export_policy_weights(options.checkpoint, options.output)
                print(f'weights={path} bytes={path.stat().st_size}', flush=True)
                return 0
            if options.command == 'arena-evaluate':
                from mahjong.arena.evaluation import evaluate_cli
                return evaluate_cli(options, cancel=stop.requested)
            if options.command == 'web-play':
                if not 1 <= options.port <= 65535:
                    raise MahjongError('CONFIG_INVALID', '端口必须在1—65535之间')
                try:
                    from mahjong.web.server import serve
                except ModuleNotFoundError as exc:
                    raise MahjongError('CONFIG_INVALID', '请先安装 requirements-rocm.txt 或 requirements-cuda.txt 中的 Web 依赖') from exc
                return serve(options)
            if options.command == 'rl':
                from . import rl
                if options.rl_command == 'evaluate':
                    return rl.evaluate(options, stop=stop)
                return getattr(rl, options.rl_command)(options, stop=stop, argv=arguments)
            if options.command == 'train':
                from .train import train
                return train(options.overrides, cwd=Path.cwd(), stop=stop, argv=arguments,
                             init_weights=options.init_weights)
            if options.command == 'pretrain':
                from .train import train
                return train(options.overrides, cwd=Path.cwd(), stop=stop, argv=arguments,
                             pretraining=True)
            if options.command == 'resume':
                from .train import resume
                return resume(options.overrides, cwd=Path.cwd(), stop=stop, argv=arguments,
                              checkpoint=options.checkpoint, run=options.run)
            if options.command == 'evaluate':
                from .evaluate import evaluate
                return evaluate(options.overrides, cwd=Path.cwd(), stop=stop, argv=arguments,
                                checkpoint=options.checkpoint)
            from .inspect import inspect_run
            print(canonical_json(inspect_run(options.run)))
            return 0
        except MahjongError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return exc.exit_code
        except OSError as exc:
            print(f'LOCAL_RECORD_FAILED: {exc}', file=sys.stderr, flush=True)
            return 1

