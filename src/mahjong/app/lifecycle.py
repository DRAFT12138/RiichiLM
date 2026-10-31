"""One terminal transaction for a training attempt or independent evaluation."""

import sys
import traceback

from mahjong.contracts import Completion, CheckpointRef, MahjongError, decode_contract, encode_json_contract


def finish(observer, execute, *, initial_checkpoint=None):
    """Execute, close once, and print a result only after its summary exists."""
    try:
        completion = execute()
    except Exception as exc:
        error = exc if isinstance(exc, MahjongError) else MahjongError(
            'CONTRACT_MISMATCH', str(exc), 'initializing')
        progress = error.details.get('progress') or {}
        reference = error.details.get('last_checkpoint')
        if reference is not None:
            try:
                reference = decode_contract(CheckpointRef, reference)
            except MahjongError:
                reference = initial_checkpoint
        else:
            reference = initial_checkpoint
        completion = Completion('failed', error.exit_code, progress.get('global_step', 0),
                                reference, progress.get('last_evaluation_ref'), str(error))
        print(str(error), file=sys.stderr, flush=True)
        if error.code != 'LOCAL_RECORD_FAILED':
            try:
                observer.event('exception', {
                    'code': error.code, 'message': error.message,
                    'traceback': traceback.format_exc(),
                    'last_checkpoint': encode_json_contract(reference),
                }, None)
            except Exception as record_error:
                print(f'LOCAL_RECORD_FAILED: {record_error}', file=sys.stderr, flush=True)
                completion = Completion('failed', 1, completion.global_step, reference,
                                        completion.report_ref, str(record_error))
    try:
        summary = observer.close(completion)
    except Exception as exc:
        print(f'LOCAL_RECORD_FAILED: terminal summary unavailable: {exc}', file=sys.stderr, flush=True)
        return 1
    committed = summary.completion
    print(f'outcome={committed.outcome} step={committed.global_step} exit_code={committed.exit_code}', flush=True)
    if committed.last_checkpoint is not None:
        print(f'checkpoint={committed.last_checkpoint.path}', flush=True)
    if committed.report_ref is not None:
        print(f'report={committed.report_ref}', flush=True)
    return committed.exit_code


def announce(context, config):
    path = context.attempt_dir if hasattr(context, 'attempt_dir') else context.evaluation_dir
    print(f'run={context.run_dir}', flush=True)
    print(f'{"attempt" if hasattr(context, "attempt_dir") else "evaluation"}={path}', flush=True)
    if hasattr(context, 'attempt_dir'):
        print(f'training_target_updates={config["training"]["max_updates"]}', flush=True)
        print(f'checkpoint_every_seconds={config["checkpoint"]["every_seconds"]} '
              f'checkpoint_every_updates={config["checkpoint"]["every_updates"]}', flush=True)
    if config['observability']['tensorboard_enabled']:
        try:
            from mahjong.observability.dashboard import ensure_dashboard
            url, reused = ensure_dashboard(context.run_dir.parent)
            print(f'tensorboard_url={url} ({"reused" if reused else "started"})', flush=True)
        except Exception as exc:
            print(f'TensorBoard unavailable: {exc}; training/evaluation will continue with local logs',
                  file=sys.stderr, flush=True)


def bind_stop(observer, stop):
    # An implementation-only mirror: the public Observer protocol stays fixed.
    observer.bind_stop(stop)
