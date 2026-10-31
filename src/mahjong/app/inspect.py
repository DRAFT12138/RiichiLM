"""Read-only status inspection; absence of a summary never implies completion."""

import json
from pathlib import Path

from mahjong.contracts import MahjongError, Summary, decode_contract, encode_json_contract
from . import runs


def _read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise MahjongError('LOCAL_RECORD_FAILED', f'Cannot inspect {path}: {exc}') from exc


def inspect_run(run):
    from mahjong.training.checkpoint import FileCheckpointStore
    run_dir = Path(run).resolve()
    run_id = runs._identity(run_dir)
    attempt_id = runs.active_attempt(run_dir)
    directory = run_dir / 'attempts' / attempt_id
    status = _read(directory / 'status.json')
    summary = _read(directory / 'summary.json')
    manifest = _read(directory / 'manifest.json')
    locked = runs.writer_is_locked(run_dir)
    latest = FileCheckpointStore().find_latest(run_dir, attempt_id)
    if summary is not None:
        terminal = decode_contract(Summary, summary)
        if (terminal.identity.kind, terminal.identity.run_id, terminal.identity.attempt_id) != ('training', run_id, attempt_id):
            raise MahjongError('LOCAL_RECORD_FAILED', 'Terminal summary identity disagrees with active attempt')
    outcome = summary['completion']['outcome'] if summary is not None else ('running' if locked else 'unknown')
    artifact_checks = None if summary is None else {
        path: {'recorded': state, 'exists': (run_dir / path).exists()}
        for path, state in summary['artifacts'].items()
    }
    from mahjong.observability.lineage import read_branch
    history = read_branch(run_dir, attempt_id)
    return {
        'run_id': run_id, 'run_dir': str(run_dir), 'active_attempt': attempt_id,
        'outcome': outcome, 'writer_locked': locked,
        'last_observed_progress': None if status is None else status.get('progress'),
        'last_heartbeat_at': None if status is None else status.get('last_heartbeat_at'),
        'last_saved_step': None if latest is None else latest.global_step,
        'last_valid_checkpoint': encode_json_contract(latest),
        'source_checkpoint': None if manifest is None else manifest.get('source_checkpoint'),
        'artifacts': None if summary is None else summary.get('artifacts'),
        'artifact_checks': artifact_checks,
        'history_incomplete': history['history_incomplete'],
        'missing_intervals': history['missing_intervals'],
        'terminal_summary': summary,
    }
