"""Shared read-only checkpoint selection for training, RL and evaluation."""
from pathlib import Path

from mahjong.contracts import MahjongError
from . import runs


def checkpoint_source(store, checkpoint, run=None):
    """Resolve an explicit immutable directory, or latest in an explicit run."""
    if checkpoint == 'latest':
        if run is None:
            raise MahjongError('CONFIG_INVALID', '--checkpoint latest requires --run')
        run_dir = Path(run).resolve()
        reference = store.find_latest(run_dir, runs.active_attempt(run_dir))
        if reference is None:
            raise MahjongError('CHECKPOINT_INVALID', 'No valid checkpoint on the active branch')
        print(f'selected_checkpoint={reference.path}', flush=True)
    else:
        path = Path(checkpoint).resolve()
        if path.parent.name != 'checkpoints':
            raise MahjongError('CHECKPOINT_INVALID', 'Expected run/checkpoints/<checkpoint_id>')
        run_dir = path.parent.parent
        if run is not None and Path(run).resolve() != run_dir:
            raise MahjongError('CONFIG_INVALID', '--run disagrees with explicit checkpoint directory')
        reference = store.resolve(path)
    if reference.run_id != runs._identity(run_dir):
        raise MahjongError('CHECKPOINT_INVALID', 'Checkpoint and run identity disagree')
    return run_dir, reference
