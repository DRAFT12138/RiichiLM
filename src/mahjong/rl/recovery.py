"""Scan published checkpoints before pruning any external dependencies."""
import shutil
from mahjong.contracts import MahjongError
from mahjong.rl.pool import members
from mahjong.rl.storage import checked_path, EpisodeReader
from mahjong.inference.actor_loading import load_actor
from mahjong.inference.policy import digest


def dependencies(data):
    return dict(actors=list(members(data['pool']).values()),
                trajectories=data['refs'] + data.get('validation_refs', []))


def reconcile(store, run_dir, limits, *, verify=True):
    published = []
    root = run_dir / 'checkpoints'
    if root.exists():
        for directory in root.iterdir():
            if directory.is_dir() and not directory.name.startswith('.'):
                manifest, ref = store._manifest(directory)
                if 'rl_dependencies' not in manifest:
                    raise MahjongError('CHECKPOINT_INVALID', 'non-RL checkpoint in RL run')
                deps = manifest['rl_dependencies']
                if verify:
                    snapshot = store.load(ref, run_dir=run_dir)
                    if digest(deps) != digest(dependencies(snapshot.data)):
                        raise MahjongError('CHECKPOINT_INVALID', 'dependency manifest differs from snapshot')
                    # External corruption must not trigger fallback to an older checkpoint.
                    for actor in deps['actors']:
                        load_actor(run_dir / 'actors' / actor['artifact_id'], expected=actor)
                    for trajectory in deps['trajectories']:
                        EpisodeReader(run_dir, trajectory, limits)
                published.append((ref, deps))
    if not published:
        return None
    published.sort(key=lambda pair: pair[0].commit_seq)
    latest = published[-1][0]
    store.prune(run_dir=run_dir, latest=latest)
    kept = [(r,d) for r,d in published if (run_dir/r.path).exists()]
    actors = {a['artifact_id'] for _,d in kept for a in d['actors']}
    paths = {t['path'] for _,d in kept for t in d['trajectories']}
    for name, protected in [('actors', actors), ('trajectories', {p.split('/')[-1] for p in paths})]:
        directory = run_dir / name
        if directory.exists():
            for child in directory.iterdir():
                if child.is_dir() and not child.is_symlink() and child.name not in protected:
                    shutil.rmtree(child)
    return latest
