"""Statistics must exclude an entire failed rotation group, preserving evidence."""

import importlib.util
import json
from pathlib import Path

import torch

from tests.unit.inference.test_weights import weights


def load_script():
    path = Path(__file__).resolve().parents[3] / 'scripts' / 'evaluate-checkpoint.py'
    spec = importlib.util.spec_from_file_location('evaluate_checkpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_weights_uses_numeric_steps_and_excludes_candidate(tmp_path):
    script = load_script()
    for name in ['3090_7w2.pt', '3090_10w.pt', '3090_13w5.pt', '3090_17w.pt', '3090_20w.pt', 'other.pt']:
        (tmp_path / name).touch()
    assert [p.name for p in script.latest_opponents(tmp_path, tmp_path / '3090_20w.pt')] == [
        '3090_17w.pt', '3090_13w5.pt', '3090_10w.pt']


def test_live_statistics_keep_failed_group_out_and_map_rotated_seats(tmp_path):
    script = load_script()
    identities = [dict(kind='model', checkpoint_id=name) for name in ['new', 'a', 'b', 'c']]
    outcomes = []
    for seed in [100, 101]:
        for rotation in range(4):
            order = [(seat - rotation) % 4 for seat in range(4)]
            outcomes.append(dict(seed=seed, rotation=rotation, player_indices=order,
                                 status='complete', scores=[[40000, 30000, 20000, 10000][i] for i in order],
                                 ranks=[i + 1 for i in order], error=None))
    outcomes[2].update(status='failed', scores=None, ranks=None, error=dict(code='ENVIRONMENT_STEP_FAILED'))
    stats = script.save_statistics(tmp_path, outcomes, identities)
    assert stats['attempted_matches'] == 8 and stats['failed_matches'] == 1
    assert stats['counted_matches'] == 4 and stats['completed_seed_groups'] == 1
    new = stats['model_summaries'][0]
    assert new['appearances'] == 4 and new['mean_net_points'] == 15000 and new['mean_rank'] == 1
    assert new['rank_counts'] == [4, 0, 0, 0]
    saved = json.loads((tmp_path / 'progress.json').read_text())
    assert saved['outcomes'][2]['error']['code'] == 'ENVIRONMENT_STEP_FAILED'
    assert len((tmp_path / 'summary.csv').read_text().splitlines()) == 5
    assert 'ENVIRONMENT_STEP_FAILED' not in (tmp_path / 'summary.md').read_text()


def test_script_exports_verifies_selects_opponents_and_persists_matches(weights, tmp_path, monkeypatch):
    from mahjong.arena import evaluation
    from mahjong.inference.weights import read_weights

    script = load_script()
    source, _, _ = weights
    payload = torch.load(source, weights_only=True)
    payload['progress']['global_step'] = 200000
    torch.save(payload, source)
    root = tmp_path / 'weights'
    root.mkdir()
    for name in ['3090_10w.pt', '3090_13w5.pt', '3090_17w.pt']:
        (root / name).write_bytes(b'fake opponent; loader is replaced')
    output = tmp_path / 'evaluation'
    monkeypatch.setattr(script.sys, 'argv', ['evaluate-checkpoint.py', '--checkpoint', str(source),
                      '--weights-root', str(root), '--games', '4', '--seeds', '123456',
                      '--device', 'cpu', '--output-dir', str(output)])
    monkeypatch.setattr(script.signal, 'signal', lambda *_: None)

    def evaluate(options, *, cancel, on_match):
        assert [Path(p.split(':', 1)[1]).name for p in options.players] == [
            '3090_20w.pt', '3090_17w.pt', '3090_13w5.pt', '3090_10w.pt']
        assert options.seeds == [123456] and options.rotate and not cancel()
        identities = [dict(kind='model', checkpoint_id=str(i)) for i in range(4)]
        for rotation in range(4):
            on_match(dict(seed=123456, rotation=rotation, status='complete',
                          player_indices=[(s - rotation) % 4 for s in range(4)],
                          ranks=[1, 2, 3, 4], scores=[40000, 30000, 20000, 10000]), identities)
        return 0

    monkeypatch.setattr(evaluation, 'evaluate_cli', evaluate)
    assert script.main() == 0
    config, state = read_weights(source)
    exported_config, exported = read_weights(root / '3090_20w.pt')
    assert config == exported_config and all(torch.equal(t, exported[k]) for k, t in state.items())
    assert len((output / 'matches.jsonl').read_text().splitlines()) == 4
    assert json.loads((output / 'progress.json').read_text())['counted_matches'] == 4
    assert json.loads((output / 'provenance.json').read_text())['export_verified'] is True
