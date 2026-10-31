"""Shared checkpoint selection keeps training, RL and evaluation identities."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mahjong.app.checkpoint_source import checkpoint_source
from mahjong.contracts import MahjongError


class Store:
    def __init__(self, reference):
        self.reference = reference
        self.calls = []

    def find_latest(self, run_dir, attempt):
        self.calls.append(('latest', run_dir, attempt))
        return self.reference

    def resolve(self, path):
        self.calls.append(('resolve', path))
        return self.reference


@pytest.fixture
def selection(tmp_path):
    run = tmp_path / 'a run'
    path = run / 'checkpoints' / 'C1'
    path.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'run_id': 'R1'}))
    (run / 'active_attempt.json').write_text(json.dumps({'run_id': 'R1', 'attempt_id': 'A1'}))
    reference = SimpleNamespace(run_id='R1', path='checkpoints/C1')
    return run, path, Store(reference)


def test_explicit_checkpoint_normalizes_paths_and_legacy_alias(selection, monkeypatch, capsys):
    from mahjong.app.train import checkpoint_source as legacy
    from mahjong.app.evaluate import checkpoint_source as evaluation
    assert legacy is evaluation is checkpoint_source
    run, path, store = selection
    monkeypatch.chdir(run.parent)
    assert checkpoint_source(store, Path('a run/checkpoints/C1'), 'a run/../a run') == (run, store.reference)
    assert store.calls == [('resolve', path)]
    assert capsys.readouterr().out == ''


def test_latest_uses_active_attempt_and_prints_selection(selection, capsys):
    run, _, store = selection
    assert checkpoint_source(store, 'latest', run) == (run, store.reference)
    assert store.calls == [('latest', run, 'A1')]
    assert capsys.readouterr().out == 'selected_checkpoint=checkpoints/C1\n'


@pytest.mark.parametrize('case,code,message', [
    ('missing_run', 'CONFIG_INVALID', '--checkpoint latest requires --run'),
    ('no_latest', 'CHECKPOINT_INVALID', 'No valid checkpoint on the active branch'),
    ('bad_directory', 'CHECKPOINT_INVALID', 'Expected run/checkpoints/<checkpoint_id>'),
    ('different_run', 'CONFIG_INVALID', '--run disagrees with explicit checkpoint directory'),
    ('explicit_identity', 'CHECKPOINT_INVALID', 'Checkpoint and run identity disagree'),
    ('latest_identity', 'CHECKPOINT_INVALID', 'Checkpoint and run identity disagree'),
])
def test_invalid_checkpoint_selection(selection, case, code, message):
    run, path, store = selection
    checkpoint, requested_run = path, None
    if case == 'missing_run':
        checkpoint = 'latest'
    elif case == 'no_latest':
        checkpoint, requested_run, store.reference = 'latest', run, None
    elif case == 'bad_directory':
        checkpoint = run / 'C1'
    elif case == 'different_run':
        requested_run = run.parent / 'other'
    else:
        store.reference.run_id = 'OTHER'
        if case == 'latest_identity':
            checkpoint, requested_run = 'latest', run
    with pytest.raises(MahjongError) as error:
        checkpoint_source(store, checkpoint, requested_run)
    assert error.value.code == code
    assert error.value.message == message
    if case in ('missing_run', 'bad_directory', 'different_run'):
        assert store.calls == []
