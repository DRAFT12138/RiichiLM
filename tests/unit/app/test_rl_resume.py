"""RL resume treats source and environment hashes as provenance."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mahjong.app import rl
from mahjong.contracts import MahjongError, RunMetadata
from tests.fixtures.contracts import make_snapshot


@pytest.mark.parametrize('change', ['source', 'dependency', 'component', 'config'])
def test_rl_resume_checks_compatibility_without_requiring_the_same_source(tmp_path, monkeypatch, change):
    snapshot = make_snapshot()
    config = {**rl.parse_new([], tmp_path), 'rl': rl.resolve(), 'rl_origin': {}}
    snapshot = replace(snapshot, config=config)
    metadata = RunMetadata('new-source', snapshot.identity.dependency_fingerprint, {},
        snapshot.descriptors, snapshot.frozen_config_hash, {})
    if change == 'dependency':
        metadata = replace(metadata, dependency_fingerprint='changed',
            environment={'python': 'changed', 'dependencies': [{'name': 'torch', 'version': 'changed'}]})
    elif change == 'component':
        metadata = replace(metadata, descriptors=())
    elif change == 'config':
        metadata = replace(metadata, frozen_config_hash='changed')
    reference = object()
    store = SimpleNamespace(load=lambda *a, **k: snapshot)
    context = SimpleNamespace(run_dir=tmp_path)
    observer = Mock()
    controller = Mock()
    controller.run.return_value = 'resumed'
    monkeypatch.setattr(rl, 'FileCheckpointStore', lambda: store)
    monkeypatch.setattr(rl, 'checkpoint_source', lambda *a: (tmp_path, reference))
    monkeypatch.setattr(rl, 'reconcile', lambda *a, **k: reference)
    monkeypatch.setattr(rl.runs, 'lock_run', lambda *a: nullcontext())
    monkeypatch.setattr(rl.runs, 'create_attempt', lambda *a: context)
    monkeypatch.setattr(rl.runs, 'publish_metadata', Mock())
    monkeypatch.setattr(rl.runs, 'activate', Mock())
    monkeypatch.setattr(rl.assembly, 'build_observer', lambda *a: observer)
    monkeypatch.setattr(rl.assembly, 'collect_metadata', lambda *a: metadata)
    monkeypatch.setattr(rl, 'build_qwen3', lambda *a, **k: object())
    monkeypatch.setattr(rl, 'Qwen3PolicyAdapter', lambda *a, **k: object())
    monkeypatch.setattr(rl, 'RLTrainSource', lambda *a: object())
    monkeypatch.setattr(rl, 'make_runtime', lambda *a: object())
    monkeypatch.setattr(rl, 'RLController', lambda **k: controller)
    monkeypatch.setattr(rl, 'finish', lambda observer, execute, **k: execute())
    options = SimpleNamespace(run=str(tmp_path), checkpoint='latest')
    if change in ('source', 'dependency'):
        assert rl.resume(options, stop=object(), argv=[]) == 'resumed'
        controller.run.assert_called_once_with(snapshot)
        assert rl.runs.publish_metadata.call_args.args[2].code_fingerprint == 'new-source'
        assert rl.runs.publish_metadata.call_args.args[2].environment == metadata.environment
    else:
        with pytest.raises(MahjongError, match='RESUME_INCOMPATIBLE'):
            rl.resume(options, stop=object(), argv=[])
        controller.run.assert_not_called()
