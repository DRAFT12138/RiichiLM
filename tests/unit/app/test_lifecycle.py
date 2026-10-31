"""A terminal message requires a committed summary, including error paths."""

from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

from mahjong.app.config import DEFAULTS
from mahjong.app.lifecycle import announce, finish
from mahjong.contracts import Completion, MahjongError, RecordIdentity, Summary


class Observer:
    def __init__(self, *, fail_event=False, fail_close=False):
        self.events = []
        self.closes = []
        self.fail_event = fail_event
        self.fail_close = fail_close

    def event(self, kind, payload, progress):
        self.events.append((kind, payload))
        if self.fail_event:
            raise MahjongError('LOCAL_RECORD_FAILED', 'disk full')

    def close(self, completion):
        self.closes.append(completion)
        if self.fail_close:
            raise MahjongError('LOCAL_RECORD_FAILED', 'summary not published')
        return Summary(1, RecordIdentity('training', 'R', 'A', None), completion, '2026-10-11T00:00:00Z', {})


def test_result_printed_only_after_successful_close(capsys):
    observer = Observer()
    completion = Completion('completed', 0, 3, None, None, None)
    assert finish(observer, lambda: completion) == 0
    assert observer.closes == [completion]
    assert 'outcome=completed' in capsys.readouterr().out


def test_summary_failure_never_announces_completed(capsys):
    observer = Observer(fail_close=True)
    assert finish(observer, lambda: Completion('completed', 0, 3, None, None, None)) == 1
    output = capsys.readouterr()
    assert 'completed' not in output.out
    assert 'terminal summary unavailable' in output.err
    assert len(observer.closes) == 1


def test_runtime_failure_has_traceback_progress_and_single_close(capsys):
    observer = Observer()
    def execute():
        raise MahjongError('NONFINITE_LOSS', 'NaN', 'training', {'progress': {'global_step': 5}})
    assert finish(observer, execute) == 1
    assert observer.closes[0].global_step == 5
    assert observer.closes[0].outcome == 'failed'
    assert observer.events[0][0] == 'exception'
    assert 'Traceback' in observer.events[0][1]['traceback']
    assert len(observer.closes) == 1


def test_broken_local_storage_does_not_retry_exception_record(capsys):
    observer = Observer(fail_close=True)
    def execute():
        raise MahjongError('LOCAL_RECORD_FAILED', 'bad fd', 'training')
    assert finish(observer, execute) == 1
    assert observer.events == []
    assert len(observer.closes) == 1


def test_exception_record_failure_changes_exit_code_to_runtime_failure(capsys):
    observer = Observer(fail_event=True)
    def execute():
        raise MahjongError('RESUME_INCOMPATIBLE', 'fingerprint differs', 'restoring')
    assert finish(observer, execute) == 1
    assert observer.closes[0].exit_code == 1
    assert len(observer.events) == len(observer.closes) == 1


def test_announce_starts_dashboard_and_prints_actual_schedule(tmp_path, monkeypatch, capsys):
    context = SimpleNamespace(run_dir=tmp_path / 'run', attempt_dir=tmp_path / 'run' / 'attempt')
    roots = []
    def service(root):
        roots.append(root)
        return 'http://127.0.0.1:6007/#scalars', True
    monkeypatch.setattr('mahjong.observability.dashboard.ensure_dashboard', service)
    settings = deepcopy(DEFAULTS)
    announce(context, settings)
    out = capsys.readouterr().out
    assert roots == [tmp_path]
    assert 'tensorboard_url=http://127.0.0.1:6007/#scalars (reused)' in out
    assert 'training_target_updates=200' in out
    assert 'checkpoint_every_seconds=1800.0 checkpoint_every_updates=None' in out
    settings['observability']['tensorboard_enabled'] = False
    announce(context, settings)
    assert roots == [tmp_path]
    assert 'tensorboard_url=' not in capsys.readouterr().out


def test_announce_dashboard_failure_is_optional(tmp_path, monkeypatch, capsys):
    def broken(root):
        raise OSError('port or disk unavailable')
    monkeypatch.setattr('mahjong.observability.dashboard.ensure_dashboard', broken)
    context = SimpleNamespace(run_dir=tmp_path / 'run', evaluation_dir=tmp_path / 'evaluation')
    announce(context, deepcopy(DEFAULTS))
    assert 'will continue with local logs' in capsys.readouterr().err

