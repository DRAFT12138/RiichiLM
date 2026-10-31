from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util.tensor_util import make_ndarray

from mahjong import contracts as c
from mahjong.app.config import DEFAULTS
from mahjong.app.runs import create_run
from mahjong.observability.events import LocalObserver
from mahjong.observability.lineage import read_branch, window_totals


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


def observer(tmp_path, **kwargs):
    config = deepcopy(DEFAULTS)
    config['output']['root'] = str(tmp_path)
    context = create_run(config)
    return LocalObserver(context, config, background=False, **kwargs)


def result(): return c.StepResult(torch.tensor(1., requires_grad=True), {'loss': c.MetricStat(2., 1, 'x')})


def interrupted(value):
    return value.close(c.Completion('interrupted', 130, value._g(), None, None, 'test stopped'))


def test_same_phase_mirrors_progress_without_writing_until_heartbeat(tmp_path,monkeypatch):
    value=observer(tmp_path,clock=Clock())
    progress=c.TrainProgress(global_step=1,samples_updated=8)
    value.phase('training',progress)
    writes=[]; original=value._write_status
    def write(**kwargs):
        writes.append(True);return original(**kwargs)
    monkeypatch.setattr(value,'_write_status',write)
    progress.global_step=2;progress.samples_updated=16
    value.phase('training',progress)
    assert not writes
    progress.global_step=999
    value.heartbeat()
    status=json.loads((value.directory/'status.json').read_text())
    assert status['progress']['global_step']==2 and writes==[True]
    value.phase('evaluating',progress)
    assert len(writes)==2
    interrupted(value)
    assert len(writes)==3


def metric_rows(value):
    return [json.loads(line) for line in (value.directory/'metrics.jsonl').read_text().splitlines()]


def test_inherited_window_and_current_attempt_rates(tmp_path):
    clock = Clock(); first = observer(tmp_path/'first', clock=clock)
    progress = c.TrainProgress()
    for g in range(1, 4):
        progress.global_step = g; progress.samples_updated = g*4
        first.record_step(result(), progress, c.Timing(2.), (.003,))
    state = first.state_dict()
    first.write_partial(progress)
    assert first.state_dict() == state  # preview does not clear numerator/threshold
    interrupted(first)
    clock = Clock(); second = observer(tmp_path/'second', clock=clock)
    source = c.CheckpointRef('C', second.context.run_id, 'old', 3, 'checkpoints/C', 'hash', 1)
    second.load_state_dict(state, source)
    second.phase('restoring', progress)
    clock.now = 6.
    for g in range(4, 11):
        clock.now += 2.
        progress.global_step = g; progress.samples_updated = g*4
        second.record_step(result(), progress, c.Timing(2.), (.003,))
        if g <= 8: assert second.performance()['training_samples_per_second'] is None
    performance = second.performance()
    assert performance['training_samples_per_second'] == 2.
    assert performance['end_to_end_samples_per_second'] == 1.4
    assert performance['step_p50_seconds'] == performance['step_p95_seconds'] == 2.
    assert performance['attempt_updates'] == 7 and performance['attempt_samples_updated'] == 28
    row = metric_rows(second)[0]
    assert row['stats']['loss'] == {'numerator': 20., 'denominator': 10, 'value': 2., 'unit': 'x', 'reason': None}
    assert row['window_start_step'] == 0 and row['window_end_step'] == 10
    assert row['source_checkpoint']['checkpoint_id'] == 'C' and row['inherited_through_step'] == 3
    assert row['learning_rates'] == [.003]
    interrupted(second)


def test_nearest_rank_and_bounded_timing_buffer(tmp_path):
    value = observer(tmp_path, clock=Clock())
    progress = c.TrainProgress()
    for index, seconds in enumerate([2.]*5+[1., 2., 3., 4.], 1):
        progress.global_step = index; progress.samples_updated = index*4
        value.record_step(result(), progress, c.Timing(seconds), (.003,))
    assert value.performance()['step_p50_seconds'] == 2.
    assert value.performance()['step_p95_seconds'] == 4.
    for index in range(10, 120):
        progress.global_step = index; progress.samples_updated = index*4
        value.record_step(result(), progress, c.Timing(7.), (.003,))
    assert value.performance()['timing_samples'] == 100
    assert value.performance()['step_p50_seconds'] == value.performance()['step_p95_seconds'] == 7.
    interrupted(value)


def test_tensorboard_matches_full_and_partial_local_values(tmp_path):
    value = observer(tmp_path)
    progress = c.TrainProgress()
    for g in range(1, 14):
        progress.global_step = g; progress.samples_updated = g*4
        value.record_step(result(), progress, c.Timing(1.), (.003,))
    value.write_partial(progress); value.write_partial(progress)
    interrupted(value)
    rows = metric_rows(value)
    assert [row['kind'] for row in rows] == ['window', 'partial']
    assert window_totals(rows)['loss']['denominator'] == 10
    event = EventAccumulator(str(value.directory/'visualization')).Reload()
    for row, tag in zip(rows, ['train/loss', 'train_partial/loss']):
        points = event.Tensors(tag)
        assert len(points) == 1 and points[0].step == row['global_step']
        assert float(make_ndarray(points[0].tensor_proto)) == row['stats']['loss']['value']


@pytest.mark.parametrize('operation', ['initialize', 'add_scalar', 'flush', 'close'])
def test_backend_degrades_once_without_subsequent_submission(tmp_path, operation):
    calls = []
    class Writer:
        def invoke(self, name):
            calls.append(name)
            if name == operation: raise OSError('backend failure')
        def add_scalar(self, *args, **kwargs): self.invoke('add_scalar')
        def flush(self): self.invoke('flush')
        def close(self): self.invoke('close')
    def factory(**kwargs):
        assert kwargs['max_queue'] == 256
        calls.append('initialize')
        if operation == 'initialize': raise OSError('backend failure')
        return Writer()
    value = observer(tmp_path, writer_factory=factory)
    value.config['observability']['metrics_every_updates'] = 1; value._next_record = 1
    progress = c.TrainProgress()
    for g in range(1, 3):
        progress.global_step = g; progress.samples_updated = g*4
        value.record_step(result(), progress, c.Timing(1.), (.003,))
        value.flush()
    summary = interrupted(value)
    events = [json.loads(line) for line in (value.directory/'events.jsonl').read_text().splitlines()]
    assert len([event for event in events if event['type'] == 'backend_degraded']) == 1
    assert calls.count(operation) == 1
    assert summary.completion.outcome == 'interrupted'
    if operation != 'close': assert calls.count('add_scalar') <= (1 if operation == 'add_scalar' else (0 if operation == 'initialize' else 20))


def test_local_error_is_sticky_and_close_never_rewrites_broken_file(tmp_path):
    value = observer(tmp_path)
    calls = []
    def broken(record):
        calls.append(record)
        raise OSError('local storage failed')
    value._events.append = broken
    with pytest.raises(c.MahjongError, match='LOCAL_RECORD_FAILED'): value.event('bad', {}, None)
    with pytest.raises(c.MahjongError, match='LOCAL_RECORD_FAILED'): value.event('again', {}, None)
    with pytest.raises(c.MahjongError, match='LOCAL_RECORD_FAILED'): interrupted(value)
    assert len(calls) == 1
    assert not (value.directory/'summary.json').exists()


def test_terminal_summary_failure_leaves_only_unconfirmed_intent(tmp_path, monkeypatch):
    import mahjong.observability.events as events
    value = observer(tmp_path)
    original = events.atomic_json
    def fail_summary(path, payload):
        if path.name == 'summary.json': raise OSError('no space for summary')
        return original(path, payload)
    monkeypatch.setattr(events, 'atomic_json', fail_summary)
    with pytest.raises(c.MahjongError, match='LOCAL_RECORD_FAILED'): interrupted(value)
    status = json.loads((value.directory/'status.json').read_text())
    assert status['outcome'] == 'unconfirmed' and status['intended_outcome'] == 'interrupted'
    assert not (value.directory/'summary.json').exists()

