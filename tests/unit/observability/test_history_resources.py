from copy import deepcopy
import json

import pytest

from mahjong import contracts as c
from mahjong.app.config import DEFAULTS
from mahjong.app.runs import atomic_json, create_attempt, create_run
from mahjong.observability.lineage import read_branch, window_totals
from mahjong.observability.resources import process_rss, resource_view
from tests.unit.observability.test_observer import Clock, interrupted, observer


def write_metrics(context, steps, *, partial=None):
    identity = c.RecordIdentity('training', context.run_id, context.attempt_id, None)
    records = []
    for seq, step in enumerate(steps, 1):
        record = c.MetricRecord(1, identity, seq, '2026-10-11T12:00:00+00:00', 'train', step,
            'partial' if step == partial else 'window', step-10, step, None, None,
            {'loss': c.AggregatedMetric(20., 10, 2., 'x', None)}, (), {}, None, None)
        records.append(c.encode_json_contract(record))
    (context.attempt_dir/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    return records


def branch_fixture(tmp_path):
    config = deepcopy(DEFAULTS); config['output']['root'] = str(tmp_path)
    a1 = create_run(config); write_metrics(a1, range(10, 101, 10))
    source = c.CheckpointRef('C60', a1.run_id, a1.attempt_id, 60, 'checkpoints/C60', 'hash', 1)
    a2 = create_attempt(a1.run_dir, source, config); write_metrics(a2, [70, 80, 90])
    cut1 = c.LogCutoff(a1.attempt_id, 0, 6)
    atomic_json(a2.attempt_dir/'lineage.json', {'schema_version': 1, 'lineage': [cut1]})
    source2 = c.CheckpointRef('C80', a1.run_id, a2.attempt_id, 80, 'checkpoints/C80', 'hash', 1)
    a3 = create_attempt(a1.run_dir, source2, config); write_metrics(a3, [90, 100, 110], partial=110)
    atomic_json(a3.attempt_dir/'lineage.json', {'schema_version': 1,
        'lineage': [cut1, c.LogCutoff(a2.attempt_id, 0, 2)]})
    sibling = create_attempt(a1.run_dir, source, config); write_metrics(sibling, [70])
    atomic_json(sibling.attempt_dir/'lineage.json', {'schema_version': 1, 'lineage': [cut1]})
    return a1, a2, a3, sibling


def test_branch_cutoffs_do_not_count_rolled_back_or_partial_points(tmp_path):
    a1, a2, a3, sibling = branch_fixture(tmp_path)
    found = read_branch(a1.run_dir, a3.attempt_id)
    assert not found['history_incomplete']
    assert [r['global_step'] for r in found['records']] == list(range(10, 111, 10))
    assert window_totals(found['records'])['loss'] == {'numerator': 200., 'denominator': 100, 'unit': 'x'}
    other = read_branch(a1.run_dir, sibling.attempt_id)
    assert [r['global_step'] for r in other['records']] == list(range(10, 71, 10))
    assert other['records'][-1]['identity']['attempt_id'] == sibling.attempt_id
    assert found['records'][6]['identity']['attempt_id'] == a2.attempt_id


@pytest.mark.parametrize('damage', ['missing', 'middle', 'sequence', 'identity', 'short'])
def test_damaged_ancestor_excludes_whole_interval_but_preserves_other_streams(tmp_path, damage):
    a1, a2, a3, _ = branch_fixture(tmp_path)
    path = a1.attempt_dir/'metrics.jsonl'
    rows = path.read_text().splitlines()
    if damage == 'missing': path.unlink()
    else:
        if damage == 'middle': rows[2] = '{broken'
        elif damage == 'short': rows = rows[:5]
        else:
            record = json.loads(rows[2])
            if damage == 'sequence': record['metric_seq'] = 99
            else: record['identity']['run_id'] = 'another-run'
            rows[2] = json.dumps(record)
        path.write_text('\n'.join(rows)+'\n')
    found = read_branch(a1.run_dir, a3.attempt_id)
    assert found['history_incomplete'] and len(found['missing_intervals']) == 1
    assert found['missing_intervals'][0]['attempt_id'] == a1.attempt_id
    assert [r['global_step'] for r in found['records']] == [70, 80, 90, 100, 110]


def test_corruption_after_cutoff_and_truncated_final_line_are_ignored(tmp_path):
    a1, _, a3, _ = branch_fixture(tmp_path)
    rows = (a1.attempt_dir/'metrics.jsonl').read_text().splitlines()
    rows[6] = '{broken after cutoff'
    (a1.attempt_dir/'metrics.jsonl').write_text('\n'.join(rows)+'\n')
    with (a3.attempt_dir/'metrics.jsonl').open('a') as stream: stream.write('{incomplete')
    assert not read_branch(a1.run_dir, a3.attempt_id)['history_incomplete']
    with (a3.attempt_dir/'metrics.jsonl').open('a') as stream: stream.write('\n')
    damaged = read_branch(a1.run_dir, a3.attempt_id)
    assert damaged['history_incomplete']
    assert all(r['identity']['attempt_id'] != a3.attempt_id for r in damaged['records'])


def test_resource_snapshot_ages_without_mutating_original(tmp_path):
    clock = Clock(); value = observer(tmp_path, clock=clock)
    available = c.ResourceValue(1234, 'bytes', 'available', None, '2026-10-11T12:00:00+00:00', 0.)
    unavailable = c.ResourceValue(None, 'bytes', 'unavailable', 'cpu_device', available.measured_at, 0.)
    snapshot = c.ResourceSnapshot('cpu', available.measured_at, {'allocated': available, 'reserved': unavailable})
    value.publish_resources(snapshot)
    clock.now = 4.
    assert value._resources(clock.now)['values']['allocated']['status'] == 'available'
    clock.now = 4.01
    stale = value._resources(clock.now)
    assert stale['values']['allocated']['status'] == 'stale'
    assert stale['values']['allocated']['age_seconds'] == 4.01
    assert stale['values']['reserved']['value'] is None
    assert stale['values']['reserved']['status'] == 'unavailable'
    assert snapshot.values['allocated'].status == 'available'
    assert value._resource['values']['allocated']['age_seconds'] == 0.
    interrupted(value)


def test_real_rss_and_unavailable_interface_have_explicit_values(monkeypatch):
    measured = process_rss()
    assert measured['status'] == 'available' and measured['value'] > 0
    import mahjong.observability.resources as resources
    def fail(self): raise OSError('proc interface missing')
    monkeypatch.setattr(resources.Path, 'read_text', fail)
    unavailable = process_rss()
    assert unavailable['status'] == 'unavailable' and unavailable['value'] is None
    assert unavailable['reason'] == 'proc interface missing'


def test_heartbeat_uses_immutable_progress_and_keeps_last_progress_time(tmp_path):
    value = observer(tmp_path, clock=Clock())
    progress = c.TrainProgress(global_step=3, samples_updated=12)
    value.phase('evaluating', progress)
    progress.global_step = 99; progress.samples_updated = 999
    value.heartbeat()
    status = json.loads((value.directory/'status.json').read_text())
    assert status['progress']['global_step'] == 3
    assert status['progress']['samples_updated'] == 12
    assert status['last_progress_at'] is None and status['last_heartbeat_at'] is not None
    interrupted(value)
