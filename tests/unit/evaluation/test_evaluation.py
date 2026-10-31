from dataclasses import replace
import json

import pytest
import torch

from mahjong import contracts as c
from mahjong.evaluation.aggregation import MetricAccumulator
from mahjong.evaluation.runner import Evaluator
from tests.fixtures.contracts.doubles import KnownEvalSource, KnownRuntime, KnownModel, RecordingObserver, MutableStopToken


def request(tmp_path, purpose='initial'):
    return c.EvaluationRequest('E', purpose, c.ModelSource('R', 'A', 3, None), 43, 2, '32-true',
        tmp_path/'R/reports/A/E.json', 'config', 'attempts/A/manifest.json')


def runtime():
    value = KnownRuntime()
    value.setup(KnownModel(), None)
    return value


def test_weighted_statistics_and_independent_counts():
    aggregate = MetricAccumulator()
    aggregate.add({'loss': c.MetricStat(6., 3, 'squared'), 'empty': c.MetricStat(0., 0, 'x')})
    aggregate.add({'loss': c.MetricStat(4., 1, 'squared'), 'other': c.MetricStat(3., 2, 'y')})
    result = aggregate.result()
    assert result['loss'].value == 2.5 and result['loss'].denominator == 4
    assert result['empty'].value is None and result['empty'].reason == 'no_valid_statistics'
    assert result['other'].value == 1.5 and result['other'].denominator == 2
    state = aggregate.state_dict()
    state['metrics']['loss']['numerator'] = 20.
    assert aggregate.result()['loss'].numerator == 10.
    restored = MetricAccumulator(); restored.load_state_dict(aggregate.state_dict())
    assert restored.result() == result


@pytest.mark.parametrize('case', ['unit', 'negative', 'nan', 'inf', 'nonzero-empty'])
def test_invalid_statistic_is_rejected_without_partial_mutation(case):
    aggregate = MetricAccumulator(); aggregate.add({'loss': c.MetricStat(1., 1, 'x')})
    state = aggregate.state_dict()
    with pytest.raises(c.MahjongError):
        if case == 'unit': value = c.MetricStat(1., 1, 'changed')
        elif case == 'negative': value = c.MetricStat(1., -1, 'x')
        elif case == 'nan': value = c.MetricStat(float('nan'), 1, 'x')
        elif case == 'inf': value = c.MetricStat(float('inf'), 1, 'x')
        else: value = c.MetricStat(1., 0, 'x')
        aggregate.add({'loss': value})
    assert aggregate.state_dict() == state


@pytest.mark.parametrize('status', ['ok', 'empty', 'invalid', 'partial', 'cancelled', 'disabled', 'failed'])
def test_reports_status_identity_counts_and_atomic_file(tmp_path, status):
    source = KnownEvalSource()
    value = runtime()
    observer, stop = RecordingObserver(), MutableStopToken()
    evaluation = Evaluator()
    if status == 'empty': source.iter_batches = lambda: iter(())
    elif status in {'invalid', 'partial'}:
        value.compute = lambda batch, phase: c.StepResult(None, {
            'empty': c.MetricStat(0., 0, 'x'), **({'valid': c.MetricStat(4., 2, 'y')} if status == 'partial' else {})})
    elif status == 'cancelled': stop.request(2)
    elif status == 'failed':
        def fail(batch, *, phase): raise ValueError('bad computation')
        value.compute = fail
    if status == 'disabled': report = evaluation.disabled(source, request(tmp_path), observer)
    elif status == 'failed':
        with pytest.raises(c.MahjongError, match='EVALUATION_FAILED'):
            evaluation.run(value, source, request(tmp_path), observer, stop)
        report = c.decode_contract(c.EvaluationReport, json.loads(request(tmp_path).report_path.read_text()))
    else: report = evaluation.run(value, source, request(tmp_path), observer, stop)
    expected = {'empty': 'not_evaluable', 'invalid': 'not_evaluable', 'partial': 'ok'}.get(status, status)
    assert report.status == expected
    assert report.model_source.global_step == 3 and report.model_source.checkpoint_id is None
    assert report.report_ref == 'reports/A/E.json'
    assert report.seed == 43 and report.batch_size == 2
    assert json.loads(request(tmp_path).report_path.read_text())['status'] == expected
    assert not list(request(tmp_path).report_path.parent.glob('*.tmp-*'))
    if status in {'empty', 'invalid'}:
        assert report.reason == ('empty_source' if status == 'empty' else 'no_valid_statistics')
    if status in {'cancelled', 'disabled', 'empty'}: assert report.samples == report.batches == 0
    if status in {'ok', 'invalid', 'partial'}: assert report.samples == 2 and report.batches == 1


def test_required_record_failure_is_not_retried_as_a_failed_report(tmp_path):
    writes = []
    def writer(report, path): writes.append(report)
    class BrokenObserver(RecordingObserver):
        def publish_resources(self, snapshot):
            raise c.MahjongError('LOCAL_RECORD_FAILED', 'no space', 'evaluating')
    with pytest.raises(c.MahjongError, match='LOCAL_RECORD_FAILED'):
        Evaluator(report_writer=writer).run(runtime(), KnownEvalSource(), request(tmp_path), BrokenObserver(), MutableStopToken())
    assert writes == []


def test_report_write_failure_has_no_ok_file_or_second_attempt(tmp_path):
    writes = []
    def writer(report, path):
        writes.append(report)
        raise OSError('write failure')
    with pytest.raises(OSError):
        Evaluator(report_writer=writer).run(runtime(), KnownEvalSource(), request(tmp_path), RecordingObserver(), MutableStopToken())
    assert len(writes) == 1 and not request(tmp_path).report_path.exists()

