"""Finite evaluation receives ready components and never constructs a Trainer."""
from datetime import datetime, timezone
import time

from mahjong.contracts import (EvaluationReport, EvaluationRequest, EvalSource, MahjongError,
                              Observer, Runtime, StopToken, validate_step_result)
from .aggregation import MetricAccumulator
from .report import report_reference, write_report


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Evaluator:
    def __init__(self, *, clock=time.monotonic, report_writer=write_report):
        self.clock = clock
        self.report_writer = report_writer

    def _report(self, source, request, accumulator, *, status, reason, batches,
                samples, started_at, started_clock):
        return EvaluationReport(1, request.evaluation_id, status, request.purpose,
                                request.model_source, source.descriptor(), request.seed,
                                request.batch_size, request.precision, batches, samples,
                                accumulator.result(), reason, started_at, utc_now(),
                                max(0.0, self.clock()-started_clock), request.config_hash,
                                request.environment_ref, report_reference(request.report_path))

    def _publish(self, report, request, observer):
        self.report_writer(report, request.report_path)
        observer.record_evaluation(report)
        return report

    def run(self, runtime: Runtime, source: EvalSource, request: EvaluationRequest,
            observer: Observer, stop: StopToken) -> EvaluationReport:
        started_at, started_clock = utc_now(), self.clock()
        observer.phase('evaluating', None)
        observer.event('evaluation_started' if request.purpose == 'standalone' else 'validation_started',
                       {'evaluation_id': request.evaluation_id, 'purpose': request.purpose,
                        'global_step': request.model_source.global_step}, None)
        accumulator = MetricAccumulator()
        batches = samples = 0
        status, reason = 'ok', None
        if stop.requested():
            report = self._report(source, request, accumulator, status='cancelled',
                                  reason='stop_requested', batches=0, samples=0,
                                  started_at=started_at, started_clock=started_clock)
            return self._publish(report, request, observer)
        try:
            with runtime.evaluation_scope(request.seed):
                for batch in source.iter_batches():
                    if stop.requested():
                        status, reason = 'cancelled', 'stop_requested'
                        break
                    result = runtime.compute(batch, phase='eval')
                    validate_step_result(result, phase='eval')
                    accumulator.add(result.metrics)
                    batches += 1
                    samples += batch.sample_count
                    snapshot = runtime.resource_snapshot()
                    try:
                        observer.publish_resources(snapshot)
                    except OSError as exc:
                        raise MahjongError('LOCAL_RECORD_FAILED', str(exc), 'evaluating') from exc
                if stop.requested():
                    status, reason = 'cancelled', 'stop_requested'
        except Exception as exc:
            if isinstance(exc, MahjongError) and exc.code == 'LOCAL_RECORD_FAILED':
                raise
            # State protection has already exited through Runtime's finally.
            report = self._report(source, request, accumulator, status='failed', reason=str(exc),
                                  batches=batches, samples=samples, started_at=started_at,
                                  started_clock=started_clock)
            self._publish(report, request, observer)
            raise MahjongError('EVALUATION_FAILED', f'Evaluation computation failed: {exc}',
                              'evaluating', {'evaluation_id': request.evaluation_id,
                                             'report_ref': report.report_ref}) from exc
        if status == 'ok' and not any(value.denominator for value in accumulator.result().values()):
            status, reason = 'not_evaluable', 'empty_source' if batches == 0 else 'no_valid_statistics'
        report = self._report(source, request, accumulator, status=status, reason=reason,
                              batches=batches, samples=samples, started_at=started_at,
                              started_clock=started_clock)
        return self._publish(report, request, observer)

    def disabled(self, source: EvalSource, request: EvaluationRequest,
                 observer: Observer) -> EvaluationReport:
        started_at, started_clock = utc_now(), self.clock()
        report = self._report(source, request, MetricAccumulator(), status='disabled',
                              reason='validation_disabled', batches=0, samples=0,
                              started_at=started_at, started_clock=started_clock)
        return self._publish(report, request, observer)
