"""Necessary attempt records, immutable heartbeat mirrors and terminal commits."""

from collections import deque
from dataclasses import replace
import json
import math
import sys
import threading
import time

from mahjong import contracts as c
from mahjong.evaluation.aggregation import MetricAccumulator
from .resources import process_rss, resource_view, utc_now
from .sinks import JsonlSink, TensorBoardSink, atomic_json, default_writer_factory


class LocalObserver:
    def __init__(self, context, config, *, clock=time.monotonic,
                 writer_factory=default_writer_factory, background=True):
        self.context, self.config, self.clock = context, config, clock
        self.settings = config['observability']
        self.training = isinstance(context, c.RunContext)
        self.directory = context.attempt_dir if self.training else context.evaluation_dir
        self.identity = c.RecordIdentity(
            'training' if self.training else 'evaluation',
            context.run_id if self.training else context.source_checkpoint.run_id,
            context.attempt_id if self.training else None,
            None if self.training else context.evaluation_id)
        self._lock = threading.RLock()
        self._halt = threading.Event()
        self._thread = None
        self._error = None
        self._closed = False
        self._stop = None
        self._started = clock()
        self._phase = 'initializing'
        self._phase_start = self._started
        self._phase_seconds = {}
        self._progress = c.encode_json_contract(c.TrainProgress()) if self.training else None
        self._event_seq = self._metric_seq = 0
        self._last_progress_at = None
        self._last_heartbeat_at = None
        self._last_checkpoint = context.source_checkpoint
        self._last_checkpoint_at = None
        self._degraded = []
        self._reports = {}
        self._accumulator = MetricAccumulator()
        self._window_start = 0
        self._next_record = self.settings['metrics_every_updates']
        self._last_rates = ()
        self._source = None
        self._inherited = None
        self._partial_written = False
        self._attempt_updates = self._attempt_samples = 0
        self._timings = deque(maxlen=100)
        self._resource = {'device': config['runtime']['device'], 'measured_at': utc_now(), 'values': {}}
        self._resource['values'] = {name: c.encode_json_contract(c.ResourceValue(
            None, 'bytes', 'unavailable', 'not_sampled', utc_now(), 0.0)) for name in ('allocated', 'reserved')}
        self._resource_at = self._started
        self._rss = process_rss()
        self._rss_at = self._started
        self._resource_tb_at = float('-inf')
        self._resource_states = {}
        self._tb = None
        self._events = self._metrics = None
        try:
            self._events = JsonlSink(self.directory / 'events.jsonl')
            self._metrics = JsonlSink(self.directory / 'metrics.jsonl')
            self._tb = TensorBoardSink(self.directory / 'visualization',
                enabled=self.settings['tensorboard_enabled'],
                flush_seconds=self.settings['tensorboard_flush_seconds'],
                factory=writer_factory, degraded=self._backend_degraded)
            self.event('attempt_started' if self.training else 'evaluation_context_started',
                       {'source_checkpoint': c.encode_json_contract(context.source_checkpoint)}, None)
            self._write_status()
        except Exception as exc:
            self._halt.set()
            error = self._local_error(exc)
            self._cleanup()
            raise error from exc
        if background:
            self._thread = threading.Thread(target=self._background, name='mahjong-observer', daemon=True)
            self._thread.start()

    def bind_stop(self, stop):
        with self._lock:
            self._stop = stop

    def _check(self):
        if self._error is not None:
            raise self._error
        if self._closed:
            raise c.MahjongError('LOCAL_RECORD_FAILED', 'Observer is closed', self._phase)

    def _local_error(self, exc):
        if self._error is None:
            self._error = exc if isinstance(exc, c.MahjongError) and exc.code == 'LOCAL_RECORD_FAILED' else c.MahjongError(
                'LOCAL_RECORD_FAILED', str(exc), self._phase,
                {'progress': self._progress, 'last_checkpoint': c.encode_json_contract(self._last_checkpoint)})
        self._halt.set()
        return self._error

    def _mirror(self, progress):
        if progress is not None and self.training:
            self._progress = c.encode_json_contract(progress)

    def _g(self):
        return self._progress['global_step'] if self.training else self.context.source_checkpoint.global_step

    def _backend_degraded(self, operation, reason):
        self._degraded.append({'backend': 'tensorboard', 'operation': operation, 'reason': reason})
        print(f'TensorBoard degraded: {operation}: {reason}', file=sys.stderr, flush=True)
        if self._error is None:
            self.event('backend_degraded', self._degraded[-1], None)

    def event(self, type, payload, progress):
        with self._lock:
            self._check()
            self._mirror(progress)
            event = c.Event(1, self.identity, self._event_seq+1, utc_now(), self._phase, self._g(), type, payload)
            try:
                self._events.append(c.encode_json_contract(event))
                self._event_seq += 1
                if type == 'checkpoint_saved':
                    self._last_checkpoint = c.decode_contract(c.CheckpointRef, payload['checkpoint'])
                    self._last_checkpoint_at = event.utc
            except Exception as exc:
                raise self._local_error(exc) from exc
            return self._event_seq

    def phase(self, name, progress):
        with self._lock:
            self._check()
            self._mirror(progress)
            if name == self._phase:
                # Heartbeat publishes the latest immutable mirror. Real phase
                # changes and terminal/checkpoint durability remain immediate.
                return
            now = self.clock()
            if name != self._phase:
                self._phase_seconds[self._phase] = self._phase_seconds.get(self._phase, 0.0) + now-self._phase_start
                self._phase, self._phase_start = name, now
                self.event('phase_changed', {'phase': name}, None)
            try:
                self._write_status()
            except Exception as exc:
                raise self._local_error(exc) from exc

    def performance(self):
        now = self.clock()
        elapsed = max(0.0, now-self._started)
        values = sorted(seconds for seconds, _ in self._timings)
        total = sum(values)
        stable = len(values) > 0 and total > 0
        speed = len(values)/total if stable else None
        target = self.config['training']['max_updates']
        phase_seconds = dict(self._phase_seconds)
        phase_seconds[self._phase] = phase_seconds.get(self._phase, 0.0) + max(0.0, now-self._phase_start)
        return {
            'attempt_elapsed_seconds': elapsed,
            'attempt_updates': self._attempt_updates, 'attempt_samples_updated': self._attempt_samples,
            'training_samples_per_second': sum(samples for _, samples in self._timings)/total if stable else None,
            'end_to_end_samples_per_second': self._attempt_samples/elapsed if self._attempt_samples and elapsed > 0 else None,
            'step_p50_seconds': values[math.ceil(.50*len(values))-1] if values else None,
            'step_p95_seconds': values[math.ceil(.95*len(values))-1] if values else None,
            'timing_samples': len(values), 'stable_reason': None if stable else 'warming_up',
            'eta_seconds': 0.0 if self._g() == target else (max(0, target-self._g())/speed if speed else None),
            'eta_scope': 'training_updates_only', 'phase_seconds': phase_seconds,
        }

    def _record(self, record, prefix, *, scalars=True):
        try:
            self._metrics.append(c.encode_json_contract(record))
            self._metric_seq += 1
        except Exception as exc:
            raise self._local_error(exc) from exc
        if scalars:
            for name, value in record.stats.items():
                self._tb.scalar(f'{prefix}/{name}', value.value, record.global_step)
            for index, value in enumerate(record.learning_rates):
                self._tb.scalar(f'train/lr/group_{index}', value, record.global_step)
            for name, value in record.performance.items():
                if type(value) in (float, int):
                    self._tb.scalar(f'performance/{name}', value, record.global_step)

    def _window_record(self, kind):
        return c.MetricRecord(1, self.identity, self._metric_seq+1, utc_now(), 'train', self._g(), kind,
            self._window_start, self._g(), self._source, self._inherited,
            self._accumulator.result(), self._last_rates, self.performance(), None, None)

    def record_step(self, result, progress, timing, lr):
        with self._lock:
            self._check()
            c.validate_step_result(result, phase='train')
            c.validate_contract(timing)
            samples = progress.samples_updated - self._progress['samples_updated']
            self._mirror(progress)
            if result.skip_reason is None:
                self._attempt_updates += 1
                self._attempt_samples += samples
                self._last_progress_at = utc_now()
                self._last_rates = tuple(lr)
                self._partial_written = False
                self._accumulator.add(result.metrics)
                if self._attempt_updates > 5 and timing.step_seconds is not None:
                    self._timings.append((timing.step_seconds, samples))
                if self._g() >= self._next_record:
                    self._record(self._window_record('window'), 'train')
                    self._accumulator.reset()
                    self._window_start = self._g()
                    self._next_record += self.settings['metrics_every_updates']
                    self._source = self._inherited = None

    def write_partial(self, progress):
        with self._lock:
            self._check()
            self._mirror(progress)
            if not self._partial_written and self._accumulator.result():
                self._record(self._window_record('partial'), 'train_partial')
                self._partial_written = True

    def record_evaluation(self, report):
        with self._lock:
            self._check()
            c.validate_contract(report)
            self._reports[report.report_ref] = report
            event_prefix = 'evaluation' if report.purpose == 'standalone' else 'validation'
            suffix = 'cancelled' if report.status == 'cancelled' else ('failed' if report.status == 'failed' else 'completed')
            self.event(f'{event_prefix}_{suffix}', {'evaluation_id': report.evaluation_id,
                'status': report.status, 'purpose': report.purpose, 'report_ref': report.report_ref,
                'reason': report.reason}, None)
            record = c.MetricRecord(1, self.identity, self._metric_seq+1, utc_now(), 'eval', report.model_source.global_step,
                'evaluation', None, None, self.context.source_checkpoint if not self.training else None,
                None, report.metrics, (), {'duration_seconds': report.duration_seconds}, report.evaluation_id, report.purpose)
            self._record(record, 'validation', scalars=report.status == 'ok')
            if report.status in {'not_evaluable', 'failed'}:
                print(f'{event_prefix}: {report.status}: {report.reason}', file=sys.stderr, flush=True)

    def publish_resources(self, snapshot):
        with self._lock:
            self._check()
            c.validate_contract(snapshot)
            self._resource = c.encode_json_contract(snapshot)
            self._resource_at = self.clock()
            view = self._resources(self._resource_at)
            submit = self._resource_at-self._resource_tb_at >= self.settings['resource_seconds']
            for name, item in view['values'].items():
                state = (item['status'], item['reason'])
                if state != self._resource_states.get(name):
                    self._resource_states[name] = state
                    if item['status'] != 'available':
                        self.event('resource_unavailable', {'resource': name, **item}, None)
                if submit and item['status'] == 'available':
                    self._tb.scalar(f'resource/{name}', item['value'], self._g())
            if submit:
                self._resource_tb_at = self._resource_at

    def _resources(self, now):
        rss = dict(self._rss)
        rss['age_seconds'] += max(0.0, now-self._rss_at)
        if rss['status'] == 'available' and rss['age_seconds'] > 2*self.settings['resource_seconds']:
            rss.update(status='stale', reason='cached_snapshot_expired')
        return resource_view(self._resource, self._resource_at, now, self.settings['resource_seconds'], rss=rss)

    def _write_status(self, *, intended_outcome=None):
        now = self.clock()
        target = self.config['training']['max_updates'] if self.training else None
        atomic_json(self.directory / 'status.json', {
            'schema_version': 1, 'identity': c.encode_json_contract(self.identity),
            'phase': self._phase, 'outcome': 'unconfirmed' if intended_outcome else 'running',
            'intended_outcome': intended_outcome, 'terminal_ref': 'summary.json' if intended_outcome else None,
            'stop_requested': self._stop.requested() if self._stop is not None else False,
            'progress': self._progress, 'target_updates': target,
            'completion_fraction': self._g()/target if target else None,
            'last_progress_at': self._last_progress_at, 'last_heartbeat_at': self._last_heartbeat_at,
            'last_checkpoint': c.encode_json_contract(self._last_checkpoint),
            'last_checkpoint_at': self._last_checkpoint_at,
            'source_checkpoint': c.encode_json_contract(self.context.source_checkpoint),
            'degraded_backends': list(self._degraded), 'performance': self.performance(),
            'resources': self._resources(now),
        })

    def heartbeat(self):
        with self._lock:
            self._check()
            self._last_heartbeat_at = utc_now()
            self.event('heartbeat', {'last_progress_at': self._last_progress_at}, None)
            try:
                self._write_status()
            except Exception as exc:
                raise self._local_error(exc) from exc

    def _background(self):
        next_heartbeat = self.clock()+self.settings['heartbeat_seconds']
        next_flush = self.clock()+self.settings['local_flush_seconds']
        next_resource = self.clock()+self.settings['resource_seconds']
        try:
            while not self._halt.wait(max(0.0, min(next_heartbeat, next_flush, next_resource)-self.clock())):
                now = self.clock()
                if now >= next_resource:
                    rss = process_rss()
                    with self._lock:
                        self._rss, self._rss_at = rss, now
                        state = (rss['status'], rss['reason'])
                        if state != self._resource_states.get('rss'):
                            self._resource_states['rss'] = state
                            if rss['status'] != 'available':
                                self.event('resource_unavailable', {'resource': 'rss', **rss}, None)
                    next_resource = now+self.settings['resource_seconds']
                if now >= next_heartbeat:
                    self.heartbeat()
                    next_heartbeat = now+self.settings['heartbeat_seconds']
                if now >= next_flush:
                    self.flush()
                    next_flush = now+self.settings['local_flush_seconds']
        except Exception as exc:
            with self._lock:
                self._local_error(exc)

    def flush(self, durable=False):
        with self._lock:
            self._check()
            try:
                self._events.flush(durable)
                self._metrics.flush(durable)
            except Exception as exc:
                raise self._local_error(exc) from exc
            self._tb.flush()
            return c.LogCutoff(self.context.attempt_id, self._event_seq, self._metric_seq) if self.training else None

    def state_dict(self):
        with self._lock:
            self._check()
            if not self.training:
                raise c.MahjongError('CONTRACT_MISMATCH', 'Evaluation observer has no training window')
            return c.WindowState(1, self._accumulator.state_dict(), self._window_start, self._next_record, self._last_rates)

    def load_state_dict(self, state, source):
        with self._lock:
            self._check()
            c.validate_contract(state)
            self._accumulator.load_state_dict(state.accumulator)
            self._window_start, self._next_record = state.window_start_step, state.next_record_step
            self._last_rates = tuple(state.last_learning_rates)
            self._source, self._inherited = source, source.global_step

    def _artifacts(self, completion):
        expected = [self.directory / name for name in ('resolved_config.yaml', 'manifest.json', 'events.jsonl', 'metrics.jsonl', 'status.json')]
        if completion.last_checkpoint is not None:
            expected += [self.context.run_dir / completion.last_checkpoint.path / name for name in ('manifest.json', 'state.pt')]
        expected += [self.context.run_dir / reference for reference in self._reports]
        result = {path.relative_to(self.context.run_dir).as_posix(): 'produced' if path.is_file() else 'missing' for path in expected}
        visual = self.directory / 'visualization'
        result[visual.relative_to(self.context.run_dir).as_posix()] = ('disabled' if not self.settings['tensorboard_enabled'] else ('produced' if visual.is_dir() else 'missing'))
        result[(self.directory / 'summary.json').relative_to(self.context.run_dir).as_posix()] = 'produced'
        missing = any(result[path.relative_to(self.context.run_dir).as_posix()] == 'missing' for path in expected)
        if completion.outcome == 'completed':
            report = self._reports.get(completion.report_ref)
            if self.training:
                missing |= completion.last_checkpoint is None or completion.last_checkpoint.attempt_id != self.context.attempt_id
                missing |= report is None or report.purpose != 'final' or report.status != ('ok' if self.config['validation']['enabled'] else 'disabled')
            else:
                missing |= report is None or report.status != 'ok'
        if completion.outcome == 'completed' and missing:
            print('LOCAL_RECORD_FAILED: required completion artifacts missing', file=sys.stderr, flush=True)
            completion = replace(completion, outcome='failed', exit_code=1, reason='required completion artifacts missing')
        return result, completion

    def _cleanup(self):
        if self._tb is not None:
            self._tb.close()
        for sink in (self._events, self._metrics):
            if sink is not None:
                sink.close()

    def close(self, completion):
        self._halt.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self._closed:
                raise c.MahjongError('LOCAL_RECORD_FAILED', 'Observer.close called twice', self._phase)
            try:
                self._check()
                artifacts, effective = self._artifacts(completion)
                self.flush(durable=True)
                self._tb.close()
                now = self.clock()
                self._phase_seconds[self._phase] = self._phase_seconds.get(self._phase, 0.0)+now-self._phase_start
                self._phase, self._phase_start = 'finalizing', now
                self.event('attempt_finished' if self.training else 'evaluation_finished',
                           {'intended_outcome': effective.outcome, 'terminal_ref': 'summary.json',
                            'reason': effective.reason, 'exit_code': effective.exit_code}, None)
                self.flush(durable=True)
                self._write_status(intended_outcome=effective.outcome)
                self._events.close()
                self._metrics.close()
                summary = c.Summary(1, self.identity, effective, utc_now(), artifacts)
                atomic_json(self.directory / 'summary.json', c.encode_json_contract(summary))
                return summary
            except Exception as exc:
                raise self._local_error(exc) from exc
            finally:
                self._cleanup()
                self._closed = True
