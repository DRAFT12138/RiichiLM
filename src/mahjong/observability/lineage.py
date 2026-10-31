"""Read-only branch projection, with whole damaged stream intervals excluded."""
import json
from pathlib import Path

from mahjong.contracts import Event, MetricRecord, LogCutoff, decode_contract


def _read_stream(path, run_id, attempt_id, stream, cutoff):
    sequence = 'metric_seq' if stream == 'metrics' else 'event_seq'
    cls = MetricRecord if stream == 'metrics' else Event
    if cutoff == 0:
        return [], None
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        return [], f'cannot_read_stream: {exc}'
    records = []
    for index, line in enumerate(lines):
        if cutoff is not None and len(records) == cutoff:
            break  # Corruption beyond the checkpoint cutoff is irrelevant.
        try:
            record = json.loads(line)
            decoded = decode_contract(cls, record)
            if (decoded.identity.kind != 'training' or decoded.identity.run_id != run_id or decoded.identity.attempt_id != attempt_id
                    or record[sequence] != len(records)+1):
                return [], 'identity_or_sequence_mismatch'
            records.append(record)
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            if index == len(lines)-1 and not line.endswith(b'\n'):
                break  # Only the incomplete final line of a crashed writer is ignorable.
            return [], f'corrupt_record: {exc}'
    if cutoff is not None and len(records) != cutoff:
        return [], 'checkpoint_cutoff_unreachable'
    return records, None


def read_branch(run_dir, attempt_id, stream='metrics'):
    """Return records, history_incomplete, and structured missing_intervals.

    Each ancestor contributes exactly its persisted sequence cutoff; current
    records keep their own attempt identity. No input file is rewritten.
    """
    if stream not in {'metrics', 'events'}:
        raise ValueError('stream must be metrics or events')
    run_dir = Path(run_dir)
    run_id = json.loads((run_dir / 'run.json').read_text())['run_id']
    intervals, records = [], []
    directory = run_dir / 'attempts' / attempt_id
    ancestors = []
    try:
        payload = json.loads((directory / 'lineage.json').read_text())
        if payload['schema_version'] != 1 or not isinstance(payload['lineage'], list):
            raise ValueError('lineage version/shape invalid')
        ancestors = [decode_contract(LogCutoff, value) for value in payload['lineage']]
        ids = [item.attempt_id for item in ancestors]
        if len(set(ids)) != len(ids) or attempt_id in ids:
            raise ValueError('duplicate or self-referencing ancestor')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        intervals.append({'attempt_id': attempt_id, 'stream': 'lineage', 'cutoff': None,
                          'reason': f'cannot_read_lineage: {exc}'})
        ancestors = []
    for ancestor in ancestors:
        cutoff = ancestor.metric_seq if stream == 'metrics' else ancestor.event_seq
        found, reason = _read_stream(run_dir / 'attempts' / ancestor.attempt_id / (stream+'.jsonl'),
                                     run_id, ancestor.attempt_id, stream, cutoff)
        if reason:
            intervals.append({'attempt_id': ancestor.attempt_id, 'stream': stream,
                              'cutoff': cutoff, 'reason': reason})
        else:
            records.extend(found)
    found, reason = _read_stream(directory / (stream+'.jsonl'), run_id, attempt_id, stream, None)
    if reason:
        intervals.append({'attempt_id': attempt_id, 'stream': stream, 'cutoff': None, 'reason': reason})
    else:
        records.extend(found)
    return {'records': records, 'history_incomplete': bool(intervals), 'missing_intervals': intervals}


def window_totals(records):
    """Sum completed windows only; overlapping partial previews are excluded."""
    result = {}
    for record in records:
        if record['kind'] != 'window':
            continue
        for name, metric in record['stats'].items():
            prior = result.setdefault(name, {'numerator': 0.0, 'denominator': 0, 'unit': metric['unit']})
            if prior['unit'] != metric['unit']:
                raise ValueError(f'Metric {name} unit changed')
            prior['numerator'] += metric['numerator']
            prior['denominator'] += metric['denominator']
    return result
