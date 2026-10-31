"""CPU-only background sampling; device values are immutable Runtime snapshots."""
from datetime import datetime, timezone
from pathlib import Path
import os


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def process_rss():
    measured = utc_now()
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                value = int(line.split()[1]) * 1024
                return {'value': value, 'unit': 'bytes', 'status': 'available',
                        'reason': None, 'measured_at': measured, 'age_seconds': 0.0}
        raise OSError('VmRSS not provided')
    except (OSError, ValueError) as exc:
        return {'value': None, 'unit': 'bytes', 'status': 'unavailable',
                'reason': str(exc), 'measured_at': measured, 'age_seconds': 0.0}


def resource_view(snapshot, received_at, now, resource_seconds, *, rss=None):
    age = max(0.0, now-received_at)
    values = {}
    for name, original in snapshot['values'].items():
        item = dict(original)
        item['age_seconds'] = original['age_seconds'] + age
        if item['status'] == 'available' and item['age_seconds'] > 2*resource_seconds:
            item.update(status='stale', reason='cached_snapshot_expired')
        values[name] = item
    values['rss'] = process_rss() if rss is None else rss
    return {'device': snapshot['device'], 'measured_at': snapshot['measured_at'], 'values': values}
