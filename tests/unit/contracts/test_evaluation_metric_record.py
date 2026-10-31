"""Evaluation records use null window boundaries in the public schema."""

from mahjong import contracts as c


def test_evaluation_record_with_null_training_window_roundtrips():
    record = c.MetricRecord(1, c.RecordIdentity('evaluation', 'R', None, 'E'), 1,
        '2026-10-11T00:00:00Z', 'eval', 0, 'evaluation', None, None, None, None,
        {}, (), {}, 'E', 'standalone')
    assert c.decode_contract(c.MetricRecord, c.encode_json_contract(record)) == record

