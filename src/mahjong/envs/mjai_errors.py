"""Shared MJAI validation errors; no persistence or source dependencies."""


class CleaningError(ValueError):
    def __init__(self, code, reason, *, stage='validation', raw_line=None,
                 round_id=None, event_id=None):
        self.code, self.reason, self.stage = code, reason, stage
        self.raw_line, self.round_id, self.event_id = raw_line, round_id, event_id
        super().__init__(f'{code}: {reason}')

    def record(self):
        return {k: getattr(self, k) for k in
                ('code', 'reason', 'stage', 'raw_line', 'round_id', 'event_id')}


def require(condition, code, reason):
    if not condition:
        raise CleaningError(code, reason)
