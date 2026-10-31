"""Ordered MJAI event records, independent of source parsing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceEvent:
    raw_line: int
    value: dict
