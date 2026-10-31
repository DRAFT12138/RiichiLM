"""Weighted metrics and finite evaluation, independent of the Trainer."""
from .aggregation import MetricAccumulator
from .runner import Evaluator

__all__ = ['Evaluator', 'MetricAccumulator']
