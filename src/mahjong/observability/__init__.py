"""Attempt-local structured records and optional TensorBoard visualization."""
from .events import LocalObserver
from .lineage import read_branch

__all__ = ['LocalObserver', 'read_branch']
