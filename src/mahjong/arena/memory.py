"""A process-wide model-player budget, shared by all seats and policies."""

import math
import weakref

from mahjong.contracts import MahjongError

GIB = 2**30
DEFAULT_GPU_MEMORY_GIB = 16.0


def validate_gpu_memory_gib(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise MahjongError('CONFIG_INVALID', 'gpu-memory-gib must be a finite positive number')
    return float(value)


class ModelMemoryBudget:
    def __init__(self, device='cpu', gpu_memory_gib=DEFAULT_GPU_MEMORY_GIB):
        self.device = device
        self.gpu_memory_gib = validate_gpu_memory_gib(gpu_memory_gib)
        self.limit_bytes = None
        self.caches = weakref.WeakSet()

    def register(self, cache):
        self.caches.add(cache)

    def refresh(self):
        if self.device != 'cuda:0':
            return
        import torch

        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        # Owned allocator reservations are reusable, not external consumption.
        margin = min(GIB, total // 10)
        requested = int(min(self.gpu_memory_gib, total / GIB) * GIB)
        limit = max(0, min(requested, total - margin, free + reserved - margin))
        working_room = min(256 * 2**20, limit // 8)
        for cache in sorted(self.caches, key=lambda c: c.last_used):
            if torch.cuda.memory_allocated(self.device) + working_room <= limit:
                break
            cache.clear(evicted=True)
        if reserved > limit or free < margin:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        if limit <= 0 or torch.cuda.memory_allocated(self.device) > limit:
            raise MahjongError('GPU_MEMORY_BUDGET_EXCEEDED',
                               'GPU model memory exceeds the current budget; free VRAM or increase --gpu-memory-gib')
        torch.cuda.set_per_process_memory_fraction(limit / total, self.device)
        self.limit_bytes = limit

    def snapshot(self):
        return dict(requested_gib=self.gpu_memory_gib,
                    effective_gib=None if self.limit_bytes is None else self.limit_bytes / GIB)
