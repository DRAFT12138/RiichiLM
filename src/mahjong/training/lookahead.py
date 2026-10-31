"""One optional CPU batch preview; no worker, device work or durable state."""


class CPUBatchLookahead:
    def __init__(self, source, runtime):
        self.source, self.runtime = source, runtime

    @classmethod
    def create(cls, source, runtime):
        source_support = getattr(source, 'supports_batch_preview', None)
        runtime_support = getattr(runtime, 'supports_cpu_lookahead', None)
        if (source_support is not None and runtime_support is not None and
                source_support() and runtime_support()):
            return cls(source, runtime)
        return None

    def prepare(self):
        if not self.source.supports_batch_preview():
            self.clear()
            return
        try:
            batch = self.source.preview_next_batch()
        except Exception:
            # Source retains its error for acquire; current update can finish.
            return
        self.runtime.prepare_cpu_batch(batch)

    def clear(self):
        self.source.discard_batch_preview()
        self.runtime.discard_batch_preview()
