"""Shared, deliberately small contract doubles; no concrete module imports."""

from .doubles import (
    KnownEvalSource, KnownEvaluator, KnownMetricAccumulator, KnownModel, KnownRuntime, KnownSource,
    MemoryStore, MutableStopToken, RecordingObserver, RecordingRunner,
    drive_evaluation, drive_training, make_context, make_metadata, make_request,
    make_snapshot, value_object_fixtures,
)
