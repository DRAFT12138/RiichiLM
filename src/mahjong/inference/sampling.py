"""Pure-Python inference sampling defaults and validation (torch-free)."""
from __future__ import annotations

import math
from typing import Any

from mahjong.contracts import MahjongError

DEFAULT_INFERENCE_TEMPERATURE = 0.7
DEFAULT_INFERENCE_TOP_P = 0.9


def validate_sampling_params(temperature: Any, top_p: Any, *, phase: str = 'initializing') -> tuple[float, float]:
    """Validate model inference temperature in [0, 2] and top_p in (0, 1]."""
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0.0 <= float(temperature) <= 2.0
    ):
        raise MahjongError('CONFIG_INVALID', 'temperature 必须为 [0, 2] 内的有限数字', phase)
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(top_p)
        or not 0.0 < float(top_p) <= 1.0
    ):
        raise MahjongError('CONFIG_INVALID', 'top_p 必须为 (0, 1] 内的有限数字', phase)
    return float(temperature), float(top_p)
