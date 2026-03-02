"""Metrics helpers for timing, lighting quality, and confidence aggregation.

These values are intended for C# UI display:
- Total pipeline time (`time_ms`)
- Lighting status (`lighting_ok`)
- Per-ROI OCR confidence (`conf`)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TypeVar

import cv2
import numpy as np

T = TypeVar("T")


def measure_processing_time(func: Callable[..., T]) -> Callable[..., tuple[T, int]]:
    """Decorator that returns `(result, elapsed_ms)` for the wrapped function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return result, elapsed_ms

    return wrapper


def estimate_lighting_ok(
    img: np.ndarray,
    min_brightness: float = 50.0,
    max_brightness: float = 205.0,
    min_contrast: float = 20.0,
) -> bool:
    """Estimate whether lighting is acceptable using brightness and contrast.

    Args:
        img: Input image.
        min_brightness: Lower mean grayscale threshold.
        max_brightness: Upper mean grayscale threshold.
        min_contrast: Lower grayscale standard deviation threshold.

    Returns:
        True if basic brightness and contrast checks pass; otherwise False.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    return (min_brightness <= brightness <= max_brightness) and (contrast >= min_contrast)


def aggregate_confidence(confidences: Sequence[float]) -> float:
    """Return average confidence over a list of ROI confidence scores."""
    if not confidences:
        return 0.0
    values = [max(0.0, min(1.0, float(c))) for c in confidences]
    return float(sum(values) / len(values))
