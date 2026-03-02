"""High-level full-frame OCR pipeline."""

from __future__ import annotations

import time

import cv2

from core.ocr_models import run_fullframe_ocr


def run_full_pipeline(image_path: str) -> dict:
    """Run the full-frame OCR pipeline for one image path.

    Args:
        image_path: Input image path.

    Returns:
        {
            "time_ms": <int>,
            "roi_a": {"label": "UPC",   "text": <str>, "conf": <float>},
            "roi_b": {"label": "S/N",   "text": <str>, "conf": <float>},
            "roi_c": {"label": "BATCH", "text": <str>, "conf": <float>},
            "lighting_ok": True
        }

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    start = time.perf_counter()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    fields = run_fullframe_ocr(image)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "time_ms": elapsed_ms,
        "roi_a": fields.get("roi_a", {"label": "UPC", "text": "", "conf": 0.0}),
        "roi_b": fields.get("roi_b", {"label": "S/N", "text": "", "conf": 0.0}),
        "roi_c": fields.get("roi_c", {"label": "BATCH", "text": "", "conf": 0.0}),
        "lighting_ok": True,
    }
