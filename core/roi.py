"""ROI configuration loading and cropping utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_roi_config(config_path: str = "config/roi_config.json") -> dict:
    """Load ROI definitions from a JSON file.

    Args:
        config_path: Path to ROI config JSON.

    Returns:
        Parsed ROI configuration dictionary.

    Raises:
        FileNotFoundError: If config file is missing.
        ValueError: If config JSON is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"ROI config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ROI config JSON in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("ROI config root must be a JSON object.")

    return config


def crop_roi(image: np.ndarray, roi_def: dict) -> np.ndarray:
    """Crop a single ROI from an image with bounds validation.

    Args:
        image: Source image (H x W x C or H x W).
        roi_def: ROI dictionary with `x`, `y`, `w`, and `h`.

    Returns:
        Cropped ROI image array.

    Raises:
        ValueError: If ROI coordinates are invalid or out of bounds.
    """
    required = ("x", "y", "w", "h")
    for key in required:
        if key not in roi_def:
            raise ValueError(f"Missing ROI field '{key}' in definition: {roi_def}")

    x = int(roi_def["x"])
    y = int(roi_def["y"])
    w = int(roi_def["w"])
    h = int(roi_def["h"])

    if w <= 0 or h <= 0:
        raise ValueError(f"ROI width/height must be > 0. Received w={w}, h={h}")

    img_h, img_w = image.shape[:2]
    x2 = x + w
    y2 = y + h

    if x < 0 or y < 0 or x2 > img_w or y2 > img_h:
        raise ValueError(
            "ROI is out of image bounds: "
            f"(x={x}, y={y}, w={w}, h={h}) vs image (w={img_w}, h={img_h})"
        )

    return image[y:y2, x:x2].copy()


def get_all_rois(image: np.ndarray, config: dict) -> dict:
    """Crop all configured ROIs and return a structured dictionary.

    Args:
        image: Source image.
        config: ROI configuration dictionary.

    Returns:
        {
            "roi_a": {"label": "S/N",   "image": <np.ndarray>},
            "roi_b": {"label": "MODEL", "image": <np.ndarray>},
            "roi_c": {"label": "BATCH", "image": <np.ndarray>}
        }
    """
    roi_map: dict = {}
    for roi_key, roi_def in config.items():
        label = str(roi_def.get("label", roi_key))
        roi_img = crop_roi(image, roi_def)
        roi_map[roi_key] = {"label": label, "image": roi_img}

    return roi_map
