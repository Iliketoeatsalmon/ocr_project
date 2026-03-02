"""Image preprocessing routines per ROI type.

These functions intentionally start simple and should be tuned using real data.
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess_for_serial(img: np.ndarray) -> np.ndarray:
    """Preprocess ROI A (Serial Number).

    Current baseline:
    1. Grayscale
    2. Small denoising blur
    3. Otsu binary threshold

    TODO:
    - Tune morphology and threshold strategy for engraved or low-contrast text.
    """
    gray = _to_grayscale(img)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def preprocess_for_model(img: np.ndarray) -> np.ndarray:
    """Preprocess ROI B (Model).

    TODO:
    - Evaluate adaptive thresholding and sharpening for font variations.
    """
    gray = _to_grayscale(img)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def preprocess_for_batch(img: np.ndarray) -> np.ndarray:
    """Preprocess ROI C (Batch or Part No).

    TODO:
    - Add optional inversion and morphology for dot-matrix style labels.
    """
    gray = _to_grayscale(img)
    blur = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
