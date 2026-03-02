"""EasyOCR full-frame OCR and pattern-based field classification."""

from __future__ import annotations

import contextlib
import io
import re
from typing import Any

import numpy as np

try:
    import easyocr
except Exception as exc:  # pragma: no cover - runtime environment dependent
    easyocr = None  # type: ignore[assignment]
    _EASYOCR_IMPORT_ERROR: Exception | None = exc
else:
    _EASYOCR_IMPORT_ERROR = None

EASYOCR_READER: Any | None = None
_EASYOCR_INIT_ERROR: Exception | None = None

if easyocr is not None:
    try:
        # Initialize once at module level (required).
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _EASYOCR_INIT_ERROR = exc
        EASYOCR_READER = None

_SERIAL_WORD_RE = re.compile(r"\bserial\b", re.IGNORECASE)
_PART_WORD_RE = re.compile(r"\bpart\b", re.IGNORECASE)
_UPC_EXACT_RE = re.compile(r"^upc$", re.IGNORECASE)
_PART_PATTERN_RE = re.compile(r"(?<!\d)(\d{3,6}(?:[-\s]\d{2,6}){2,5})(?!\d)")
_SERIAL_DIGITS_RE = re.compile(r"(?<!\d)(\d[\d\s-]{8,20}\d|\d{10,16})(?!\d)")


def _ensure_reader() -> Any:
    """Return initialized EasyOCR reader or raise a clear runtime error."""
    if EASYOCR_READER is not None:
        return EASYOCR_READER

    if _EASYOCR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "EasyOCR import failed. "
            f"Cause: {_EASYOCR_IMPORT_ERROR}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from _EASYOCR_IMPORT_ERROR

    if _EASYOCR_INIT_ERROR is not None:
        raise RuntimeError(
            f"EasyOCR reader initialization failed. Cause: {_EASYOCR_INIT_ERROR}"
        ) from _EASYOCR_INIT_ERROR

    raise RuntimeError("EasyOCR reader is unavailable for an unknown reason.")


def _safe_conf(value: Any) -> float:
    """Convert confidence to a bounded 0..1 float."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = 0.0
    return max(0.0, min(1.0, conf))


def _normalize_space(text: str) -> str:
    """Normalize repeated whitespace to single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _digits_only(text: str) -> str:
    """Return text containing digits only."""
    return re.sub(r"\D", "", text)


def _format_part_number(text: str) -> str:
    """Normalize part-number separators to single dashes."""
    groups = [grp for grp in re.split(r"[-\s]+", text) if grp]
    return "-".join(groups)


def _to_candidates(read_results: list[Any]) -> list[dict]:
    """Convert EasyOCR tuples into normalized candidate dictionaries."""
    candidates: list[dict] = []
    for row in read_results:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue

        bbox, raw_text, raw_conf = row[0], row[1], row[2]
        text = _normalize_space(str(raw_text))
        if not text:
            continue

        # Ignore short garbage strings, except exact "UPC".
        if len(text) < 4 and not _UPC_EXACT_RE.fullmatch(text):
            continue

        conf = _safe_conf(raw_conf)
        points: list[list[float]] = []
        if isinstance(bbox, (list, tuple)):
            for point in bbox:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append([float(point[0]), float(point[1])])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_center = float(sum(xs) / len(xs)) if xs else 0.0
        y_center = float(sum(ys) / len(ys)) if ys else 0.0
        height = float(max(ys) - min(ys)) if len(ys) >= 2 else 0.0

        candidates.append(
            {
                "text": text,
                "conf": conf,
                "bbox": points,
                "x_center": x_center,
                "y_center": y_center,
                "height": height,
            }
        )

    return candidates


def _group_lines(candidates: list[dict]) -> list[dict]:
    """Group OCR candidates into approximate text lines using y-center."""
    if not candidates:
        return []

    heights = [c["height"] for c in candidates if c["height"] > 0]
    median_h = float(np.median(heights)) if heights else 20.0
    y_tol = max(12.0, 0.8 * median_h)

    sorted_candidates = sorted(candidates, key=lambda c: c["y_center"])
    lines: list[dict] = []
    for cand in sorted_candidates:
        placed = False
        for line in lines:
            if abs(cand["y_center"] - line["y_center"]) <= y_tol:
                line["items"].append(cand)
                line["y_center"] = (line["y_center"] + cand["y_center"]) / 2.0
                placed = True
                break
        if not placed:
            lines.append({"y_center": cand["y_center"], "items": [cand]})

    normalized_lines: list[dict] = []
    for line in lines:
        items = sorted(line["items"], key=lambda c: c["x_center"])
        line_text = _normalize_space(" ".join(item["text"] for item in items))
        line_conf = max((item["conf"] for item in items), default=0.0)
        normalized_lines.append({"text": line_text, "conf": line_conf, "items": items})

    return normalized_lines


def _extract_serial_candidates(text: str) -> list[str]:
    """Extract serial-like digit strings (10-16 digits) from text."""
    matches = _SERIAL_DIGITS_RE.findall(text)
    serials: list[str] = []
    for match in matches:
        value = _digits_only(match)
        if 10 <= len(value) <= 16:
            serials.append(value)
    return serials


def _extract_part_candidates(text: str) -> list[str]:
    """Extract part-number-like numeric groups separated by dash/space."""
    matches = _PART_PATTERN_RE.findall(text)
    return [_format_part_number(match) for match in matches]


def classify_fields(candidates: list[dict]) -> dict:
    """Classify OCR candidates into UPC / Serial Number / Part Number fields."""
    result = {
        "roi_a": {"label": "UPC", "text": "", "conf": 0.0},
        "roi_b": {"label": "S/N", "text": "", "conf": 0.0},
        "roi_c": {"label": "BATCH", "text": "", "conf": 0.0},
    }

    lines = _group_lines(candidates)

    # 1) UPC: exact literal "UPC", highest confidence.
    upc_hits = [c for c in candidates if _UPC_EXACT_RE.fullmatch(c["text"])]
    if upc_hits:
        best = max(upc_hits, key=lambda c: c["conf"])
        result["roi_a"]["text"] = "UPC"
        result["roi_a"]["conf"] = round(float(best["conf"]), 3)

    # 2) Serial Number:
    # Prefer lines containing "Serial", then fallback to pure digit candidates.
    serial_options: list[tuple[str, float]] = []
    preferred_lines = [line for line in lines if _SERIAL_WORD_RE.search(line["text"])]
    for line in preferred_lines:
        for serial in _extract_serial_candidates(line["text"]):
            serial_options.append((serial, line["conf"]))

    if not serial_options:
        for cand in candidates:
            text = cand["text"]
            if re.fullmatch(r"\d{10,16}", text):
                serial_options.append((text, cand["conf"]))

    if serial_options:
        best_text, best_conf = max(serial_options, key=lambda row: row[1])
        result["roi_b"]["text"] = best_text
        result["roi_b"]["conf"] = round(float(best_conf), 3)

    # 3) Part Number:
    # Prefer lines containing "Part", else fallback to all lines/candidates.
    part_options: list[tuple[str, float]] = []
    preferred_part_lines = [line for line in lines if _PART_WORD_RE.search(line["text"])]
    scan_lines = preferred_part_lines if preferred_part_lines else lines

    for line in scan_lines:
        for part in _extract_part_candidates(line["text"]):
            part_options.append((part, line["conf"]))

    if not part_options:
        for cand in candidates:
            for part in _extract_part_candidates(cand["text"]):
                part_options.append((part, cand["conf"]))

    if part_options:
        best_text, best_conf = max(part_options, key=lambda row: row[1])
        result["roi_c"]["text"] = best_text
        result["roi_c"]["conf"] = round(float(best_conf), 3)

    return result


def run_fullframe_ocr(img: np.ndarray) -> dict:
    """Run EasyOCR on full frame and classify required output fields."""
    reader = _ensure_reader()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        read_results = reader.readtext(img, detail=1)

    candidates = _to_candidates(read_results)
    return classify_fields(candidates)
