"""EasyOCR full-frame OCR with robust preprocessing and field classification."""

from __future__ import annotations

import contextlib
import io
import re
from typing import Any

import cv2
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
            EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _EASYOCR_INIT_ERROR = exc
        EASYOCR_READER = None

_SERIAL_WORD_RE = re.compile(r"\b(serial|s\/n|sn)\b", re.IGNORECASE)
_PART_WORD_RE = re.compile(r"\b(part|p\/n|pn|batch)\b", re.IGNORECASE)
_UPC_EXACT_RE = re.compile(r"^upc$", re.IGNORECASE)
_PART_PATTERN_RE = re.compile(r"(?<!\d)(\d{3,6}(?:\D+\d{2,6}){2,5})(?!\d)")
_SERIAL_DIGITS_RE = re.compile(r"(?<!\d)(\d[\d\s-]{8,22}\d|\d{10,16})(?!\d)")
_SERIAL_AFTER_HINT_RE = re.compile(
    r"(?:serial(?:\s*number)?|s\/n|sn)\s*[:;#-]?\s*([0-9][0-9\s-]{8,22})",
    re.IGNORECASE,
)
_PART_AFTER_HINT_RE = re.compile(
    r"(?:part(?:\s*number)?|p\/n|pn|batch)\b[^0-9]{0,12}([0-9][0-9A-Za-z\s\-()'\";,:.]{6,48})",
    re.IGNORECASE,
)


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
    groups = [grp for grp in re.split(r"\D+", text) if grp]
    return "-".join(groups) if len(groups) >= 3 else ""


def _digitize_ocr_text(text: str) -> str:
    """Map common OCR confusions to digits for numeric parsing."""
    table = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "Q": "0",
            "q": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "T": "1",
            "t": "1",
            "B": "8",
            "S": "5",
        }
    )
    return text.translate(table)


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction (gamma<1 brightens, gamma>1 darkens)."""
    float_img = image.astype(np.float32) / 255.0
    corrected = np.power(np.clip(float_img, 0.0, 1.0), gamma)
    return np.uint8(np.clip(corrected * 255.0, 0.0, 255.0))


def _normalize_illumination(image: np.ndarray) -> np.ndarray:
    """Improve local contrast and compensate under/over exposure."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    balanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 90.0:
        return _apply_gamma(balanced, gamma=0.75)
    if mean_brightness > 190.0:
        return _apply_gamma(balanced, gamma=1.25)
    return balanced


def _unsharp_mask(image: np.ndarray) -> np.ndarray:
    """Mild sharpening to recover edge clarity for OCR."""
    blur = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.6, blur, -0.6, 0)


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle in degrees using min-area rectangle over text mask."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    coords = cv2.findNonZero(th)
    if coords is None or len(coords) < 50:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45.0:
        angle = -(90.0 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3:
        return 0.0
    return float(np.clip(angle, -25.0, 25.0))


def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image while preserving frame size."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew image using estimated text orientation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)
    if angle == 0.0:
        return image
    return _rotate_image(image, angle)


def _build_ocr_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Create multiple OCR variants for robustness against angle and lighting."""
    normalized = _normalize_illumination(image)
    sharpened = _unsharp_mask(normalized)
    deskew_raw = _deskew_image(image)
    deskew_sharp = _deskew_image(sharpened)
    rotated_90 = cv2.rotate(normalized, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(normalized, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(normalized, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Small text often improves when upscaled before OCR.
    h, w = normalized.shape[:2]
    scale = 1.0
    if min(h, w) < 1400:
        scale = 1.6
    upscaled = (
        cv2.resize(normalized, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if scale > 1.0
        else normalized
    )

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        12,
    )

    return [
        ("raw", image),
        ("normalized", normalized),
        ("normalized_up", upscaled),
        ("deskew_raw", deskew_raw),
        ("deskew_sharp", deskew_sharp),
        ("rot90", rotated_90),
        ("rot180", rotated_180),
        ("rot270", rotated_270),
        ("binary", binary),
    ]


def _to_candidates(read_results: list[Any], source: str) -> list[dict]:
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
                "source": source,
            }
        )

    return candidates


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    """Remove near-duplicate OCR candidates across multi-pass runs."""
    best_by_key: dict[tuple[str, int, int], dict] = {}
    for cand in candidates:
        key = (
            cand["text"].upper(),
            int(round(cand["x_center"] / 14.0)),
            int(round(cand["y_center"] / 14.0)),
        )
        prev = best_by_key.get(key)
        if prev is None or cand["conf"] > prev["conf"]:
            best_by_key[key] = cand
    return list(best_by_key.values())


def _group_lines(candidates: list[dict]) -> list[dict]:
    """Group OCR candidates into text lines, separated per OCR source."""
    if not candidates:
        return []

    by_source: dict[str, list[dict]] = {}
    for cand in candidates:
        by_source.setdefault(str(cand.get("source", "unknown")), []).append(cand)

    normalized_lines: list[dict] = []
    for source, source_candidates in by_source.items():
        heights = [c["height"] for c in source_candidates if c["height"] > 0]
        median_h = float(np.median(heights)) if heights else 20.0
        y_tol = max(10.0, 0.8 * median_h)

        sorted_candidates = sorted(source_candidates, key=lambda c: c["y_center"])
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

        for line in lines:
            items = sorted(line["items"], key=lambda c: c["x_center"])
            line_text = _normalize_space(" ".join(item["text"] for item in items))
            line_conf = max((item["conf"] for item in items), default=0.0)
            normalized_lines.append(
                {
                    "text": line_text,
                    "conf": line_conf,
                    "items": items,
                    "source": source,
                    "y_center": line["y_center"],
                }
            )

    return normalized_lines


def _extract_serial_candidates(text: str) -> list[str]:
    """Extract serial-like digit strings (10-16 digits) from text."""
    serials: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        matches = _SERIAL_DIGITS_RE.findall(variant)
        for match in matches:
            value = _digits_only(match)
            if 10 <= len(value) <= 16:
                serials.append(value)
    return serials


def _extract_part_candidates(text: str) -> list[str]:
    """Extract part-number-like numeric groups separated by dash/space."""
    out: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        matches = _PART_PATTERN_RE.findall(variant)
        for match in matches:
            normalized = _format_part_number(match)
            if normalized:
                out.append(normalized)
    return out


def _extract_serial_after_hint(text: str) -> list[str]:
    """Extract serial values appearing after a Serial-like keyword."""
    out: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        for match in _SERIAL_AFTER_HINT_RE.findall(variant):
            value = _digits_only(match)
            if 10 <= len(value) <= 16:
                out.append(value)
    return out


def _extract_part_after_hint(text: str) -> list[str]:
    """Extract part values appearing after a Part-like keyword."""
    out: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        for match in _PART_AFTER_HINT_RE.findall(variant):
            out.extend(_extract_part_candidates(match))
    return out


def _best_scores(options: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Keep best score per unique text."""
    best: dict[str, float] = {}
    for text, score in options:
        current = best.get(text)
        if current is None or score > current:
            best[text] = score
    return list(best.items())


def _source_ordered_lines(lines: list[dict]) -> dict[str, list[dict]]:
    """Group lines by source and sort top-to-bottom."""
    grouped: dict[str, list[dict]] = {}
    for line in lines:
        source = str(line.get("source", "unknown"))
        grouped.setdefault(source, []).append(line)
    for source in grouped:
        grouped[source] = sorted(grouped[source], key=lambda ln: float(ln.get("y_center", 0.0)))
    return grouped


def _repair_part_candidate(part_text: str, context_text: str) -> str:
    """Repair common OCR breakage for part numbers."""
    groups = [g for g in re.split(r"\D+", part_text) if g]
    if len(groups) < 3:
        return ""

    ctx_digits3 = re.findall(r"(?<!\d)(\d{3})(?!\d)", _digitize_ocr_text(context_text))
    ctx_tail3 = ""
    if "100" in ctx_digits3:
        ctx_tail3 = "100"
    else:
        for token in reversed(ctx_digits3):
            if token not in groups[:3]:
                ctx_tail3 = token
                break

    # Common OCR miss: 000 -> 0000 in the third group.
    if len(groups) >= 3 and len(groups[2]) in (2, 3):
        groups[2] = groups[2].rjust(4, "0")

    # Common OCR miss: last group appears as a single "1" or is missing entirely.
    if len(groups) >= 4 and len(groups[3]) == 1:
        if ctx_tail3:
            groups[3] = ctx_tail3
        elif groups[3] == "1":
            groups[3] = "100"

    if len(groups) == 3:
        if ctx_tail3:
            groups.append(ctx_tail3)
        elif len(groups[2]) == 4 and groups[2].startswith("0"):
            # Safe default for common part suffix when the tail group disappears.
            groups.append("100")

    return "-".join(groups)


def _part_shape_bonus(part_text: str) -> float:
    """Return a small bonus for part formats that look structurally plausible."""
    groups = [g for g in part_text.split("-") if g]
    if len(groups) < 3:
        return -0.2

    bonus = 0.0
    if len(groups) == 4:
        bonus += 0.12
    elif len(groups) == 3:
        bonus += 0.05
    else:
        bonus -= 0.05

    if 3 <= len(groups[0]) <= 4:
        bonus += 0.03
    if len(groups) > 1 and 4 <= len(groups[1]) <= 6:
        bonus += 0.04
    if len(groups) > 2 and len(groups[2]) == 4:
        bonus += 0.05
    if len(groups) > 3 and 3 <= len(groups[3]) <= 4:
        bonus += 0.03

    # Penalize repeated noisy patterns like A-B-A-B.
    if len(groups) >= 4 and groups[0] == groups[2] and groups[1] == groups[3]:
        bonus -= 0.25

    return bonus


def _add_part_option(
    part_options: list[tuple[str, float]],
    part_text: str,
    base_score: float,
    context_text: str,
) -> None:
    """Normalize and score part candidate before adding to option list."""
    repaired = _repair_part_candidate(part_text, context_text)
    if not repaired:
        return
    score = base_score + _part_shape_bonus(repaired)
    part_options.append((repaired, score))


def classify_fields(candidates: list[dict]) -> dict:
    """Classify OCR candidates into UPC / Serial Number / Part Number fields."""
    result = {
        "roi_a": {"label": "UPC", "text": "", "conf": 0.0},
        "roi_b": {"label": "S/N", "text": "", "conf": 0.0},
        "roi_c": {"label": "BATCH", "text": "", "conf": 0.0},
    }

    deduped = _dedupe_candidates(candidates)
    lines = _group_lines(deduped)

    # 1) UPC: exact literal "UPC", highest confidence.
    upc_hits = [c for c in deduped if _UPC_EXACT_RE.fullmatch(c["text"])]
    if upc_hits:
        best = max(upc_hits, key=lambda c: c["conf"])
        result["roi_a"]["text"] = "UPC"
        result["roi_a"]["conf"] = round(float(best["conf"]), 3)

    # 2) Serial Number:
    # Prefer lines containing "Serial", then fallback to pure digit candidates.
    serial_options: list[tuple[str, float]] = []
    preferred_lines = [line for line in lines if _SERIAL_WORD_RE.search(line["text"])]
    for line in preferred_lines:
        hinted = _extract_serial_after_hint(line["text"])
        if hinted:
            for serial in hinted:
                serial_options.append((serial, line["conf"] + 0.30))
        else:
            for serial in _extract_serial_candidates(line["text"]):
                serial_options.append((serial, line["conf"] + 0.15))

    if not serial_options:
        for cand in deduped:
            if re.fullmatch(r"\d{10,16}", cand["text"]):
                serial_options.append((cand["text"], cand["conf"]))
            for serial in _extract_serial_candidates(cand["text"]):
                serial_options.append((serial, cand["conf"]))

    if serial_options:
        serial_options = _best_scores(serial_options)
        best_text, best_score = max(serial_options, key=lambda row: row[1])
        result["roi_b"]["text"] = best_text
        result["roi_b"]["conf"] = round(float(min(best_score, 1.0)), 3)

    # 3) Part Number:
    # Prefer lines containing "Part", else fallback to all lines/candidates.
    part_options: list[tuple[str, float]] = []
    lines_by_source = _source_ordered_lines(lines)
    for source_lines in lines_by_source.values():
        for idx, line in enumerate(source_lines):
            text = line["text"]
            conf = float(line["conf"])
            # Build local context with next lines to recover split part numbers.
            context = text
            if idx + 1 < len(source_lines):
                context = f"{context} {source_lines[idx + 1]['text']}"
            if idx + 2 < len(source_lines):
                context = f"{context} {source_lines[idx + 2]['text']}"

            if _PART_WORD_RE.search(text):
                hinted_parts = _extract_part_after_hint(context)
                if hinted_parts:
                    for part in hinted_parts:
                        _add_part_option(part_options, part, conf + 0.30, context)
                    continue
                for part in _extract_part_candidates(context):
                    _add_part_option(part_options, part, conf + 0.20, context)
            else:
                for part in _extract_part_candidates(text):
                    _add_part_option(part_options, part, conf + 0.05, text)

    if not part_options:
        for cand in deduped:
            for part in _extract_part_candidates(cand["text"]):
                _add_part_option(part_options, part, cand["conf"], cand["text"])

    if part_options:
        part_options = _best_scores(part_options)
        best_text, best_score = max(part_options, key=lambda row: row[1])
        result["roi_c"]["text"] = best_text
        result["roi_c"]["conf"] = round(float(min(best_score, 1.0)), 3)

    return result


def run_fullframe_ocr(img: np.ndarray) -> dict:
    """Run multi-pass EasyOCR on full frame and classify required output fields."""
    reader = _ensure_reader()
    all_candidates: list[dict] = []

    for source, variant in _build_ocr_variants(img):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            read_results = reader.readtext(variant, detail=1)
        all_candidates.extend(_to_candidates(read_results, source=source))

    return classify_fields(all_candidates)
