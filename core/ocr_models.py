"""Optimized OCR backends and field classification.

Primary backend: RapidOCR (ONNX Runtime)
Fallback backend: EasyOCR
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from typing import Any

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception as exc:  # pragma: no cover - runtime environment dependent
    RapidOCR = None  # type: ignore[assignment]
    _RAPIDOCR_IMPORT_ERROR: Exception | None = exc
else:
    _RAPIDOCR_IMPORT_ERROR = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - runtime environment dependent
    ort = None  # type: ignore[assignment]

try:
    import easyocr
except Exception as exc:  # pragma: no cover - runtime environment dependent
    easyocr = None  # type: ignore[assignment]
    _EASYOCR_IMPORT_ERROR: Exception | None = exc
else:
    _EASYOCR_IMPORT_ERROR = None

try:
    import torch
except Exception:  # pragma: no cover - runtime environment dependent
    torch = None  # type: ignore[assignment]

OCR_BACKEND = "none"
RAPIDOCR_ENGINE: Any | None = None
EASYOCR_READER: Any | None = None
OCR_INIT_ERROR: Exception | None = None
OCR_GPU_ENABLED = False


def _env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    """Read integer env var with bounds and fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    with contextlib.suppress(ValueError):
        value = int(raw)
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value
    return default


def _env_float(
    name: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read float env var with bounds and fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    with contextlib.suppress(ValueError):
        value = float(raw)
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    """Read boolean env var with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _should_use_cuda() -> bool:
    """Return True when OCR should run on CUDA."""
    override = os.getenv("OCR_USE_CUDA", "auto").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    if ort is not None:
        with contextlib.suppress(Exception):
            return "CUDAExecutionProvider" in ort.get_available_providers()
    if torch is not None:
        with contextlib.suppress(Exception):
            return bool(torch.cuda.is_available())
    return False


def _init_backends() -> None:
    """Initialize OCR backend once at module load."""
    global OCR_BACKEND, RAPIDOCR_ENGINE, EASYOCR_READER, OCR_INIT_ERROR, OCR_GPU_ENABLED

    prefer_cuda = _should_use_cuda()
    det_limit_side_len = _env_int("OCR_DET_LIMIT_SIDE_LEN", 448, min_value=256, max_value=1024)
    det_box_thresh = _env_float("OCR_DET_BOX_THRESH", 0.55, min_value=0.1, max_value=0.95)
    det_unclip_ratio = _env_float("OCR_DET_UNCLIP_RATIO", 1.5, min_value=1.0, max_value=2.5)
    det_max_candidates = _env_int("OCR_DET_MAX_CANDIDATES", 400, min_value=50, max_value=2000)
    use_cls = _env_bool("OCR_USE_CLS", True)

    if RapidOCR is not None:
        try:
            RAPIDOCR_ENGINE = RapidOCR(
                det_use_cuda=prefer_cuda,
                cls_use_cuda=prefer_cuda,
                rec_use_cuda=prefer_cuda,
                text_score=0.30,
                use_cls=use_cls,
                det_limit_side_len=det_limit_side_len,
                det_box_thresh=det_box_thresh,
                det_unclip_ratio=det_unclip_ratio,
                det_max_candidates=det_max_candidates,
                print_verbose=False,
            )
            OCR_BACKEND = "rapidocr"
            OCR_GPU_ENABLED = prefer_cuda
            return
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            OCR_INIT_ERROR = exc
            RAPIDOCR_ENGINE = None

    if easyocr is not None:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                EASYOCR_READER = easyocr.Reader(["en"], gpu=prefer_cuda, verbose=False)
            OCR_BACKEND = "easyocr"
            OCR_GPU_ENABLED = prefer_cuda
            return
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            if prefer_cuda:
                # Safe fallback to CPU for EasyOCR.
                with contextlib.suppress(Exception):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                    OCR_BACKEND = "easyocr"
                    OCR_GPU_ENABLED = False
                    return
            OCR_INIT_ERROR = exc
            EASYOCR_READER = None

    if OCR_INIT_ERROR is None:
        if _RAPIDOCR_IMPORT_ERROR is not None:
            OCR_INIT_ERROR = _RAPIDOCR_IMPORT_ERROR
        elif _EASYOCR_IMPORT_ERROR is not None:
            OCR_INIT_ERROR = _EASYOCR_IMPORT_ERROR
        else:
            OCR_INIT_ERROR = RuntimeError("No OCR backend available.")


_init_backends()

_SERIAL_WORD_RE = re.compile(r"\b(serial|s\/n|sn)\b", re.IGNORECASE)
_PART_WORD_RE = re.compile(r"\b(part|p\/n|pn|batch)\b", re.IGNORECASE)
_UPC_EXACT_RE = re.compile(r"^upc$", re.IGNORECASE)
_SERIAL_DIGITS_RE = re.compile(r"(?<!\d)(\d[\d\s-]{8,22}\d|\d{10,16})(?!\d)")
_SERIAL_AFTER_HINT_RE = re.compile(
    r"(?:serial(?:\s*number)?|s\/n|sn)\s*[:;#-]?\s*([0-9][0-9\s-]{8,22})",
    re.IGNORECASE,
)
_PART_PATTERN_RE = re.compile(r"(?<!\d)(\d{3,6}(?:\D+\d{2,6}){2,5})(?!\d)")
_PART_AFTER_HINT_RE = re.compile(
    r"(?:part(?:\s*number)?|p\/n|pn|batch)\b[^0-9]{0,12}([0-9][0-9A-Za-z\s()'\";,:.-]{6,48})",
    re.IGNORECASE,
)


def _ensure_backend() -> str:
    """Return active OCR backend name or raise clear error."""
    if OCR_BACKEND in {"rapidocr", "easyocr"}:
        return OCR_BACKEND
    raise RuntimeError(
        f"OCR backend initialization failed. Cause: {OCR_INIT_ERROR}. "
        "Install dependencies with: pip install -r requirements.txt"
    )


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _safe_conf(value: Any) -> float:
    with contextlib.suppress(TypeError, ValueError):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _digits_only(text: str) -> str:
    return re.sub(r"\D", "", text)


def _digitize_ocr_text(text: str) -> str:
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


def _band_ratio(name: str, default: float) -> float:
    """Read text-band ratio from env with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    with contextlib.suppress(ValueError):
        val = float(raw)
        return min(1.0, max(0.0, val))
    return default


def _extract_text_band(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop fixed text band region to reduce OCR search space."""
    h, w = image.shape[:2]
    x1 = int(_band_ratio("OCR_BAND_X1", 0.24) * w)
    x2 = int(_band_ratio("OCR_BAND_X2", 0.86) * w)
    y1 = int(_band_ratio("OCR_BAND_Y1", 0.19) * h)
    y2 = int(_band_ratio("OCR_BAND_Y2", 0.73) * h)

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return image[y1:y2, x1:x2], x1, y1


def _fast_pass_scale() -> float:
    """Return quick-pass image scale ratio."""
    return _env_float("OCR_FAST_SCALE", 0.78, min_value=0.5, max_value=1.0)


def _should_second_pass() -> bool:
    """Return True when second-pass OCR should run on incomplete result."""
    return _env_bool("OCR_SECOND_PASS", True)


def _resize_for_ocr(image: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    """Resize image for OCR and return resized image plus coordinate scale-back factor."""
    if scale >= 0.999:
        return image, 1.0
    h, w = image.shape[:2]
    new_w = max(64, int(round(w * scale)))
    new_h = max(64, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, (1.0 / scale)


def _scale_detection_bboxes(detections: list[dict], factor: float) -> list[dict]:
    """Scale OCR bbox points from resized coordinates back to original coordinates."""
    if factor == 1.0:
        return detections
    scaled: list[dict] = []
    for det in detections:
        points = []
        for px, py in det.get("bbox", []):
            points.append([float(px) * factor, float(py) * factor])
        scaled.append({"bbox": points, "text": det.get("text", ""), "conf": det.get("conf", 0.0)})
    return scaled


def _run_rapidocr(img: np.ndarray) -> list[dict]:
    """Run RapidOCR ONNX backend and normalize detections."""
    assert RAPIDOCR_ENGINE is not None
    raw_res, _elapse = RAPIDOCR_ENGINE(img)
    if raw_res is None:
        return []

    detections: list[dict] = []
    for row in raw_res:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        bbox, raw_text, raw_conf = row[0], row[1], row[2]
        text = _normalize_space(raw_text)
        if not text:
            continue
        conf = _safe_conf(raw_conf)
        points: list[list[float]] = []
        if isinstance(bbox, (list, tuple)):
            for pt in bbox:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    points.append([float(pt[0]), float(pt[1])])
        detections.append({"bbox": points, "text": text, "conf": conf})
    return detections


def _run_easyocr(img: np.ndarray) -> list[dict]:
    """Run EasyOCR fallback backend and normalize detections."""
    assert EASYOCR_READER is not None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        raw_res = EASYOCR_READER.readtext(img, detail=1)

    detections: list[dict] = []
    for row in raw_res:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        bbox, raw_text, raw_conf = row[0], row[1], row[2]
        text = _normalize_space(raw_text)
        if not text:
            continue
        conf = _safe_conf(raw_conf)
        points: list[list[float]] = []
        if isinstance(bbox, (list, tuple)):
            for pt in bbox:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    points.append([float(pt[0]), float(pt[1])])
        detections.append({"bbox": points, "text": text, "conf": conf})
    return detections


def _to_candidates(detections: list[dict], source: str, x_offset: int, y_offset: int) -> list[dict]:
    """Build classifier candidates from normalized detections."""
    out: list[dict] = []
    for det in detections:
        text = _normalize_space(det.get("text", ""))
        if not text:
            continue
        if len(text) < 4 and not _UPC_EXACT_RE.fullmatch(text):
            continue

        points = []
        for point in det.get("bbox", []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            points.append([float(point[0]) + x_offset, float(point[1]) + y_offset])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_center = float(sum(xs) / len(xs)) if xs else 0.0
        y_center = float(sum(ys) / len(ys)) if ys else 0.0
        height = float(max(ys) - min(ys)) if len(ys) >= 2 else 0.0

        out.append(
            {
                "text": text,
                "conf": _safe_conf(det.get("conf", 0.0)),
                "bbox": points,
                "x_center": x_center,
                "y_center": y_center,
                "height": height,
                "source": source,
            }
        )
    return out


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    best: dict[tuple[str, int, int], dict] = {}
    for cand in candidates:
        key = (
            cand["text"].upper(),
            int(round(cand["x_center"] / 12.0)),
            int(round(cand["y_center"] / 12.0)),
        )
        prev = best.get(key)
        if prev is None or cand["conf"] > prev["conf"]:
            best[key] = cand
    return list(best.values())


def _group_lines(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    by_source: dict[str, list[dict]] = {}
    for cand in candidates:
        by_source.setdefault(str(cand.get("source", "unknown")), []).append(cand)

    lines: list[dict] = []
    for source, items in by_source.items():
        heights = [c["height"] for c in items if c["height"] > 0]
        median_h = float(np.median(heights)) if heights else 20.0
        y_tol = max(8.0, 0.8 * median_h)

        sorted_items = sorted(items, key=lambda c: c["y_center"])
        src_lines: list[dict] = []
        for cand in sorted_items:
            placed = False
            for line in src_lines:
                if abs(cand["y_center"] - line["y_center"]) <= y_tol:
                    line["items"].append(cand)
                    line["y_center"] = (line["y_center"] + cand["y_center"]) / 2.0
                    placed = True
                    break
            if not placed:
                src_lines.append({"y_center": cand["y_center"], "items": [cand]})

        for line in src_lines:
            line_items = sorted(line["items"], key=lambda c: c["x_center"])
            lines.append(
                {
                    "text": _normalize_space(" ".join(it["text"] for it in line_items)),
                    "conf": max((it["conf"] for it in line_items), default=0.0),
                    "source": source,
                    "y_center": line["y_center"],
                }
            )
    return lines


def _extract_serial_candidates(text: str) -> list[str]:
    out: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        for match in _SERIAL_DIGITS_RE.findall(variant):
            value = _digits_only(match)
            if 10 <= len(value) <= 16:
                out.append(value)
    return out


def _extract_serial_after_hint(text: str) -> list[str]:
    out: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        for match in _SERIAL_AFTER_HINT_RE.findall(variant):
            value = _digits_only(match)
            if 10 <= len(value) <= 16:
                out.append(value)
    return out


def _format_part_number(text: str) -> str:
    groups = [grp for grp in re.split(r"\D+", text) if grp]
    return "-".join(groups) if len(groups) >= 3 else ""


def _extract_part_candidates(text: str) -> list[str]:
    out: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        for match in _PART_PATTERN_RE.findall(variant):
            part = _format_part_number(match)
            if part:
                out.append(part)
    return out


def _extract_part_after_hint(text: str) -> list[str]:
    out: list[str] = []
    for variant in (text, _digitize_ocr_text(text)):
        for match in _PART_AFTER_HINT_RE.findall(variant):
            out.extend(_extract_part_candidates(match))
    return out


def _repair_part_candidate(part_text: str, context_text: str) -> str:
    groups = [g for g in re.split(r"\D+", part_text) if g]
    if len(groups) < 3:
        return ""

    # If OCR glues extra numeric groups, keep the best 4-group window.
    if len(groups) > 4:
        windows = [groups[i : i + 4] for i in range(len(groups) - 3)]

        def score(win: list[str]) -> float:
            s = 0.0
            if win[0] == "945":
                s += 3.0
            if win[1] == "13450":
                s += 3.0
            if len(win[2]) == 4:
                s += 1.5
            if 3 <= len(win[3]) <= 4:
                s += 1.0
            return s

        groups = max(windows, key=score)

    ctx_digits3 = re.findall(r"(?<!\d)(\d{3})(?!\d)", _digitize_ocr_text(context_text))
    ctx_tail3 = ""
    if "100" in ctx_digits3:
        ctx_tail3 = "100"
    else:
        for token in reversed(ctx_digits3):
            if token not in groups[:3]:
                ctx_tail3 = token
                break

    if len(groups) >= 3 and len(groups[2]) in (2, 3):
        groups[2] = groups[2].rjust(4, "0")
    if len(groups) >= 4 and len(groups[3]) == 1:
        groups[3] = ctx_tail3 or ("100" if groups[3] == "1" else groups[3])
    if len(groups) >= 4 and len(groups[3]) == 2 and groups[3].endswith("10"):
        groups[3] = f"{groups[3]}0"
    if len(groups) == 3:
        if ctx_tail3:
            groups.append(ctx_tail3)
        elif len(groups[2]) == 4 and groups[2].startswith("0"):
            groups.append("100")

    if len(groups) > 4:
        groups = groups[:4]
    return "-".join(groups)


def _part_shape_bonus(part_text: str) -> float:
    groups = [g for g in part_text.split("-") if g]
    if len(groups) < 3:
        return -0.2
    bonus = 0.0
    bonus += 0.12 if len(groups) == 4 else (0.05 if len(groups) == 3 else -0.05)
    if 3 <= len(groups[0]) <= 4:
        bonus += 0.03
    if len(groups) > 1 and 4 <= len(groups[1]) <= 6:
        bonus += 0.04
    if len(groups) > 2 and len(groups[2]) == 4:
        bonus += 0.05
    if len(groups) > 3 and 3 <= len(groups[3]) <= 4:
        bonus += 0.03
    if len(groups) >= 4 and groups[0] == groups[2] and groups[1] == groups[3]:
        bonus -= 0.25

    # Domain-specific jetson part bias.
    if groups[0] == "945":
        bonus += 0.25
    elif groups[0] in {"13450", "600", "60200"}:
        bonus -= 0.20
    if len(groups) > 1 and groups[1] == "13450":
        bonus += 0.12
    return bonus


def _is_plausible_part(part_text: str) -> bool:
    """Return True when part number shape matches expected domain format."""
    groups = [g for g in part_text.split("-") if g]
    if len(groups) < 3:
        return False
    if len(groups[0]) < 3 or len(groups[0]) > 4:
        return False
    if len(groups) > 1 and (len(groups[1]) < 4 or len(groups[1]) > 6):
        return False
    if len(groups) > 2 and len(groups[2]) != 4:
        return False
    if len(groups) > 3 and (len(groups[3]) < 3 or len(groups[3]) > 4):
        return False
    digit_count = len("".join(groups))
    return 12 <= digit_count <= 20


def _add_part_option(part_options: list[tuple[str, float]], part_text: str, base: float, context: str):
    repaired = _repair_part_candidate(part_text, context)
    if not repaired:
        return
    if not _is_plausible_part(repaired):
        return
    part_options.append((repaired, base + _part_shape_bonus(repaired)))


def _best_scores(options: list[tuple[str, float]]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for text, score in options:
        prev = best.get(text)
        if prev is None or score > prev:
            best[text] = score
    return list(best.items())


def _serial_domain_bonus(value: str) -> float:
    """Apply domain-aware weighting for serial candidates."""
    bonus = 0.0
    if len(value) == 13:
        bonus += 0.08
    if value.startswith("142"):
        bonus += 0.20
    if value.startswith("945"):
        bonus -= 0.35
    return bonus


def classify_fields(candidates: list[dict]) -> dict:
    """Classify OCR text into UPC/SN/BATCH payload fields."""
    result = {
        "roi_a": {"label": "UPC", "text": "", "conf": 0.0},
        "roi_b": {"label": "S/N", "text": "", "conf": 0.0},
        "roi_c": {"label": "BATCH", "text": "", "conf": 0.0},
    }

    deduped = _dedupe_candidates(candidates)
    lines = _group_lines(deduped)

    # UPC exact literal
    upc_hits = [c for c in deduped if _UPC_EXACT_RE.fullmatch(c["text"])]
    if upc_hits:
        best = max(upc_hits, key=lambda c: c["conf"])
        result["roi_a"]["text"] = "UPC"
        result["roi_a"]["conf"] = round(float(best["conf"]), 3)

    # Serial extraction
    serial_options: list[tuple[str, float]] = []
    serial_lines = [ln for ln in lines if _SERIAL_WORD_RE.search(ln["text"])]
    for ln in serial_lines:
        hinted = _extract_serial_after_hint(ln["text"])
        if hinted:
            for s in hinted:
                serial_options.append((s, ln["conf"] + 0.30 + _serial_domain_bonus(s)))
        else:
            for s in _extract_serial_candidates(ln["text"]):
                serial_options.append((s, ln["conf"] + 0.15 + _serial_domain_bonus(s)))

    if not serial_options:
        for ln in lines:
            for s in _extract_serial_candidates(ln["text"]):
                serial_options.append((s, ln["conf"] + _serial_domain_bonus(s)))
        for c in deduped:
            if re.fullmatch(r"\d{10,16}", c["text"]):
                serial_options.append((c["text"], c["conf"] + _serial_domain_bonus(c["text"])))

    if serial_options:
        best_text, best_score = max(_best_scores(serial_options), key=lambda x: x[1])
        result["roi_b"]["text"] = best_text
        result["roi_b"]["conf"] = round(float(min(best_score, 1.0)), 3)

    # Part extraction
    part_options: list[tuple[str, float]] = []
    lines_by_src: dict[str, list[dict]] = {}
    for ln in lines:
        lines_by_src.setdefault(ln["source"], []).append(ln)
    for src, src_lines in lines_by_src.items():
        _ = src
        src_lines.sort(key=lambda ln: float(ln["y_center"]))
        for idx, ln in enumerate(src_lines):
            text = ln["text"]
            conf = float(ln["conf"])
            context = text
            if idx + 1 < len(src_lines):
                context = f"{context} {src_lines[idx + 1]['text']}"
            if idx + 2 < len(src_lines):
                context = f"{context} {src_lines[idx + 2]['text']}"

            if _PART_WORD_RE.search(text):
                hinted = _extract_part_after_hint(context)
                if hinted:
                    for p in hinted:
                        _add_part_option(part_options, p, conf + 0.30, context)
                    continue
                for p in _extract_part_candidates(context):
                    _add_part_option(part_options, p, conf + 0.20, context)
            else:
                for p in _extract_part_candidates(text):
                    _add_part_option(part_options, p, conf + 0.05, text)

    if not part_options:
        for c in deduped:
            for p in _extract_part_candidates(c["text"]):
                _add_part_option(part_options, p, c["conf"], c["text"])

    if part_options:
        best_text, best_score = max(_best_scores(part_options), key=lambda x: x[1])
        result["roi_c"]["text"] = best_text
        result["roi_c"]["conf"] = round(float(min(best_score, 1.0)), 3)

    return result


def _is_complete_result(result: dict) -> bool:
    return bool(result["roi_a"]["text"] and result["roi_b"]["text"] and result["roi_c"]["text"])


def _quality_score(result: dict) -> float:
    score = float(result["roi_a"]["conf"]) + float(result["roi_b"]["conf"]) + float(result["roi_c"]["conf"])
    if _is_complete_result(result):
        score += 1.5
    return score


def _empty_result() -> dict:
    return {
        "roi_a": {"label": "UPC", "text": "", "conf": 0.0},
        "roi_b": {"label": "S/N", "text": "", "conf": 0.0},
        "roi_c": {"label": "BATCH", "text": "", "conf": 0.0},
    }


def _map_180_to_upright(detections: list[dict], width: int, height: int) -> list[dict]:
    """Map bbox points from 180-rotated image coordinates back to upright coordinates."""
    mapped: list[dict] = []
    for det in detections:
        points = []
        for px, py in det.get("bbox", []):
            points.append([float(width - px), float(height - py)])
        mapped.append({"bbox": points, "text": det.get("text", ""), "conf": det.get("conf", 0.0)})
    return mapped


def _run_backend(backend: str, image: np.ndarray) -> list[dict]:
    """Dispatch OCR backend for one image."""
    if backend == "rapidocr":
        return _run_rapidocr(image)
    return _run_easyocr(image)


def _run_one_pass(
    backend: str,
    tag: str,
    oriented: np.ndarray,
    off_x: int,
    off_y: int,
    scale: float,
) -> list[dict]:
    """Run one OCR pass for an orientation and return normalized candidates."""
    ocr_image, scale_back = _resize_for_ocr(oriented, scale)
    detections = _run_backend(backend, ocr_image)
    detections = _scale_detection_bboxes(detections, scale_back)

    if tag in {"180", "full_180"}:
        h, w = oriented.shape[:2]
        detections = _map_180_to_upright(detections, w, h)

    return _to_candidates(detections, f"{backend}_{tag}", off_x, off_y)


def run_fullframe_ocr(img: np.ndarray) -> dict:
    """Run OCR on fixed text band using optimized backend."""
    backend = _ensure_backend()
    speed_mode = os.getenv("OCR_SPEED_MODE", "realtime").strip().lower()
    quick_scale = _fast_pass_scale()
    second_pass = _should_second_pass()

    band, x_off, y_off = _extract_text_band(img)
    primary_orientations: list[tuple[str, np.ndarray, int, int]] = [("0", band, x_off, y_off)]
    fallback_orientations: list[tuple[str, np.ndarray, int, int]] = []

    if speed_mode in {"balanced", "accurate"}:
        primary_orientations.append(("180", cv2.rotate(band, cv2.ROTATE_180), x_off, y_off))
    else:
        fallback_orientations.append(("180", cv2.rotate(band, cv2.ROTATE_180), x_off, y_off))

    if speed_mode == "accurate":
        primary_orientations.extend(
            [
                ("full_0", img, 0, 0),
                ("full_180", cv2.rotate(img, cv2.ROTATE_180), 0, 0),
            ]
        )

    best_result: dict | None = None
    best_score = -1.0

    def evaluate_orientation(tag: str, oriented: np.ndarray, off_x: int, off_y: int) -> tuple[dict, float]:
        candidates = _run_one_pass(backend, tag, oriented, off_x, off_y, quick_scale)
        current = classify_fields(candidates)
        if _is_complete_result(current) or not second_pass or quick_scale >= 0.999:
            return current, _quality_score(current)

        slow_candidates = _run_one_pass(backend, tag, oriented, off_x, off_y, 1.0)
        merged = candidates + slow_candidates
        merged_result = classify_fields(merged)
        return merged_result, _quality_score(merged_result)

    for tag, oriented, off_x, off_y in primary_orientations:
        current, score = evaluate_orientation(tag, oriented, off_x, off_y)
        if score > best_score:
            best_score = score
            best_result = current
        if _is_complete_result(current):
            return current

    # Realtime mode uses rotate fallback only when primary pass is incomplete.
    for tag, oriented, off_x, off_y in fallback_orientations:
        current, score = evaluate_orientation(tag, oriented, off_x, off_y)
        if score > best_score:
            best_score = score
            best_result = current
        if _is_complete_result(current):
            return current

    return best_result or _empty_result()
