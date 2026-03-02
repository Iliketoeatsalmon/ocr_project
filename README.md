# Jetson OCR Backend Engine (Python)

Backend OCR engine intended to be called by a C# GUI application on Windows.
The C# app passes an image path to `ocr_engine.py`, and Python returns one JSON
object to stdout containing OCR text, confidence, timing, and lighting status.

Default extraction strategy is **full-frame OCR** (EasyOCR) + text pattern
classification:
- `roi_a` <= Serial Number (S/N)
- `roi_b` <= UPC/top-line numeric code
- `roi_c` <= Part Number / Batch

Fixed-ROI flow is still available as legacy mode.

## High-Level Architecture

C# GUI -> Python CLI (`ocr_engine.py`) -> Core pipeline modules (`core/`) -> JSON stdout

## Project Layout

- `ocr_engine.py`: main entry point called by C#.
- `config/roi_config.json`: fixed ROI definitions for legacy ROI mode.
- `core/`: OCR pipeline modules (ROI, preprocessing, OCR abstraction, metrics).
- `samples/raw/`: raw camera captures for local testing.
- `samples/debug/`: intermediate artifacts (ROI crops, thresholded outputs).
- `notebooks/experiments.ipynb`: manual experimentation for tuning.

## Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
```

### 2) Activate it

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Pipeline Modes

- Default: `fullframe` (no manual ROI tuning required)
- Legacy: `legacy_roi` (uses `config/roi_config.json`)

To run legacy mode:

```bash
OCR_PIPELINE_MODE=legacy_roi python ocr_engine.py samples/raw/example.jpg
```

## Run a Test

```bash
python ocr_engine.py samples/raw/example.jpg
```

## Expected JSON Output Shape

```json
{
  "time_ms": 42,
  "roi_a": {"label": "S/N", "text": "89921-X", "conf": 0.998},
  "roi_b": {"label": "MODEL", "text": "J-NANO", "conf": 0.985},
  "roi_c": {"label": "BATCH", "text": "22A", "conf": 0.87},
  "lighting_ok": true
}
```

On error, the CLI prints JSON with an `error` field and exits non-zero.

## ROI Calibration Note

The values in `config/roi_config.json` are placeholders. You must calibrate `x`,
`y`, `w`, and `h` to your real Basler camera framing and label placement.
