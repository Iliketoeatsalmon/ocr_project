# Jetson OCR Backend (Persistent Worker + ONNX)

This project provides a backend OCR service for a Windows C# GUI.

## What Changed

- OCR engine is now optimized for realtime with **RapidOCR (ONNX Runtime)**.
- Search space is reduced to a fixed **text band** (instead of full-frame multi-pass).
- Added a **persistent Python worker API** to avoid spawning a new process for each image.
- `easyocr` is still available as a fallback backend.

## Output Schema

The API and CLI return this schema:

```json
{
  "time_ms": 42,
  "roi_a": {"label": "UPC", "text": "UPC", "conf": 0.99},
  "roi_b": {"label": "S/N", "text": "1425022007009", "conf": 0.99},
  "roi_c": {"label": "BATCH", "text": "945-13450-0000-100", "conf": 0.99},
  "lighting_ok": true
}
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Start Persistent Worker (Recommended)

```bash
python run_worker.py
```

Default worker address:

- `http://127.0.0.1:8088`

Endpoints:

- `GET /health`
- `POST /ocr` with JSON body:

```json
{"image_path": "CV/test.bmp"}
```

## CLI Usage (Still Supported)

```bash
python ocr_engine.py CV/test.bmp
```

Optional: let CLI call a running worker instead of local inference:

```bash
OCR_WORKER_URL=http://127.0.0.1:8088 python ocr_engine.py CV/test.bmp
```

## Performance/Accuracy Modes

`OCR_SPEED_MODE`:

- `realtime`: fastest
- `balanced` (default): fast path first, fallback when quality is low
- `accurate`: most robust, slowest

Example:

```bash
OCR_SPEED_MODE=realtime python ocr_engine.py CV/test.bmp
```

## CUDA

CUDA is auto-detected (ONNX Runtime provider / Torch availability).

Override:

```bash
OCR_USE_CUDA=1 python run_worker.py   # force GPU
OCR_USE_CUDA=0 python run_worker.py   # force CPU
```

## Fixed Text Band Tuning

To adjust search band (relative ratios):

- `OCR_BAND_X1` (default `0.20`)
- `OCR_BAND_X2` (default `0.90`)
- `OCR_BAND_Y1` (default `0.16`)
- `OCR_BAND_Y2` (default `0.76`)

Example:

```bash
OCR_BAND_X1=0.18 OCR_BAND_X2=0.92 OCR_BAND_Y1=0.14 OCR_BAND_Y2=0.78 python run_worker.py
```
