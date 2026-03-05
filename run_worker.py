"""Start persistent OCR worker API.

Usage:
    python run_worker.py
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("OCR_WORKER_HOST", "127.0.0.1")
    port = int(os.getenv("OCR_WORKER_PORT", "8088"))
    uvicorn.run("ocr_worker_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
