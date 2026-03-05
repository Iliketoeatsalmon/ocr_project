"""Persistent OCR worker API.

Run:
    uvicorn ocr_worker_api:app --host 127.0.0.1 --port 8088
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.ocr_models import OCR_BACKEND, OCR_GPU_ENABLED
from core.pipeline import run_full_pipeline


class OCRRequest(BaseModel):
    """Request contract for OCR endpoint."""

    image_path: str


app = FastAPI(title="Jetson OCR Worker", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """Worker health and backend status."""
    return {
        "status": "ok",
        "backend": OCR_BACKEND,
        "gpu_enabled": OCR_GPU_ENABLED,
        "speed_mode": os.getenv("OCR_SPEED_MODE", "balanced"),
    }


@app.post("/ocr")
def ocr(payload: OCRRequest) -> JSONResponse:
    """Run OCR pipeline on image path and return JSON payload."""
    try:
        result = run_full_pipeline(payload.image_path)
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:  # pylint: disable=broad-except
        return JSONResponse(
            status_code=500,
            content={"error": "Pipeline failed.", "detail": str(exc)},
        )
