"""CLI entry point for the Jetson OCR backend engine.

Usage:
    python ocr_engine.py <image_path>

This script is the only Python executable intended to be called from the C# GUI.
It validates inputs, runs the OCR pipeline, and prints exactly one JSON object
to stdout for machine parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run OCR pipeline on a single image.")
    parser.add_argument("image_path", help="Path to the input image file.")
    return parser


def main() -> int:
    """Run CLI flow and return process exit code."""
    parser = _build_parser()
    args = parser.parse_args()
    image_path = os.path.abspath(args.image_path)

    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image file does not exist: {image_path}"}))
        return 2

    if not os.path.isfile(image_path):
        print(json.dumps({"error": f"Path is not a file: {image_path}"}))
        return 2

    try:
        from core.pipeline import run_full_pipeline

        result = run_full_pipeline(image_path)
        print(json.dumps(result))
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(
            json.dumps(
                {
                    "error": "Pipeline failed.",
                    "detail": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
