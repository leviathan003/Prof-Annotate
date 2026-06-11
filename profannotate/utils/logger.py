"""
profannotate/utils/logger.py
Centralised logging setup.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from profannotate.config.constants import APP_CACHE_DIR


def _is_packaged() -> bool:
    """True when running as a frozen/packaged build (Nuitka onefile, AppImage).

    In packaged builds we never write to the console — all user-facing
    information is surfaced through the UI instead. Logs still go to the
    on-disk log file for support/debugging.
    """
    if "__compiled__" in globals():  # set in every Nuitka-compiled module
        return True
    if getattr(sys, "frozen", False):
        return True
    # AppImage runtime exports these into the environment.
    return bool(os.environ.get("APPIMAGE") or os.environ.get("APPDIR"))


def setup_logging(level: int = logging.INFO) -> None:
    APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = APP_CACHE_DIR / "profannotate.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    # File log always; console log only in unpackaged (dev) runs so AppImage
    # builds stay silent on stdout/stderr.
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    if not _is_packaged():
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    # Quiet noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)
