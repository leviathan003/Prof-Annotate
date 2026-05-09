"""
bytemark/utils/logger.py
Centralised logging setup.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bytemark.config.constants import APP_CACHE_DIR


def setup_logging(level: int = logging.INFO) -> None:
    APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = APP_CACHE_DIR / "bytemark.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    # Quiet noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)
