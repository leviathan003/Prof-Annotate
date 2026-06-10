"""
profannotate/utils/image.py
Image loading using Pillow only. cv2 is never imported here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _pil_image_module():
    """Import PIL.Image robustly whether frozen or not."""
    import importlib

    try:
        mod = importlib.import_module("PIL.Image")
        return mod
    except ImportError:
        pass
    try:
        import os
        import sys

        # When frozen, PIL submodules may need explicit path injection
        if getattr(sys, "frozen", False):
            pil_path = os.path.join(sys._MEIPASS, "PIL")
            if pil_path not in sys.path:
                sys.path.insert(0, sys._MEIPASS)
        mod = importlib.import_module("PIL.Image")
        return mod
    except ImportError as exc:
        raise ImportError(f"PIL.Image unavailable: {exc}") from exc


def is_image_corrupted(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return True
    try:
        PilImage = _pil_image_module()
        with PilImage.open(path) as img:
            img.verify()
        return False
    except Exception:
        return True


def load_image_rgb(path: str | Path) -> Optional[np.ndarray]:
    try:
        PilImage = _pil_image_module()
        with PilImage.open(path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        logger.error("load_image_rgb failed %s: %s", path, exc)
        return None


def image_dimensions(path: str | Path) -> Optional[tuple[int, int]]:
    try:
        PilImage = _pil_image_module()
        with PilImage.open(path) as img:
            return img.size  # (w, h)
    except Exception:
        return None


def numpy_to_qpixmap(arr: np.ndarray):
    """RGB (H,W,3) uint8 → QPixmap. Call from UI thread only."""
    from PySide6.QtGui import QImage, QPixmap

    h, w, ch = arr.shape
    qimg = QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def derive_label_path(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    parts = list(image_path.parts)
    try:
        idx = [p.lower() for p in parts].index("images")
        parts[idx] = "labels"
    except ValueError:
        pass
    return Path(*parts).with_suffix(".txt")
