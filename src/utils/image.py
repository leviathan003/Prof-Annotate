"""
bytemark/utils/image.py
Image loading, corruption detection, path helpers.
Heavy ops belong on worker threads — never the UI thread.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def is_image_corrupted(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return True
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return False
    except Exception:
        pass
    try:
        from PIL import Image as PilImage

        with PilImage.open(path) as img:
            img.verify()
        return False
    except Exception:
        return True


def load_image_rgb(path: str | Path) -> Optional[np.ndarray]:
    try:
        import cv2

        raw = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.error("load_image_rgb failed %s: %s", path, exc)
        return None


def image_dimensions(path: str | Path) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image as PilImage

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
