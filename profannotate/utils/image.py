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

_PIL_IMAGE = None  # cached module — the import dance below is per-call otherwise


def _pil_image_module():
    """Import PIL.Image robustly whether frozen or not (cached)."""
    global _PIL_IMAGE
    if _PIL_IMAGE is not None:
        return _PIL_IMAGE
    _PIL_IMAGE = _import_pil_image()
    return _PIL_IMAGE


def _import_pil_image():
    import importlib

    try:
        mod = importlib.import_module("PIL.Image")
        return mod
    except ImportError:
        pass
    try:
        import os
        import sys

        # When frozen, PIL submodules may need explicit path injection.
        # _MEIPASS is PyInstaller-only — absent under Nuitka despite frozen.
        meipass = getattr(sys, "_MEIPASS", None)
        if getattr(sys, "frozen", False) and meipass:
            pil_path = os.path.join(meipass, "PIL")
            if pil_path not in sys.path:
                sys.path.insert(0, meipass)
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


def load_image_rgb(path: str | Path, max_side: int | None = None) -> Optional[np.ndarray]:
    """Decode an image to an RGB uint8 array.

    `max_side` enables JPEG draft-mode decoding: libjpeg decodes directly at
    1/2, 1/4 or 1/8 scale so the result's longest side is >= max_side without
    ever decoding full resolution — near-free, and a huge win on weak CPUs.
    No-op for non-JPEG formats and for images already at or below the cap.
    Saved annotation coordinates are normalized, so they are unaffected.
    """
    try:
        PilImage = _pil_image_module()
        with PilImage.open(path) as img:
            if max_side:
                img.draft("RGB", (max_side, max_side))
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
    lowered = [p.lower() for p in parts]
    # Replace the LAST "images" component (the dataset dir nearest the file);
    # matching the first mis-maps when an ancestor dir is itself named
    # "images" (e.g. /srv/images/project/images/train/x.jpg).
    if "images" in lowered:
        idx = len(lowered) - 1 - lowered[::-1].index("images")
        parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")
