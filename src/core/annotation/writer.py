"""
bytemark/core/annotation/writer.py
Atomic YOLO11 label file writer.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from src.core.annotation.models import Annotation, ImageAnnotations

logger = logging.getLogger(__name__)


def write_label_file(annotations: ImageAnnotations) -> bool:
    label_path = Path(annotations.label_path)
    try:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [_serialize(inst) for inst in annotations.instances]
        fd, tmp = tempfile.mkstemp(dir=label_path.parent, prefix=".bytemark_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
                if lines:
                    fh.write("\n")
            os.replace(tmp, label_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except OSError as exc:
        logger.error("Failed to write %s: %s", label_path, exc)
        return False


def _serialize(ann: Annotation) -> str:
    parts: list[str] = [str(ann.class_id)]

    if ann.has_bbox():
        b = ann.bbox
        parts += [_f(b.cx), _f(b.cy), _f(b.w), _f(b.h)]

    if ann.has_keypoints() and ann.has_bbox():
        for kp in ann.keypoints:
            if kp is None:
                parts += ["0", "0", "0"]
            else:
                parts += [_f(kp.x), _f(kp.y), str(kp.visibility)]

    if ann.has_mask():
        for x, y in ann.mask.points:
            parts += [_f(x), _f(y)]

    return " ".join(parts)


def _f(v: float, precision: int = 6) -> str:
    return f"{v:.{precision}f}".rstrip("0").rstrip(".")
