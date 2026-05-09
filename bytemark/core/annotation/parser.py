"""
bytemark/core/annotation/parser.py
Reads YOLO11 label files into annotation model objects.
Line-by-line streaming — never loads entire file at once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bytemark.config.constants import NUM_KEYPOINTS
from bytemark.core.annotation.models import (
    Annotation,
    BBox,
    ImageAnnotations,
    Keypoint,
    SegmentationMask,
)
from bytemark.utils.image import is_image_corrupted

logger = logging.getLogger(__name__)


def parse_label_file(image_path: str | Path, label_path: str | Path) -> ImageAnnotations:
    image_path = Path(image_path)
    label_path = Path(label_path)

    corrupted = is_image_corrupted(image_path)
    result = ImageAnnotations(
        image_path=str(image_path),
        label_path=str(label_path),
        is_corrupted=corrupted,
    )

    if not label_path.exists():
        return result

    try:
        with label_path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                ann = _parse_line(line, lineno, label_path)
                if ann is not None:
                    result.instances.append(ann)
    except OSError as exc:
        logger.error("Cannot read label file %s: %s", label_path, exc)

    return result


def _parse_line(line: str, lineno: int, label_path: Path) -> Optional[Annotation]:
    parts = line.split()
    if not parts:
        return None

    try:
        class_id = int(parts[0])
    except ValueError:
        logger.warning("%s:%d — invalid class id: %s", label_path, lineno, parts[0])
        return None

    rest = parts[1:]
    n = len(rest)

    if n == 4:
        return _parse_bbox_only(class_id, rest, lineno, label_path)

    pose_fields = 3 * NUM_KEYPOINTS
    if n == 4 + pose_fields:
        return _parse_pose(class_id, rest, lineno, label_path)

    if n >= 6 and n % 2 == 0:
        return _parse_segmentation(class_id, rest, lineno, label_path)

    if n > 4 + pose_fields and (n - 4 - pose_fields) % 2 == 0:
        return _parse_combined(class_id, rest, lineno, label_path)

    logger.warning("%s:%d — unrecognised field count %d", label_path, lineno, n)
    return None


def _floats(parts: list[str]) -> list[float]:
    return [float(p) for p in parts]


def _parse_bbox_only(class_id, rest, lineno, path):
    try:
        cx, cy, w, h = _floats(rest)
        return Annotation(class_id=class_id, bbox=BBox(cx, cy, w, h))
    except ValueError as e:
        logger.warning("%s:%d bbox error: %s", path, lineno, e)
        return None


def _parse_pose(class_id, rest, lineno, path):
    try:
        vals = _floats(rest)
        bbox = BBox(*vals[:4])
        kv = vals[4:]
        kpts = [Keypoint(kv[i], kv[i + 1], int(kv[i + 2])) for i in range(0, len(kv), 3)]
        return Annotation(class_id=class_id, bbox=bbox, keypoints=kpts)
    except (ValueError, IndexError) as e:
        logger.warning("%s:%d pose error: %s", path, lineno, e)
        return None


def _parse_segmentation(class_id, rest, lineno, path):
    try:
        vals = _floats(rest)
        mask = SegmentationMask(points=[(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)])
        return Annotation(class_id=class_id, mask=mask)
    except (ValueError, IndexError) as e:
        logger.warning("%s:%d seg error: %s", path, lineno, e)
        return None


def _parse_combined(class_id, rest, lineno, path):
    try:
        vals = _floats(rest)
        bbox = BBox(*vals[:4])
        pe = 4 + 3 * NUM_KEYPOINTS
        kv = vals[4:pe]
        kpts = [Keypoint(kv[i], kv[i + 1], int(kv[i + 2])) for i in range(0, len(kv), 3)]
        sv = vals[pe:]
        mask = SegmentationMask(points=[(sv[i], sv[i + 1]) for i in range(0, len(sv), 2)])
        return Annotation(class_id=class_id, bbox=bbox, keypoints=kpts, mask=mask)
    except (ValueError, IndexError) as e:
        logger.warning("%s:%d combined error: %s", path, lineno, e)
        return None
