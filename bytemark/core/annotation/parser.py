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

logger = logging.getLogger(__name__)


def parse_label_file(
    image_path: str | Path,
    label_path: str | Path,
    num_keypoints: int = NUM_KEYPOINTS,
) -> ImageAnnotations:
    # Note: corruption detection is intentionally *not* run here. The legacy
    # implementation called `is_image_corrupted` on every parse — which
    # `cv2.imread`s the image just to probe. That meant every navigation step
    # decoded the image twice (once here, once in the canvas's loader thread).
    # The canvas's loader now reports failure via _on_image_failed instead,
    # which is sufficient signal.
    image_path = Path(image_path)
    label_path = Path(label_path)

    result = ImageAnnotations(
        image_path=str(image_path),
        label_path=str(label_path),
        is_corrupted=False,
    )

    try:
        raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return result
    except OSError as exc:
        logger.error("Cannot read label file %s: %s", label_path, exc)
        return result

    effective_n = _detect_file_num_keypoints(raw_lines, num_keypoints)
    if effective_n != num_keypoints:
        logger.info(
            "%s: kpt count %d in file does not match dataset default %d — parsing with %d",
            label_path,
            effective_n,
            num_keypoints,
            effective_n,
        )

    for lineno, raw in enumerate(raw_lines, start=1):
        line = raw.strip()
        if not line:
            continue
        ann = _parse_line(line, lineno, label_path, effective_n)
        if ann is not None:
            result.instances.append(ann)

    return result


def _line_accepts(n: int, num_keypoints: int) -> bool:
    """Return True if a YOLO line with `n` fields-after-class-id matches any
    known shape for the given kpt count."""
    if n == 4:
        return True
    pose_fields = 3 * num_keypoints
    if n == 4 + pose_fields:
        return True
    if n > 4 + pose_fields and (n - 4 - pose_fields) % 2 == 0:
        return True
    if n >= 6 and n % 2 == 0:
        return True
    return False


def _detect_file_num_keypoints(lines: list[str], default_n: int) -> int:
    """Find the kpt count consistent with every non-empty line in the file.

    Prefers `default_n` when it fits. Otherwise scans a window of nearby values
    and picks the closest fit. Falls back to `default_n` if nothing matches —
    `_parse_line` will then warn line-by-line.
    """
    counts: list[int] = []
    for raw in lines:
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        counts.append(len(parts) - 1)
    if not counts:
        return default_n
    if all(_line_accepts(n, default_n) for n in counts):
        return default_n
    # Search a reasonable window around the default and small kpt-counts.
    candidates = sorted(set(range(0, 33)), key=lambda k: (abs(k - default_n), k))
    for k in candidates:
        if k == default_n:
            continue
        if all(_line_accepts(n, k) for n in counts):
            return k
    return default_n


def _parse_line(
    line: str,
    lineno: int,
    label_path: Path,
    num_keypoints: int = NUM_KEYPOINTS,
) -> Optional[Annotation]:
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

    pose_fields = 3 * num_keypoints
    if n == 4 + pose_fields:
        return _parse_pose(class_id, rest, lineno, label_path, num_keypoints)

    # Combined must be checked BEFORE seg-only: a bbox+kpts+seg line with an even
    # kpt count yields an even n that would otherwise be mis-parsed as seg-only.
    if n > 4 + pose_fields and (n - 4 - pose_fields) % 2 == 0:
        return _parse_combined(class_id, rest, lineno, label_path, num_keypoints)

    if n >= 6 and n % 2 == 0:
        return _parse_segmentation(class_id, rest, lineno, label_path)

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


def _parse_pose(class_id, rest, lineno, path, num_keypoints: int = NUM_KEYPOINTS):
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


def _parse_combined(class_id, rest, lineno, path, num_keypoints: int = NUM_KEYPOINTS):
    try:
        vals = _floats(rest)
        bbox = BBox(*vals[:4])
        pe = 4 + 3 * num_keypoints
        kv = vals[4:pe]
        kpts = [Keypoint(kv[i], kv[i + 1], int(kv[i + 2])) for i in range(0, len(kv), 3)]
        sv = vals[pe:]
        mask = SegmentationMask(points=[(sv[i], sv[i + 1]) for i in range(0, len(sv), 2)])
        return Annotation(class_id=class_id, bbox=bbox, keypoints=kpts, mask=mask)
    except (ValueError, IndexError) as e:
        logger.warning("%s:%d combined error: %s", path, lineno, e)
        return None
