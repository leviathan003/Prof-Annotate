"""
profannotate/core/annotation/writer.py
Atomic YOLO11 label file writer.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from profannotate.config.constants import (
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)
from profannotate.core.annotation.models import Annotation, ImageAnnotations
from profannotate.utils.image import derive_label_path

logger = logging.getLogger(__name__)


def label_path_for_image(root: Path, image_path: Path) -> Path:
    """Map an image path to its label path using the known dataset structure.

    For a structured root (`root/images/<split>/<stem>.<ext>`) this returns
    `root/labels/<split>/<stem>.txt` deterministically — matching the loader's
    convention and avoiding `derive_label_path`'s first-"images" replacement,
    which mis-maps when an ancestor directory is itself named "images".

    Falls back to `derive_label_path` when the image is not under `root` or the
    first relative component is not the images subdir.
    """
    root = Path(root)
    image_path = Path(image_path)
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        return derive_label_path(image_path)
    parts = rel.parts
    if parts and parts[0].lower() == YOLO_IMAGES_SUBDIR:
        rebuilt = root / YOLO_LABELS_SUBDIR / Path(*parts[1:])
        return rebuilt.with_suffix(YOLO_LABEL_EXT)
    return derive_label_path(image_path)


def dataset_writable(root: Path) -> bool:
    """Best-effort check of whether label files can be created/saved under `root`.

    Checks the `labels/` directory when it already exists (that's where splits
    are written), otherwise the root itself (where `labels/` would be created).
    Used to warn the annotator when a dataset sits on read-only media.
    """
    root = Path(root)
    labels = root / YOLO_LABELS_SUBDIR
    target = labels if labels.is_dir() else root
    return os.access(target, os.W_OK)


def _touch_empty_label(path: Path) -> bool:
    """Create a 0-byte label file at `path` only if it does not already exist.

    Uses O_CREAT|O_EXCL so an existing label (empty or not) is never clobbered,
    which also makes this safe against concurrent writers. Returns True iff a
    new file was created.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError as exc:
        logger.error("Failed to create empty label %s: %s", path, exc)
        return False
    os.close(fd)
    return True


def materialize_empty_labels(root: Path) -> int:
    """Ensure every image in a structured dataset has a label file.

    For each split (`train`, `val`) that has an `images/<split>` directory:
      - create the matching `labels/<split>` directory (so the labels folder
        exists even when the split holds no images), and
      - create a 0-byte `.txt` for every image that lacks one.

    Existing label files are left untouched (see `_touch_empty_label`). Returns
    the number of empty label files newly created. Idempotent.
    """
    root = Path(root)
    created = 0
    ext_len = len(YOLO_LABEL_EXT)
    for split in (YOLO_TRAIN_DIR, YOLO_VAL_DIR):
        img_dir = root / YOLO_IMAGES_SUBDIR / split
        if not img_dir.is_dir():
            continue
        lbl_dir = root / YOLO_LABELS_SUBDIR / split
        try:
            lbl_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Read-only dataset (read-only media / no write permission): skip
            # label creation rather than abort the open — viewing still works.
            logger.warning("Cannot create labels dir %s: %s", lbl_dir, exc)
            continue

        # One scandir of the labels dir gives O(1) membership checks below, so
        # the common re-open case (every label already present) costs a single
        # readdir instead of one failed open() syscall per image.
        existing: set[str] = set()
        try:
            with os.scandir(lbl_dir) as lit:
                for le in lit:
                    n = le.name
                    if n.endswith(YOLO_LABEL_EXT):
                        existing.add(n[:-ext_len])
        except OSError:
            pass

        try:
            it = os.scandir(img_dir)
        except OSError:
            continue
        with it:
            for entry in it:
                name = entry.name
                dot = name.rfind(".")
                if dot < 0 or name[dot:].lower() not in YOLO_IMAGE_EXTS:
                    continue
                stem = name[:dot]
                if stem in existing:
                    continue
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                if _touch_empty_label(lbl_dir / (stem + YOLO_LABEL_EXT)):
                    created += 1
    return created


def write_label_file(annotations: ImageAnnotations) -> bool:
    label_path = Path(annotations.label_path)
    try:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [_serialize(inst) for inst in annotations.instances]
        fd, tmp = tempfile.mkstemp(dir=label_path.parent, prefix=".profannotate_", suffix=".tmp")
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
