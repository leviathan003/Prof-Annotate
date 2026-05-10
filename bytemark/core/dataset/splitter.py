"""
bytemark/core/dataset/splitter.py
Splits a flat image collection into train/val with random shuffle.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from bytemark.config.constants import (
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)

logger = logging.getLogger(__name__)


def split_dataset(
    source: str | Path,
    dest: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, int]:
    source = Path(source).resolve()
    dest = Path(dest).resolve()

    images = sorted(
        [p for p in source.rglob("*") if p.suffix.lower() in YOLO_IMAGE_EXTS and p.is_file()]
    )
    if not images:
        raise ValueError(f"No images found in {source}")

    rng = random.Random(seed)
    rng.shuffle(images)

    split_idx = max(1, int(len(images) * train_ratio))
    splits = {
        YOLO_TRAIN_DIR: images[:split_idx],
        YOLO_VAL_DIR: images[split_idx:],
    }
    counts = {}

    for split_name, split_images in splits.items():
        img_out = dest / YOLO_IMAGES_SUBDIR / split_name
        lbl_out = dest / YOLO_LABELS_SUBDIR / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, img_out / img.name)
            # Check proper YOLO structure first, fall back to same-dir
            lbl = _find_label_sibling(img, source)
            if lbl is None or not lbl.exists():
                lbl = img.with_suffix(YOLO_LABEL_EXT)
            if lbl and lbl.exists():
                shutil.copy2(lbl, lbl_out / (img.stem + YOLO_LABEL_EXT))

        counts[split_name] = len(split_images)
        logger.info("Split %s: %d images", split_name, len(split_images))

    return counts


def _find_label_sibling(img: Path, source: Path):
    """Resolve label path for a structured YOLO dataset (images/ → labels/)."""
    try:
        parts = list(img.relative_to(source).parts)
        for i, p in enumerate(parts):
            if p.lower() == "images":
                parts[i] = "labels"
                break
        else:
            return None
        from bytemark.config.constants import YOLO_LABEL_EXT

        return (source / Path(*parts)).with_suffix(YOLO_LABEL_EXT)
    except Exception:
        return None
