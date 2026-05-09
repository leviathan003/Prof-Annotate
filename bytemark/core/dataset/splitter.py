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
    """
    Copy images (and labels if present) from `source` into
    `dest/images/train`, `dest/images/val`, etc.
    Returns {"train": n, "val": m}.
    """
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
            lbl = img.with_suffix(YOLO_LABEL_EXT)
            if lbl.exists():
                shutil.copy2(lbl, lbl_out / lbl.name)

        counts[split_name] = len(split_images)
        logger.info("Split %s: %d images", split_name, len(split_images))

    return counts
