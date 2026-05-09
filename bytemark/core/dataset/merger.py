"""
bytemark/core/dataset/merger.py
Merges two raw dataset directories into a single YOLO11-formatted dataset.
Output name: dataset1_dataset2_merged_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import logging
import random
import shutil
from datetime import datetime
from pathlib import Path

from bytemark.config.constants import (
    MERGED_DATASET_FORMAT,
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)

logger = logging.getLogger(__name__)


def merge_datasets(
    source_a: str | Path,
    source_b: str | Path,
    output_parent: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """
    Collect all images (and labels if present) from both sources,
    shuffle randomly, split train/val, write to output_parent/<merged_name>/.
    Returns the merged dataset root path.
    """
    source_a = Path(source_a).resolve()
    source_b = Path(source_b).resolve()
    output_parent = Path(output_parent).resolve()

    name_a = source_a.name
    name_b = source_b.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_name = (
        MERGED_DATASET_FORMAT.replace("{name1}", name_a)
        .replace("{name2}", name_b)
        .replace("{datetime}", timestamp)
    )
    dest = output_parent / merged_name

    # Collect all images from both sources
    all_images = _collect_images(source_a) + _collect_images(source_b)
    if not all_images:
        raise ValueError("No images found in either source directory.")

    rng = random.Random(seed)
    rng.shuffle(all_images)

    split_idx = max(1, int(len(all_images) * train_ratio))
    splits = {
        YOLO_TRAIN_DIR: all_images[:split_idx],
        YOLO_VAL_DIR: all_images[split_idx:],
    }

    for split_name, images in splits.items():
        img_out = dest / YOLO_IMAGES_SUBDIR / split_name
        lbl_out = dest / YOLO_LABELS_SUBDIR / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        seen_names: set[str] = set()
        for img_path in images:
            # Avoid name collisions by prefixing with source dir name
            stem = img_path.stem
            if stem in seen_names:
                stem = f"{img_path.parent.parent.name}_{stem}"
            seen_names.add(stem)

            out_img = img_out / (stem + img_path.suffix)
            shutil.copy2(img_path, out_img)

            lbl_path = img_path.with_suffix(YOLO_LABEL_EXT)
            if not lbl_path.exists():
                lbl_path = _find_label(img_path)
            if lbl_path and lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / (stem + YOLO_LABEL_EXT))

    logger.info(
        "Merged %d images from '%s' + '%s' → %s",
        len(all_images),
        name_a,
        name_b,
        dest,
    )
    return dest


def _collect_images(source: Path) -> list[Path]:
    return [p for p in source.rglob("*") if p.suffix.lower() in YOLO_IMAGE_EXTS and p.is_file()]


def _find_label(img: Path) -> Path | None:
    parts = list(img.parts)
    for i, p in enumerate(parts):
        if p.lower() == "images":
            parts[i] = "labels"
            candidate = Path(*parts).with_suffix(YOLO_LABEL_EXT)
            return candidate if candidate.exists() else None
    return None
