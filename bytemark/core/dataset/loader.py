"""
bytemark/core/dataset/loader.py
Loads a validated YOLO11 dataset root into memory-efficient structures.
Uses lazy per-image loading — only metadata indexed at open time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from bytemark.config.constants import (
    DATA_YAML_FILENAME,
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)
from bytemark.utils.image import derive_label_path, is_image_corrupted

logger = logging.getLogger(__name__)


@dataclass
class ImageEntry:
    """Lightweight index entry — full annotation loaded on demand."""

    image_path: Path
    label_path: Path
    split: str  # "train" or "val"
    is_corrupted: bool = False
    has_label: bool = False


@dataclass
class DatasetIndex:
    root: Path
    entries: list[ImageEntry] = field(default_factory=list)
    yaml_path: Path | None = None

    @property
    def train_entries(self) -> list[ImageEntry]:
        return [e for e in self.entries if e.split == YOLO_TRAIN_DIR]

    @property
    def val_entries(self) -> list[ImageEntry]:
        return [e for e in self.entries if e.split == YOLO_VAL_DIR]

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def annotated_count(self) -> int:
        return sum(1 for e in self.entries if e.has_label)

    @property
    def corrupted_count(self) -> int:
        return sum(1 for e in self.entries if e.is_corrupted)


def load_flat_dataset(root: Path) -> DatasetIndex:
    root = Path(root).resolve()
    index = DatasetIndex(root=root)
    yaml = root / DATA_YAML_FILENAME
    if yaml.exists():
        index.yaml_path = yaml

    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in YOLO_IMAGE_EXTS or not img_path.is_file():
            continue
        lbl_path = img_path.with_suffix(YOLO_LABEL_EXT)
        entry = ImageEntry(
            image_path=img_path,
            label_path=lbl_path,
            split=YOLO_TRAIN_DIR,
            is_corrupted=False,  # checked lazily on open
            has_label=lbl_path.exists() and lbl_path.stat().st_size > 0,
        )
        index.entries.append(entry)

    logger.info("Indexed %d images (flat) in %s", index.total, root)
    return index


def load_dataset(root: str | Path) -> DatasetIndex:
    root = Path(root).resolve()
    index = DatasetIndex(root=root)
    yaml = root / DATA_YAML_FILENAME
    if yaml.exists():
        index.yaml_path = yaml

    for split in (YOLO_TRAIN_DIR, YOLO_VAL_DIR):
        img_dir = root / YOLO_IMAGES_SUBDIR / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in YOLO_IMAGE_EXTS:
                continue
            lbl_path = (root / YOLO_LABELS_SUBDIR / split / img_path.stem).with_suffix(
                YOLO_LABEL_EXT
            )
            entry = ImageEntry(
                image_path=img_path,
                label_path=lbl_path,
                split=split,
                is_corrupted=False,  # checked lazily on open
                has_label=lbl_path.exists() and lbl_path.stat().st_size > 0,
            )
            index.entries.append(entry)

    logger.info("Indexed %d images in %s", index.total, root)
    return index
