"""
bytemark/core/dataset/validator.py
YOLO11 format validation and reshuffling.
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bytemark.config.constants import (
    DATA_YAML_FILENAME,
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    root: Path
    issues: list[str] = field(default_factory=list)
    image_count: int = 0
    label_count: int = 0
    has_yaml: bool = False


def validate_dataset(root: str | Path) -> ValidationResult:
    root = Path(root).resolve()
    issues = []

    if not root.exists():
        return ValidationResult(False, root, ["Root directory does not exist."])
    if not root.is_dir():
        return ValidationResult(False, root, ["Path is not a directory."])

    images_dir = root / YOLO_IMAGES_SUBDIR
    labels_dir = root / YOLO_LABELS_SUBDIR

    if not images_dir.exists():
        issues.append(f"Missing '{YOLO_IMAGES_SUBDIR}/' directory.")
    if not labels_dir.exists():
        issues.append(f"Missing '{YOLO_LABELS_SUBDIR}/' directory.")

    if images_dir.exists():
        if not (images_dir / YOLO_TRAIN_DIR).exists():
            issues.append(f"Missing '{YOLO_IMAGES_SUBDIR}/{YOLO_TRAIN_DIR}/' split.")
        if not (images_dir / YOLO_VAL_DIR).exists():
            issues.append(f"Missing '{YOLO_IMAGES_SUBDIR}/{YOLO_VAL_DIR}/' split.")

    return ValidationResult(
        is_valid=len(issues) == 0,
        root=root,
        issues=issues,
        image_count=_count_images(root),
        label_count=_count_labels(root),
        has_yaml=(root / DATA_YAML_FILENAME).exists(),
    )


def reshuffle_into_yolo_format(
    source: str | Path,
    dest: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    source = Path(source).resolve()
    dest = Path(dest).resolve()

    images = [p for p in source.rglob("*") if p.suffix.lower() in YOLO_IMAGE_EXTS and p.is_file()]
    if not images:
        raise ValueError(f"No images found in {source}")

    rng = random.Random(seed)
    rng.shuffle(images)

    split_idx = max(1, int(len(images) * train_ratio))
    splits = {
        YOLO_TRAIN_DIR: images[:split_idx],
        YOLO_VAL_DIR: images[split_idx:],
    }

    for split_name, split_images in splits.items():
        img_out = dest / YOLO_IMAGES_SUBDIR / split_name
        lbl_out = dest / YOLO_LABELS_SUBDIR / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, img_out / img.name)
            lbl = img.with_suffix(YOLO_LABEL_EXT)
            if not lbl.exists():
                lbl = _find_label_sibling(img, source)
            if lbl and lbl.exists():
                shutil.copy2(lbl, lbl_out / lbl.name)

    logger.info("Reshuffled %d images → %s", len(images), dest)
    return dest


def _find_label_sibling(img: Path, root: Path) -> Optional[Path]:
    try:
        parts = list(img.relative_to(root).parts)
        for i, p in enumerate(parts):
            if p.lower() == "images":
                parts[i] = "labels"
                break
        return (root / Path(*parts)).with_suffix(YOLO_LABEL_EXT)
    except Exception:
        return None


def _count_images(root: Path) -> int:
    d = root / YOLO_IMAGES_SUBDIR
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in YOLO_IMAGE_EXTS) if d.exists() else 0


def _count_labels(root: Path) -> int:
    d = root / YOLO_LABELS_SUBDIR
    return sum(1 for p in d.rglob("*") if p.suffix == YOLO_LABEL_EXT) if d.exists() else 0
