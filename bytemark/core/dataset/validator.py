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
            lbl = _find_label_sibling(img, source)
            if lbl is None or not lbl.exists():
                lbl = img.with_suffix(YOLO_LABEL_EXT)
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


# ── Dataset Diagnosis ─────────────────────────────────────────────────────────

SCENARIO_EMPTY = "empty"
SCENARIO_IMAGES_ONLY_FLAT = "images_only_flat"
SCENARIO_LABELS_ONLY = "labels_only"
SCENARIO_STRUCTURED_ALL_EMPTY = "structured_all_empty"
SCENARIO_STRUCTURED_LABELS_EMPTY = "structured_labels_empty"
SCENARIO_STRUCTURED_ONE_SPLIT = "structured_one_split"
SCENARIO_OK = "ok"


@dataclass
class DatasetDiagnosis:
    scenario: str
    root: Path
    flat_image_count: int = 0
    train_image_count: int = 0
    val_image_count: int = 0
    train_label_count: int = 0
    val_label_count: int = 0
    has_train_dir: bool = False
    has_val_dir: bool = False

    @property
    def total_structured_images(self) -> int:
        return self.train_image_count + self.val_image_count

    @property
    def total_structured_labels(self) -> int:
        return self.train_label_count + self.val_label_count

    @property
    def active_split(self) -> str:
        """When only one split has images, returns its directory name."""
        return YOLO_TRAIN_DIR if self.train_image_count >= self.val_image_count else YOLO_VAL_DIR


def diagnose_dataset(root: Path) -> DatasetDiagnosis:
    """
    Fast structural scan: only calls iterdir() on the four split dirs, then
    does bounded next()-rglob for quick existence checks.  No image decoding.
    """
    root = Path(root).resolve()
    diag = DatasetDiagnosis(scenario=SCENARIO_EMPTY, root=root)

    images_dir = root / YOLO_IMAGES_SUBDIR
    labels_dir = root / YOLO_LABELS_SUBDIR

    has_images_dir = images_dir.exists() and images_dir.is_dir()
    has_labels_dir = labels_dir.exists() and labels_dir.is_dir()

    def _count_ext(directory: Optional[Path], extensions: set[str]) -> int:
        if directory is None or not directory.is_dir():
            return 0
        return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)

    label_exts = {YOLO_LABEL_EXT}

    train_img_dir: Optional[Path] = images_dir / YOLO_TRAIN_DIR if has_images_dir else None
    val_img_dir: Optional[Path] = images_dir / YOLO_VAL_DIR if has_images_dir else None
    train_lbl_dir: Optional[Path] = labels_dir / YOLO_TRAIN_DIR if has_labels_dir else None
    val_lbl_dir: Optional[Path] = labels_dir / YOLO_VAL_DIR if has_labels_dir else None

    diag.has_train_dir = bool(train_img_dir and train_img_dir.is_dir())
    diag.has_val_dir = bool(val_img_dir and val_img_dir.is_dir())

    diag.train_image_count = _count_ext(train_img_dir, YOLO_IMAGE_EXTS)
    diag.val_image_count = _count_ext(val_img_dir, YOLO_IMAGE_EXTS)
    diag.train_label_count = _count_ext(train_lbl_dir, label_exts)
    diag.val_label_count = _count_ext(val_lbl_dir, label_exts)

    # Images outside the two structured split dirs (flat / mis-structured)
    structured_dirs = {d for d in (train_img_dir, val_img_dir) if d is not None}
    diag.flat_image_count = sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in YOLO_IMAGE_EXTS and p.parent not in structured_dirs
    )

    # ── Classify ──────────────────────────────────────────────────────────────
    has_any_split = diag.has_train_dir or diag.has_val_dir

    if not has_images_dir or not has_any_split:
        # No YOLO split structure present
        if diag.flat_image_count == 0:
            has_any_lbl = next(root.rglob(f"*{YOLO_LABEL_EXT}"), None) is not None
            diag.scenario = SCENARIO_LABELS_ONLY if has_any_lbl else SCENARIO_EMPTY
        else:
            diag.scenario = SCENARIO_IMAGES_ONLY_FLAT
        return diag

    if diag.total_structured_images == 0:
        # Dirs exist but no images inside them
        has_any_lbl = (
            diag.total_structured_labels > 0
            or next(root.rglob(f"*{YOLO_LABEL_EXT}"), None) is not None
        )
        diag.scenario = SCENARIO_LABELS_ONLY if has_any_lbl else SCENARIO_STRUCTURED_ALL_EMPTY
        return diag

    # At least one split has images — check balance
    if (diag.train_image_count == 0) != (diag.val_image_count == 0):
        diag.scenario = SCENARIO_STRUCTURED_ONE_SPLIT
        return diag

    # Both splits populated
    diag.scenario = (
        SCENARIO_OK if diag.total_structured_labels > 0 else SCENARIO_STRUCTURED_LABELS_EMPTY
    )
    return diag
