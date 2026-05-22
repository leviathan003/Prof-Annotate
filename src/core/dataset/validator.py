"""
bytemark/core/dataset/validator.py
YOLO11 format validation and reshuffling.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config.constants import (
    DATA_YAML_FILENAME,
    SCENARIO_EMPTY,
    SCENARIO_IMAGES_ONLY_FLAT,
    SCENARIO_LABELS_ONLY,
    SCENARIO_OK,
    SCENARIO_STRUCTURED_ALL_EMPTY,
    SCENARIO_STRUCTURED_LABELS_EMPTY,
    SCENARIO_STRUCTURED_ONE_SPLIT,
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)

# Re-export for legacy `from bytemark.core.dataset.validator import SCENARIO_*` callers.
__all__ = [
    "SCENARIO_EMPTY",
    "SCENARIO_IMAGES_ONLY_FLAT",
    "SCENARIO_LABELS_ONLY",
    "SCENARIO_OK",
    "SCENARIO_STRUCTURED_ALL_EMPTY",
    "SCENARIO_STRUCTURED_LABELS_EMPTY",
    "SCENARIO_STRUCTURED_ONE_SPLIT",
    "DatasetDiagnosis",
    "diagnose_dataset",
    "reshuffle_into_yolo_format",
]

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


def _count_ext_scandir(directory: Optional[Path], extensions: set[str]) -> int:
    """Count files in `directory` whose lowercase suffix is in `extensions`.
    Uses `os.scandir` so file-type info comes from the readdir result without
    extra per-entry stat calls."""
    if directory is None:
        return 0
    try:
        it = os.scandir(directory)
    except OSError:
        return 0
    count = 0
    with it:
        for entry in it:
            name = entry.name
            dot = name.rfind(".")
            if dot < 0:
                continue
            if name[dot:].lower() not in extensions:
                continue
            try:
                if entry.is_file():
                    count += 1
            except OSError:
                pass
    return count


def _has_any_flat_image_outside(
    root: Path,
    skip_dirs: set[Path],
    cap: int = 1,
) -> int:
    """Return up to `cap` count of image files anywhere under `root` whose
    immediate parent isn't in `skip_dirs`. Short-circuits — once `cap` is
    reached we stop walking. Avoids the full root.rglob('*') on every open."""
    found = 0
    skip_str = {str(d) for d in skip_dirs}
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip walking into either structured dir tree entirely — their
        # contents are accounted for via the split counters.
        if dirpath in skip_str:
            dirnames[:] = []
            continue
        for fn in filenames:
            dot = fn.rfind(".")
            if dot < 0 or fn[dot:].lower() not in YOLO_IMAGE_EXTS:
                continue
            found += 1
            if found >= cap:
                return found
    return found


def _has_any_label_under(root: Path) -> bool:
    """True if any `.txt` file exists under `root`. Walks lazily."""
    for _, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(YOLO_LABEL_EXT):
                return True
    return False


def diagnose_dataset(root: Path) -> DatasetDiagnosis:
    """
    Fast structural scan. Calls `os.scandir` on the four split dirs and only
    performs the (expensive) full-tree walk when there's a reason to —
    specifically when no structured images were found and we need to decide
    between "flat images" and "empty". No image decoding.
    """
    root = Path(root).resolve()
    diag = DatasetDiagnosis(scenario=SCENARIO_EMPTY, root=root)

    images_dir = root / YOLO_IMAGES_SUBDIR
    labels_dir = root / YOLO_LABELS_SUBDIR

    has_images_dir = images_dir.is_dir()
    has_labels_dir = labels_dir.is_dir()

    label_exts = {YOLO_LABEL_EXT}

    train_img_dir: Optional[Path] = images_dir / YOLO_TRAIN_DIR if has_images_dir else None
    val_img_dir: Optional[Path] = images_dir / YOLO_VAL_DIR if has_images_dir else None
    train_lbl_dir: Optional[Path] = labels_dir / YOLO_TRAIN_DIR if has_labels_dir else None
    val_lbl_dir: Optional[Path] = labels_dir / YOLO_VAL_DIR if has_labels_dir else None

    diag.has_train_dir = bool(train_img_dir and train_img_dir.is_dir())
    diag.has_val_dir = bool(val_img_dir and val_img_dir.is_dir())

    diag.train_image_count = _count_ext_scandir(train_img_dir, YOLO_IMAGE_EXTS)
    diag.val_image_count = _count_ext_scandir(val_img_dir, YOLO_IMAGE_EXTS)
    diag.train_label_count = _count_ext_scandir(train_lbl_dir, label_exts)
    diag.val_label_count = _count_ext_scandir(val_lbl_dir, label_exts)

    has_any_split = diag.has_train_dir or diag.has_val_dir
    structured_image_count = diag.train_image_count + diag.val_image_count

    # ── Classify ──────────────────────────────────────────────────────────────
    if not has_images_dir or not has_any_split:
        # No YOLO split structure present — we have to find out whether
        # there are any flat images or labels lurking in the root.
        flat = _has_any_flat_image_outside(root, set(), cap=1)
        if flat == 0:
            diag.flat_image_count = 0
            diag.scenario = (
                SCENARIO_LABELS_ONLY if _has_any_label_under(root) else SCENARIO_EMPTY
            )
        else:
            # Use the full count only when the user-facing prompt needs it.
            diag.flat_image_count = _count_flat_images(root, set())
            diag.scenario = SCENARIO_IMAGES_ONLY_FLAT
        return diag

    structured_dirs = {d for d in (train_img_dir, val_img_dir) if d is not None}

    if structured_image_count == 0:
        # Dirs exist but no images inside them. Check labels.
        has_any_lbl = (
            diag.total_structured_labels > 0 or _has_any_label_under(root)
        )
        diag.scenario = SCENARIO_LABELS_ONLY if has_any_lbl else SCENARIO_STRUCTURED_ALL_EMPTY
        # Flat-count is irrelevant in this branch — leave at 0.
        return diag

    # Structured splits already hold images. The "flat images outside split"
    # number is only consumed by user-facing text in the structured-OK and
    # structured-one-split paths, neither of which surfaces it. Skip the
    # full-tree walk entirely — that was the open-dataset bottleneck.
    diag.flat_image_count = 0

    # At least one split has images — check balance.
    if (diag.train_image_count == 0) != (diag.val_image_count == 0):
        diag.scenario = SCENARIO_STRUCTURED_ONE_SPLIT
        return diag

    # Both splits populated.
    diag.scenario = (
        SCENARIO_OK if diag.total_structured_labels > 0 else SCENARIO_STRUCTURED_LABELS_EMPTY
    )
    return diag


def _count_flat_images(root: Path, skip_dirs: set[Path]) -> int:
    """Exact count — used only when SCENARIO_IMAGES_ONLY_FLAT is selected,
    because the wizard prompt shows the number to the user."""
    skip_str = {str(d) for d in skip_dirs}
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath in skip_str:
            dirnames[:] = []
            continue
        for fn in filenames:
            dot = fn.rfind(".")
            if dot >= 0 and fn[dot:].lower() in YOLO_IMAGE_EXTS:
                count += 1
    return count
