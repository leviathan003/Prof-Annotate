"""
profannotate/utils/filesystem.py
Path helpers and dataset filesystem utilities.
"""

from __future__ import annotations

from pathlib import Path

from profannotate.config.constants import (
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in YOLO_IMAGE_EXTS


def is_label_file(path: Path) -> bool:
    return path.is_file() and path.suffix == YOLO_LABEL_EXT


def label_for_image(image_path: Path, dataset_root: Path) -> Path:
    """
    Given an image path inside dataset_root/.../images/split/img.jpg,
    return the corresponding label path .../labels/split/img.txt.
    """
    try:
        rel = image_path.relative_to(dataset_root)
        parts = list(rel.parts)
        for i, p in enumerate(parts):
            if p == YOLO_IMAGES_SUBDIR:
                parts[i] = YOLO_LABELS_SUBDIR
                break
        return (dataset_root / Path(*parts)).with_suffix(YOLO_LABEL_EXT)
    except ValueError:
        return image_path.with_suffix(YOLO_LABEL_EXT)


def image_for_label(label_path: Path, dataset_root: Path) -> Path | None:
    """Reverse of label_for_image — find the image for a given label."""
    try:
        rel = label_path.relative_to(dataset_root)
        parts = list(rel.parts)
        for i, p in enumerate(parts):
            if p == YOLO_LABELS_SUBDIR:
                parts[i] = YOLO_IMAGES_SUBDIR
                break
        stem = Path(*parts).with_suffix("")
        for ext in YOLO_IMAGE_EXTS:
            candidate = dataset_root / stem.with_suffix(ext)
            if candidate.exists():
                return candidate
    except ValueError:
        pass
    return None


def safe_stem(name: str, existing: set[str]) -> str:
    """Return a unique stem by appending _N if name already exists."""
    if name not in existing:
        return name
    i = 1
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"
