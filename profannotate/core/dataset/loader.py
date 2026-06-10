"""
profannotate/core/dataset/loader.py
Loads a validated YOLO11 dataset root into memory-efficient structures.
Uses lazy per-image loading — only metadata indexed at open time.

Hot-path performance: `load_dataset` pre-scans the labels directory once
into a stem-set instead of doing per-image `exists() + stat()`. On a
10k-image dataset this drops indexing from 20k+ syscalls to ~2.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from profannotate.config.constants import (
    DATA_YAML_FILENAME,
    NUM_KEYPOINTS,
    YOLO_IMAGE_EXTS,
    YOLO_IMAGES_SUBDIR,
    YOLO_LABEL_EXT,
    YOLO_LABELS_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)
from profannotate.utils.image import derive_label_path, is_image_corrupted

logger = logging.getLogger(__name__)

_DEFAULT_NUM_KPT = NUM_KEYPOINTS
_LBL_EXT_LEN = len(YOLO_LABEL_EXT)


def _build_label_stem_set(labels_dir: Path) -> set[str]:
    """One-pass scan of a labels directory. Returns the set of label-file
    stems (filename without `.txt`) that exist and are non-empty.

    Replaces two per-image syscalls (`exists()` + `stat()`) with one
    `os.scandir` over the labels directory.
    """
    stems: set[str] = set()
    if not labels_dir.is_dir():
        return stems
    try:
        it = os.scandir(labels_dir)
    except OSError:
        return stems
    with it:
        for entry in it:
            name = entry.name
            if not name.endswith(YOLO_LABEL_EXT):
                continue
            try:
                if entry.is_file() and entry.stat().st_size > 0:
                    stems.add(name[:-_LBL_EXT_LEN])
            except OSError:
                continue
    return stems


def _scan_images(img_dir: Path) -> list[os.DirEntry]:
    """Return image DirEntry objects in stable lexicographic order.

    Using `os.scandir` over `Path.iterdir()` saves one stat per entry
    (DirEntry caches `is_file()` from the directory readdir result).
    """
    if not img_dir.is_dir():
        return []
    try:
        it = os.scandir(img_dir)
    except OSError:
        return []
    out: list[os.DirEntry] = []
    with it:
        for entry in it:
            name = entry.name
            dot = name.rfind(".")
            if dot < 0:
                continue
            if name[dot:].lower() not in YOLO_IMAGE_EXTS:
                continue
            if not entry.is_file():
                continue
            out.append(entry)
    out.sort(key=lambda e: e.name)
    return out


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
    active_keypoint_names: list[str] = field(default_factory=list)
    kpt_config_synthesized: bool = False

    # Cached split sub-lists. Populated by `freeze()`; consumed by nav code
    # that previously filtered the full entries list every step. None until
    # frozen so callers that mutate `entries` post-construction don't see
    # stale views.
    _train_entries: list[ImageEntry] | None = field(default=None, repr=False, compare=False)
    _val_entries: list[ImageEntry] | None = field(default=None, repr=False, compare=False)
    _annotated_count: int | None = field(default=None, repr=False, compare=False)
    _corrupted_count: int | None = field(default=None, repr=False, compare=False)

    @property
    def num_keypoints(self) -> int:
        return len(self.active_keypoint_names) if self.active_keypoint_names else _DEFAULT_NUM_KPT

    @property
    def train_entries(self) -> list[ImageEntry]:
        if self._train_entries is None:
            self._train_entries = [e for e in self.entries if e.split == YOLO_TRAIN_DIR]
        return self._train_entries

    @property
    def val_entries(self) -> list[ImageEntry]:
        if self._val_entries is None:
            self._val_entries = [e for e in self.entries if e.split == YOLO_VAL_DIR]
        return self._val_entries

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def annotated_count(self) -> int:
        if self._annotated_count is None:
            self._annotated_count = sum(1 for e in self.entries if e.has_label)
        return self._annotated_count

    @property
    def corrupted_count(self) -> int:
        if self._corrupted_count is None:
            self._corrupted_count = sum(1 for e in self.entries if e.is_corrupted)
        return self._corrupted_count

    def freeze(self) -> None:
        """Pre-compute the cached views. Call once after `entries` is fully
        populated by the loader — avoids per-property re-walks during nav."""
        self._train_entries = [e for e in self.entries if e.split == YOLO_TRAIN_DIR]
        self._val_entries = [e for e in self.entries if e.split == YOLO_VAL_DIR]
        self._annotated_count = sum(1 for e in self.entries if e.has_label)
        self._corrupted_count = sum(1 for e in self.entries if e.is_corrupted)

    def invalidate_cache(self) -> None:
        """Discard cached views — call if `entries` is mutated after freeze."""
        self._train_entries = None
        self._val_entries = None
        self._annotated_count = None
        self._corrupted_count = None


def _apply_yaml_kpt_config(index: DatasetIndex) -> None:
    """
    Read kpt_shape / keypoint_names from data.yaml into the index.
    If both are missing, infer the count from existing label files (or fall back
    to the full skeleton) and write the result back so the yaml is canonical.
    """
    from profannotate.config.skeleton import KEYPOINT_NAMES
    from profannotate.core.dataset.yaml_handler import (
        _detect_num_keypoints,
        load_yaml,
        save_yaml,
    )

    data = load_yaml(index.root)
    names = data.get("keypoint_names")
    shape = data.get("kpt_shape")

    if isinstance(names, list) and names:
        index.active_keypoint_names = list(names)
        if not (isinstance(shape, list) and len(shape) == 2 and shape[0] == len(names)):
            data["kpt_shape"] = [len(names), 3]
            data["keypoint_names"] = list(names)
            save_yaml(index.root, data)
        return

    if isinstance(shape, list) and len(shape) == 2 and isinstance(shape[0], int):
        n = shape[0]
        default_names = [KEYPOINT_NAMES[i] for i in sorted(KEYPOINT_NAMES)]
        synthesized = (
            default_names[:n] if n <= len(default_names) else [f"kpt_{i}" for i in range(n)]
        )
        index.active_keypoint_names = synthesized
        data["keypoint_names"] = synthesized
        save_yaml(index.root, data)
        return

    # Neither key present — infer from labels, fall back to full skeleton.
    default_names = [KEYPOINT_NAMES[i] for i in sorted(KEYPOINT_NAMES)]
    detected = _detect_num_keypoints(index.root)
    if detected is not None and detected != len(default_names):
        synthesized = [f"kpt_{i}" for i in range(detected)]
    else:
        synthesized = default_names
    index.active_keypoint_names = synthesized
    index.kpt_config_synthesized = True
    data["kpt_shape"] = [len(synthesized), 3]
    data["keypoint_names"] = synthesized
    save_yaml(index.root, data)


def load_flat_dataset(root: Path) -> DatasetIndex:
    root = Path(root).resolve()
    index = DatasetIndex(root=root)
    yaml = root / DATA_YAML_FILENAME
    if yaml.exists():
        index.yaml_path = yaml

    # Walk via os.walk: avoids Path-allocation overhead of rglob and lets us
    # batch-check label existence with a single stat per sibling label file.
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            dot = fn.rfind(".")
            if dot < 0 or fn[dot:].lower() not in YOLO_IMAGE_EXTS:
                continue
            img_path = Path(dirpath) / fn
            lbl_path = derive_label_path(img_path)
            has_label = False
            try:
                st = lbl_path.stat()
                has_label = st.st_size > 0
            except OSError:
                pass
            entry = ImageEntry(
                image_path=img_path,
                label_path=lbl_path,
                split=YOLO_TRAIN_DIR,
                is_corrupted=False,
                has_label=has_label,
            )
            index.entries.append(entry)

    index.entries.sort(key=lambda e: str(e.image_path))
    _apply_yaml_kpt_config(index)
    index.freeze()
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
        if not img_dir.is_dir():
            continue
        lbl_dir = root / YOLO_LABELS_SUBDIR / split
        # One pass over labels gives O(1) per-image lookups below.
        label_stems = _build_label_stem_set(lbl_dir)
        for entry in _scan_images(img_dir):
            name = entry.name
            dot = name.rfind(".")
            stem = name[:dot]
            lbl_path = lbl_dir / (stem + YOLO_LABEL_EXT)
            index.entries.append(
                ImageEntry(
                    image_path=Path(entry.path),
                    label_path=lbl_path,
                    split=split,
                    is_corrupted=False,
                    has_label=stem in label_stems,
                )
            )

    _apply_yaml_kpt_config(index)
    index.freeze()
    logger.info("Indexed %d images in %s", index.total, root)
    return index


def stream_dataset(
    root: str | Path,
    chunk_size: int = 500,
):
    """Generator that yields DatasetIndex chunks for progressive UI updates.
    Each yield is a partial index — caller accumulates entries."""
    root = Path(root).resolve()
    index = DatasetIndex(root=root)
    yaml = root / DATA_YAML_FILENAME
    if yaml.exists():
        index.yaml_path = yaml

    chunk: list[ImageEntry] = []

    for split in (YOLO_TRAIN_DIR, YOLO_VAL_DIR):
        img_dir = root / YOLO_IMAGES_SUBDIR / split
        if not img_dir.is_dir():
            continue
        lbl_dir = root / YOLO_LABELS_SUBDIR / split
        label_stems = _build_label_stem_set(lbl_dir)
        for entry in _scan_images(img_dir):
            name = entry.name
            dot = name.rfind(".")
            stem = name[:dot]
            lbl_path = lbl_dir / (stem + YOLO_LABEL_EXT)
            chunk.append(
                ImageEntry(
                    image_path=Path(entry.path),
                    label_path=lbl_path,
                    split=split,
                    is_corrupted=False,
                    has_label=stem in label_stems,
                )
            )
            if len(chunk) >= chunk_size:
                index.entries.extend(chunk)
                chunk = []
                yield index, False  # partial

    if chunk:
        index.entries.extend(chunk)

    _apply_yaml_kpt_config(index)
    index.freeze()
    yield index, True  # final
