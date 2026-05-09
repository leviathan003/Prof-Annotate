"""
bytemark/core/dataset/yaml_handler.py
Reads, writes, and auto-generates data.yaml for YOLO11 datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from bytemark.config.constants import (
    DATA_YAML_FILENAME,
    YOLO_IMAGES_SUBDIR,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE: dict[str, Any] = {
    "path": "",  # filled at generation time
    "train": f"{YOLO_IMAGES_SUBDIR}/{YOLO_TRAIN_DIR}",
    "val": f"{YOLO_IMAGES_SUBDIR}/{YOLO_VAL_DIR}",
    "nc": 1,
    "names": ["object"],
}


def load_yaml(root: str | Path) -> dict[str, Any]:
    path = Path(root) / DATA_YAML_FILENAME
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Failed to load data.yaml: %s", exc)
        return {}


def save_yaml(root: str | Path, data: dict[str, Any]) -> bool:
    path = Path(root) / DATA_YAML_FILENAME
    try:
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as exc:
        logger.error("Failed to save data.yaml: %s", exc)
        return False


def generate_yaml(root: str | Path, class_names: list[str] | None = None) -> dict[str, Any]:
    """
    Auto-generate a data.yaml for a dataset root.
    Detects class count from existing label files if class_names not provided.
    """
    root = Path(root).resolve()
    data = dict(_DEFAULT_TEMPLATE)
    data["path"] = str(root)

    if class_names:
        data["nc"] = len(class_names)
        data["names"] = class_names
    else:
        detected = _detect_classes(root)
        data["nc"] = max(1, len(detected))
        data["names"] = detected if detected else ["object"]

    save_yaml(root, data)
    logger.info("Generated data.yaml at %s with %d classes", root, data["nc"])
    return data


def _detect_classes(root: Path) -> list[str]:
    """Scan label files to find unique class IDs, return generic names."""
    from bytemark.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR

    ids: set[int] = set()
    lbl_dir = root / YOLO_LABELS_SUBDIR
    if not lbl_dir.exists():
        return []

    for lbl in lbl_dir.rglob(f"*{YOLO_LABEL_EXT}"):
        try:
            with lbl.open("r") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if parts:
                        try:
                            ids.add(int(parts[0]))
                        except ValueError:
                            pass
        except OSError:
            pass

    return [f"class_{i}" for i in sorted(ids)]
