"""
bytemark/core/dataset/yaml_handler.py
Reads, writes, and auto-generates data.yaml for YOLO11 datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config.constants import (
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
        body_dict = {k: v for k, v in data.items() if k not in ("kpt_shape", "keypoint_names")}
        kpt_shape = data.get("kpt_shape")
        kpt_names = data.get("keypoint_names")

        body = yaml.dump(
            body_dict, default_flow_style=False, allow_unicode=True, sort_keys=True
        )

        extras: list[str] = []
        if kpt_shape is not None:
            try:
                n, c = int(kpt_shape[0]), int(kpt_shape[1])
                extras.append(f"kpt_shape: [{n}, {c}]")
            except (TypeError, ValueError, IndexError):
                extras.append(
                    "kpt_shape: "
                    + yaml.dump(kpt_shape, default_flow_style=True, allow_unicode=True).strip()
                )
        if isinstance(kpt_names, list) and kpt_names:
            extras.append("")
            extras.append("# Keypoint ordering — index → name (positional; do not reorder)")
            flow = "[" + ", ".join(str(n) for n in kpt_names) + "]"
            extras.append(f"keypoint_names: {flow}")

        text = body.rstrip("\n")
        if extras:
            text += "\n\n" + "\n".join(extras)
        text += "\n"

        path.write_text(text, encoding="utf-8")
        return True
    except Exception as exc:
        logger.error("Failed to save data.yaml: %s", exc)
        return False


def generate_yaml(root: str | Path, class_names: list[str] | None = None) -> dict[str, Any]:
    """
    Auto-generate a data.yaml for a dataset root.
    Detects class count from existing label files if class_names not provided.
    Always writes kpt_shape + keypoint_names — inferred from label files when possible,
    otherwise defaults to the full skeleton.
    """
    from src.config.skeleton import KEYPOINT_NAMES

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

    default_names = [KEYPOINT_NAMES[i] for i in sorted(KEYPOINT_NAMES)]
    detected_kpts = _detect_num_keypoints(root)
    if detected_kpts is not None and detected_kpts != len(default_names):
        kpt_names = [f"kpt_{i}" for i in range(detected_kpts)]
    else:
        kpt_names = default_names
    data["kpt_shape"] = [len(kpt_names), 3]
    data["keypoint_names"] = kpt_names

    save_yaml(root, data)
    logger.info(
        "Generated data.yaml at %s with %d classes, %d keypoints",
        root,
        data["nc"],
        len(kpt_names),
    )
    return data


def _detect_num_keypoints(root: Path) -> int | None:
    """
    Scan label files for pose-only lines (class + 4 bbox + 3·N kpts) and return N
    if it's consistent across files. Returns None if no pose-only line is found
    or the inferred counts disagree.
    """
    from src.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR

    lbl_dir = root / YOLO_LABELS_SUBDIR
    if not lbl_dir.exists():
        return None

    candidates: set[int] = set()
    for lbl in lbl_dir.rglob(f"*{YOLO_LABEL_EXT}"):
        try:
            with lbl.open("r") as fh:
                for line in fh:
                    parts = line.strip().split()
                    n = len(parts) - 1  # drop class_id
                    if n <= 4:
                        continue
                    rest = n - 4
                    if rest % 3 == 0 and rest > 0:
                        candidates.add(rest // 3)
                        if len(candidates) > 1:
                            return None
        except OSError:
            continue
    return next(iter(candidates), None) if len(candidates) == 1 else None


def _detect_classes(root: Path) -> list[str]:
    """Scan label files to find unique class IDs, return generic names."""
    from src.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR

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
