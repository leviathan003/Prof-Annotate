"""
profannotate/core/recovery/autosave.py
Persists unsaved annotation state to ~/.profannotate/sessions/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from profannotate.config.constants import SESSION_CACHE_DIR
from profannotate.core.annotation.models import (
    Annotation,
    BBox,
    ImageAnnotations,
    Keypoint,
    SegmentationMask,
)

logger = logging.getLogger(__name__)
_VERSION = 1


def _key(root: str | Path) -> str:
    return hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]


def _path(root: str | Path) -> Path:
    return SESSION_CACHE_DIR / f"{_key(root)}.json"


def save_session(root: str | Path, dirty: dict[str, ImageAnnotations]) -> bool:
    SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _VERSION,
        "dataset_root": str(Path(root).resolve()),
        "saved_at": time.time(),
        "annotations": {k: _ser_img(v) for k, v in dirty.items()},
    }
    try:
        target = _path(root)
        tmp = target.with_suffix(".tmp")
        # Compact JSON — the file is machine-only, no indent needed. Saves
        # both serialization time and disk space on large dirty maps.
        tmp.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(target)
        return True
    except OSError as exc:
        logger.error("Autosave failed: %s", exc)
        return False


def load_session(root: str | Path) -> Optional[dict[str, ImageAnnotations]]:
    p = _path(root)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("version") != _VERSION:
            return None
        return {k: _deser_img(v) for k, v in payload.get("annotations", {}).items()}
    except Exception as exc:
        logger.error("Session load failed: %s", exc)
        return None


def clear_session(root: str | Path) -> None:
    _path(root).unlink(missing_ok=True)


def _ser_img(a: ImageAnnotations) -> dict:
    return {
        "image_path": a.image_path,
        "label_path": a.label_path,
        "is_corrupted": a.is_corrupted,
        "instances": [_ser_ann(i) for i in a.instances],
    }


def _ser_ann(a: Annotation) -> dict:
    d: dict = {"class_id": a.class_id}
    if a.bbox:
        d["bbox"] = {"cx": a.bbox.cx, "cy": a.bbox.cy, "w": a.bbox.w, "h": a.bbox.h}
    if a.keypoints:
        d["keypoints"] = [
            {"x": k.x, "y": k.y, "v": k.visibility} if k is not None else None
            for k in a.keypoints
        ]
    if a.mask:
        d["mask"] = {"points": a.mask.points}
    return d


def _deser_img(d: dict) -> ImageAnnotations:
    a = ImageAnnotations(
        image_path=d["image_path"],
        label_path=d["label_path"],
        is_corrupted=d.get("is_corrupted", False),
    )
    for i in d.get("instances", []):
        a.instances.append(_deser_ann(i))
    return a


def _deser_ann(d: dict) -> Annotation:
    bbox = None
    if "bbox" in d:
        b = d["bbox"]
        bbox = BBox(b["cx"], b["cy"], b["w"], b["h"])
    kpts = None
    if "keypoints" in d:
        kpts = [
            Keypoint(k["x"], k["y"], k.get("v", 2)) if k is not None else None
            for k in d["keypoints"]
        ]
    mask = None
    if "mask" in d:
        mask = SegmentationMask(points=[tuple(p) for p in d["mask"]["points"]])
    return Annotation(class_id=d["class_id"], bbox=bbox, keypoints=kpts, mask=mask)
