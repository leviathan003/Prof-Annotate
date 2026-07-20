"""
profannotate/core/dataset/migrate.py

One-time import normalization for external, standard YOLO-segmentation datasets.

Standard YOLO-seg stores a bare polygon per instance (`class x1 y1 x2 y2 …`, the
box is implicit). This tool's canonical form keeps an explicit box
(`class cx cy w h x1 y1 …`) so a mask is always bbox-anchored. The two are
ambiguous by field count alone, so on import we detect bare-polygon datasets and
rewrite them once to the canonical form (deriving each box from its polygon),
recording `kpt_shape: [0, 3]`. Idempotent: a canonical line already contains its
box and is left untouched.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from profannotate.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR
from profannotate.core.annotation.models import BBox

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _looks_bboxed(vals: list[float]) -> bool:
    """True if `vals` reads as a valid `cx cy w h <polygon>` line: the leading 4
    form a box with w,h in (0,1] that contains every trailing point. Our own
    canonical seg lines always pass (bbox-contains-mask is enforced at save); a
    bare polygon almost never does."""
    if len(vals) < 4 + 6:  # need a box + at least a 3-point polygon
        return False
    cx, cy, w, h = vals[:4]
    if not (0.0 < w <= 1.0 + _EPS and 0.0 < h <= 1.0 + _EPS):
        return False
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    pts = vals[4:]
    if len(pts) % 2 != 0:
        return False
    for i in range(0, len(pts), 2):
        px, py = pts[i], pts[i + 1]
        if not (x1 - _EPS <= px <= x2 + _EPS and y1 - _EPS <= py <= y2 + _EPS):
            return False
    return True


def _is_bare_polygon(vals: list[float]) -> bool:
    """True if `vals` is a bare (bbox-less) polygon: an even count of ≥6 numbers
    that does NOT already read as a bbox+polygon."""
    return len(vals) >= 6 and len(vals) % 2 == 0 and not _looks_bboxed(vals)


def _iter_label_files(root: Path):
    lbl_dir = root / YOLO_LABELS_SUBDIR
    if lbl_dir.exists():
        yield from lbl_dir.rglob(f"*{YOLO_LABEL_EXT}")


def looks_like_yolo_seg(root: str | Path, sample_limit: int = 5000) -> bool:
    """Heuristic: a foreign dataset whose polygon lines are bare (no explicit
    box). Only fires when data.yaml declares no kpt_shape (i.e. not already a
    configured pose/detect dataset) and a majority of the even-count lines are
    bare polygons."""
    from profannotate.core.dataset.yaml_handler import load_yaml

    root = Path(root)
    meta = load_yaml(root)
    if meta.get("kpt_shape") is not None or meta.get("keypoint_names"):
        return False  # already a declared dataset — don't second-guess it

    bare = bboxed = 0
    scanned = 0
    for lbl in _iter_label_files(root):
        if scanned >= sample_limit:
            break
        try:
            with lbl.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 7:  # class + < a 3-point polygon
                        continue
                    scanned += 1
                    try:
                        vals = [float(p) for p in parts[1:]]
                    except ValueError:
                        continue
                    if len(vals) % 2 != 0:
                        continue  # odd → pose-ish, not a polygon line
                    if _looks_bboxed(vals):
                        bboxed += 1
                    else:
                        bare += 1
                    if scanned >= sample_limit:
                        break
        except OSError:
            continue
    return bare > bboxed and bare > 0


def _f(v: float, precision: int = 6) -> str:
    return f"{v:.{precision}f}".rstrip("0").rstrip(".")


def _convert_line(line: str) -> str | None:
    """Return the canonical rewrite of a bare-polygon line, or None to keep it."""
    parts = line.split()
    if len(parts) < 7:
        return None
    cls = parts[0]
    try:
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if not _is_bare_polygon(vals):
        return None
    pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    box = BBox.from_polygon(pts)
    if box is None:
        return None
    out = [cls, _f(box.cx), _f(box.cy), _f(box.w), _f(box.h)]
    for px, py in pts:
        out += [_f(px), _f(py)]
    return " ".join(out)


def normalize_seg_labels(root: str | Path) -> int:
    """Rewrite every bare-polygon label line under `root` to canonical
    `class bbox <polygon>` (box = polygon bounds), and record kpt_shape [0,3].
    Idempotent. Returns the number of lines converted."""
    root = Path(root)
    converted = 0
    for lbl in _iter_label_files(root):
        try:
            original = lbl.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.error("Cannot read %s: %s", lbl, exc)
            continue
        out_lines: list[str] = []
        changed = False
        for raw in original.splitlines():
            if not raw.strip():
                out_lines.append(raw)
                continue
            rewritten = _convert_line(raw)
            if rewritten is not None:
                out_lines.append(rewritten)
                changed = True
                converted += 1
            else:
                out_lines.append(raw)
        if not changed:
            continue
        try:
            fd, tmp = tempfile.mkstemp(dir=lbl.parent, prefix=".profannotate_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(out_lines))
                if out_lines:
                    fh.write("\n")
            os.replace(tmp, lbl)
        except OSError as exc:
            logger.error("Cannot rewrite %s: %s", lbl, exc)

    # Declare the dataset keypoint-free so it resolves to num_keypoints == 0.
    from profannotate.core.dataset.yaml_handler import generate_yaml, load_yaml, save_yaml

    if (root / "data.yaml").exists():
        data = load_yaml(root)
        data["kpt_shape"] = [0, 3]
        data.pop("keypoint_names", None)
        save_yaml(root, data)
    else:
        generate_yaml(root, num_keypoints=0)
    return converted
