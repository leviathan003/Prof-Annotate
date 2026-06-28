"""
profannotate/config/pose_template.py

Canonical *diagram* layout for keypoint schemas — a readable reference pose used
by the selection-time skeleton preview (NOT anatomically to scale). Each schema
maps every keypoint name → normalized (x, y) in [0, 1], where x grows rightward
and y grows downward (Qt screen convention).

For the 133-kpt whole-body schema the layout is a composite: a central body
stick figure plus a face inset and two hand insets, because 68 face + 42 hand
points are illegible crammed onto one small head/hands.

`reference_pose(schema)` always returns a coord for every name in the schema
(falls back to a grid for unknown schemas), so the preview can never KeyError.
"""

from __future__ import annotations

import math

from profannotate.config.skeleton import KeypointSchema

# Region rectangles within the [0,1] diagram: (x0, y0, x1, y1).
_BODY_RECT = (0.34, 0.04, 0.66, 0.96)
_FACE_RECT = (0.02, 0.04, 0.30, 0.40)
_LHAND_RECT = (0.02, 0.58, 0.28, 0.96)
_RHAND_RECT = (0.70, 0.58, 0.98, 0.96)


def _map(
    local: tuple[float, float], rect: tuple[float, float, float, float]
) -> tuple[float, float]:
    lx, ly = local
    x0, y0, x1, y1 = rect
    return (x0 + lx * (x1 - x0), y0 + ly * (y1 - y0))


# ── Body stick figure (local unit box) ──────────────────────────────────────────
# Keyed by the COCO-WholeBody body+feet names. "left_*" is drawn on the viewer's
# left — this is a schematic legend, not a mirrored medical figure.
_BODY_LOCAL: dict[str, tuple[float, float]] = {
    "nose": (0.50, 0.07),
    "left_eye": (0.45, 0.05),
    "right_eye": (0.55, 0.05),
    "left_ear": (0.40, 0.07),
    "right_ear": (0.60, 0.07),
    # BODY_19 extras (mouth corners) — harmless for WB133 (names absent there).
    "left_mouth": (0.46, 0.10),
    "right_mouth": (0.54, 0.10),
    "left_shoulder": (0.38, 0.20),
    "right_shoulder": (0.62, 0.20),
    "left_elbow": (0.30, 0.33),
    "right_elbow": (0.70, 0.33),
    "left_wrist": (0.26, 0.46),
    "right_wrist": (0.74, 0.46),
    "left_hip": (0.43, 0.52),
    "right_hip": (0.57, 0.52),
    "left_knee": (0.41, 0.72),
    "right_knee": (0.59, 0.72),
    "left_ankle": (0.40, 0.90),
    "right_ankle": (0.60, 0.90),
    "left_heel": (0.40, 0.94),
    "left_big_toe": (0.36, 0.98),
    "left_small_toe": (0.44, 0.98),
    "right_heel": (0.60, 0.94),
    "right_big_toe": (0.64, 0.98),
    "right_small_toe": (0.56, 0.98),
}


def _face_local() -> list[tuple[float, float]]:
    """68 landmarks in a local unit box, by region (matches schema face chains)."""
    pts: list[tuple[float, float]] = []

    # 0-16 jaw contour: arc across the lower face.
    for i in range(17):
        t = i / 16.0
        x = 0.15 + 0.70 * t
        y = 0.42 + 0.45 * math.sin(math.pi * t)  # peak at chin
        pts.append((x, y))
    # 17-21 left eyebrow, 22-26 right eyebrow.
    for i in range(5):
        pts.append((0.20 + 0.055 * i, 0.30 - 0.02 * math.sin(math.pi * i / 4)))
    for i in range(5):
        pts.append((0.575 + 0.055 * i, 0.30 - 0.02 * math.sin(math.pi * i / 4)))
    # 27-30 nose bridge (vertical).
    for i in range(4):
        pts.append((0.50, 0.36 + 0.05 * i))
    # 31-35 lower nose.
    for i in range(5):
        pts.append((0.42 + 0.04 * i, 0.57))
    # 36-41 left eye ring, 42-47 right eye ring.
    for cx in (0.34, 0.66):
        for i in range(6):
            a = 2 * math.pi * i / 6
            pts.append((cx + 0.06 * math.cos(a), 0.40 + 0.035 * math.sin(a)))
    # 48-59 outer lip ring (12), 60-67 inner lip ring (8).
    for i in range(12):
        a = 2 * math.pi * i / 12
        pts.append((0.50 + 0.12 * math.cos(a), 0.74 + 0.06 * math.sin(a)))
    for i in range(8):
        a = 2 * math.pi * i / 8
        pts.append((0.50 + 0.07 * math.cos(a), 0.74 + 0.035 * math.sin(a)))
    return pts


def _hand_local(mirror: bool) -> list[tuple[float, float]]:
    """21 points: root + 5 fingers × 4 joints, fanning up from a palm root.

    Order matches the schema: root, then thumb, forefinger, middle, ring, pinky.
    `mirror` flips horizontally (used for the left hand so the thumb sits inward).
    """
    root = (0.5, 0.92)
    pts: list[tuple[float, float]] = [root]
    # angle from vertical (deg, +right) and relative length per finger.
    fingers = [(-52, 0.78), (-22, 0.96), (0, 1.0), (20, 0.94), (40, 0.82)]
    for ang_deg, scale in fingers:
        a = math.radians(ang_deg)
        for j in range(1, 5):
            r = (0.16 + 0.16 * (j - 1)) * scale
            x = root[0] + r * math.sin(a)
            y = root[1] - r * math.cos(a)
            pts.append((x, y))
    if mirror:
        pts = [(1.0 - x, y) for (x, y) in pts]
    return pts


def _wholebody_pose(schema: KeypointSchema) -> dict[str, tuple[float, float]]:
    names = schema.keypoint_names
    pose: dict[str, tuple[float, float]] = {}

    # Body + feet.
    for i in range(0, 23):
        nm = names[i]
        if nm in _BODY_LOCAL:
            pose[nm] = _map(_BODY_LOCAL[nm], _BODY_RECT)

    # Face 23-90.
    face = _face_local()
    for k, (lx, ly) in enumerate(face):
        pose[names[23 + k]] = _map((lx, ly), _FACE_RECT)

    # Hands: left 91-111 (mirrored), right 112-132.
    lh = _hand_local(mirror=True)
    for k, loc in enumerate(lh):
        pose[names[91 + k]] = _map(loc, _LHAND_RECT)
    rh = _hand_local(mirror=False)
    for k, loc in enumerate(rh):
        pose[names[112 + k]] = _map(loc, _RHAND_RECT)

    return pose


def _body_only_pose(schema: KeypointSchema) -> dict[str, tuple[float, float]]:
    """BODY_19 (or any all-on-the-figure schema) mapped onto the body rect."""
    pose: dict[str, tuple[float, float]] = {}
    for nm in schema.names_in_order():
        if nm in _BODY_LOCAL:
            pose[nm] = _map(_BODY_LOCAL[nm], (0.20, 0.04, 0.80, 0.96))
    return pose


def _grid_pose(schema: KeypointSchema) -> dict[str, tuple[float, float]]:
    """Fallback: lay every name out on a grid so coverage is guaranteed."""
    names = schema.names_in_order()
    n = len(names)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    pose: dict[str, tuple[float, float]] = {}
    for i, nm in enumerate(names):
        r, c = divmod(i, cols)
        x = (c + 0.5) / cols
        y = (r + 0.5) / rows
        pose[nm] = (x, y)
    return pose


def reference_pose(schema: KeypointSchema) -> dict[str, tuple[float, float]]:
    """Normalized diagram coords for every keypoint name in `schema`."""
    if schema.name == "wholebody133":
        pose = _wholebody_pose(schema)
    elif schema.name == "body19":
        pose = _body_only_pose(schema)
    else:
        pose = {}

    # Guarantee full coverage regardless of schema specifics.
    missing = [nm for nm in schema.names_in_order() if nm not in pose]
    if missing:
        if not pose:
            return _grid_pose(schema)
        # place any stragglers along the top edge
        for i, nm in enumerate(missing):
            pose[nm] = ((i + 0.5) / max(1, len(missing)), 0.01)
    return pose
