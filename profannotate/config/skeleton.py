"""
profannotate/config/skeleton.py

Keypoint-schema registry. The app ships a single whole-body seg-pose model
(`wb_s_full100_best.onnx`, 133 COCO-WholeBody keypoints), but the keypoint
system is schema-driven so a re-headed model only needs a new schema entry.

- `KeypointSchema` describes one canonical keypoint layout: ordered names,
  skeleton connections, left/right symmetry pairs, and preset groups.
- `BODY_19` is the original custom 19-kpt body schema (kept for back-compat /
  legacy datasets). `WHOLEBODY_133` is the COCO-WholeBody layout the shipped
  model emits.
- The active schema (default `WHOLEBODY_133`) drives the module-level aliases
  `KEYPOINT_NAMES`, `SKELETON_CONNECTIONS`, `SYMMETRY_PAIRS`, `NUM_KEYPOINTS`
  so existing `from skeleton import KEYPOINT_NAMES` imports keep working.

NOTE on `from skeleton import KEYPOINT_NAMES`: such imports bind the alias at
import time, so they reflect the *default* schema. Code that must follow a
runtime schema swap should call `get_active_schema()` / `connections_for()`
directly rather than the bare alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VISIBILITY_NOT_LABELED = 0
VISIBILITY_LABELED_HIDDEN = 1
VISIBILITY_LABELED_VISIBLE = 2


@dataclass(frozen=True)
class KeypointSchema:
    """One canonical keypoint layout.

    `keypoint_names` maps the *positional* model-channel index → name. Order is
    significant: it must match the model's output channel order exactly.
    `groups` maps a preset label → the ordered list of keypoint names it covers.
    """

    name: str
    keypoint_names: dict[int, str]
    connections: list[tuple[int, int]]
    symmetry_pairs: list[tuple[int, int]] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.keypoint_names)
        # Indices must be a dense 0..n-1 range — they are positional channels.
        if set(self.keypoint_names) != set(range(n)):
            raise ValueError(
                f"{self.name}: keypoint indices must be dense 0..{n - 1}, "
                f"got {sorted(self.keypoint_names)}"
            )
        for a, b in self.connections:
            if not (0 <= a < n and 0 <= b < n):
                raise ValueError(f"{self.name}: connection ({a},{b}) out of range 0..{n - 1}")
        known = set(self.keypoint_names.values())
        for label, members in self.groups.items():
            unknown = [m for m in members if m not in known]
            if unknown:
                raise ValueError(f"{self.name}: group '{label}' has unknown names {unknown}")

    @property
    def count(self) -> int:
        return len(self.keypoint_names)

    def names_in_order(self) -> list[str]:
        return [self.keypoint_names[i] for i in range(self.count)]


# ── BODY_19 — original custom body schema ──────────────────────────────────────

_BODY_19_NAMES: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_mouth",
    4: "right_mouth",
    5: "left_ear",
    6: "right_ear",
    7: "left_shoulder",
    8: "right_shoulder",
    9: "left_elbow",
    10: "right_elbow",
    11: "left_wrist",
    12: "right_wrist",
    13: "left_hip",
    14: "right_hip",
    15: "left_knee",
    16: "right_knee",
    17: "left_ankle",
    18: "right_ankle",
}

_BODY_19_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (1, 5),
    (2, 6),
    (5, 7),
    (6, 8),
    (7, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (10, 12),
    (7, 13),
    (8, 14),
    (13, 14),
    (13, 15),
    (14, 16),
    (15, 17),
    (16, 18),
]

_BODY_19_SYMMETRY: list[tuple[int, int]] = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
]

BODY_19 = KeypointSchema(
    name="body19",
    keypoint_names=_BODY_19_NAMES,
    connections=_BODY_19_CONNECTIONS,
    symmetry_pairs=_BODY_19_SYMMETRY,
    groups={
        "Face": [
            "nose",
            "left_eye",
            "right_eye",
            "left_mouth",
            "right_mouth",
            "left_ear",
            "right_ear",
        ],
        "Hands": ["left_wrist", "right_wrist"],
        "Whole body": [_BODY_19_NAMES[i] for i in range(19)],
    },
)


# ── WHOLEBODY_133 — COCO-WholeBody layout (the shipped model) ───────────────────
#
# Canonical ordering (positional channels 0..132):
#   0–16   body (COCO-17)
#   17–22  feet
#   23–90  face (68 landmarks)
#   91–111 left hand (21)
#   112–132 right hand (21)
# Connections / names follow the COCO-WholeBody / mmpose definition. They are
# built programmatically below to avoid transcription errors across 130+ links.

_WB_BODY = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
_WB_FEET = [
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]
_FACE_OFFSET = 23
_LHAND_OFFSET = 91
_RHAND_OFFSET = 112

_HAND_PARTS = ["thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger"]


def _build_wholebody_names() -> dict[int, str]:
    names: list[str] = list(_WB_BODY) + list(_WB_FEET)
    # Face: 68 landmarks → face-0 .. face-67
    names += [f"face-{i}" for i in range(68)]
    # Hands: root + 4 joints per finger
    for side in ("left", "right"):
        names.append(f"{side}_hand_root")
        for part in _HAND_PARTS:
            for j in range(1, 5):
                names.append(f"{side}_{part}{j}")
    return {i: n for i, n in enumerate(names)}


def _build_wholebody_connections() -> list[tuple[int, int]]:
    c: list[tuple[int, int]] = []

    # Body (COCO-17 skeleton)
    c += [
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ]
    # Feet: ankle → big toe / small toe / heel
    c += [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]

    # Face (68-point) — sequential chains within each region, eyes/mouth closed.
    def f(i: int) -> int:
        return _FACE_OFFSET + i

    def chain(idxs: list[int], closed: bool = False) -> None:
        for a, b in zip(idxs, idxs[1:]):
            c.append((f(a), f(b)))
        if closed and len(idxs) > 2:
            c.append((f(idxs[-1]), f(idxs[0])))

    chain(list(range(0, 17)))  # jaw contour
    chain(list(range(17, 22)))  # left eyebrow
    chain(list(range(22, 27)))  # right eyebrow
    chain(list(range(27, 31)))  # nose bridge
    chain(list(range(31, 36)))  # lower nose
    chain(list(range(36, 42)), True)  # left eye
    chain(list(range(42, 48)), True)  # right eye
    chain(list(range(48, 60)), True)  # outer lip
    chain(list(range(60, 68)), True)  # inner lip

    # Hands: root → each finger base, then along the finger.
    for off in (_LHAND_OFFSET, _RHAND_OFFSET):
        for fi in range(5):  # 5 fingers
            base = off + 1 + fi * 4
            c.append((off, base))  # root → finger joint 1
            for j in range(3):
                c.append((base + j, base + j + 1))  # along the finger
    return c


def _build_wholebody_symmetry(names: dict[int, str]) -> list[tuple[int, int]]:
    name_to_idx = {v: k for k, v in names.items()}
    out: list[tuple[int, int]] = []
    for nm, idx in name_to_idx.items():
        if nm.startswith("left_"):
            mirror = "right_" + nm[len("left_") :]
            if mirror in name_to_idx:
                out.append((idx, name_to_idx[mirror]))
    return out


_WB_NAMES = _build_wholebody_names()

WHOLEBODY_133 = KeypointSchema(
    name="wholebody133",
    keypoint_names=_WB_NAMES,
    connections=_build_wholebody_connections(),
    symmetry_pairs=_build_wholebody_symmetry(_WB_NAMES),
    groups={
        "Body": _WB_BODY,
        "Feet": _WB_FEET,
        "Face": [_WB_NAMES[i] for i in range(_FACE_OFFSET, _FACE_OFFSET + 68)],
        "Left hand": [_WB_NAMES[i] for i in range(_LHAND_OFFSET, _LHAND_OFFSET + 21)],
        "Right hand": [_WB_NAMES[i] for i in range(_RHAND_OFFSET, _RHAND_OFFSET + 21)],
        "Hands": [_WB_NAMES[i] for i in range(_LHAND_OFFSET, _RHAND_OFFSET + 21)],
        "Whole body": [_WB_NAMES[i] for i in range(133)],
    },
)


# ── Registry + active-schema accessors ──────────────────────────────────────────

SCHEMAS: dict[str, KeypointSchema] = {s.name: s for s in (BODY_19, WHOLEBODY_133)}

_ACTIVE: KeypointSchema = WHOLEBODY_133


def get_active_schema() -> KeypointSchema:
    return _ACTIVE


def schema_for_kpt_count(k: int) -> KeypointSchema | None:
    """Return the registered schema whose keypoint count is `k`, else None."""
    for s in SCHEMAS.values():
        if s.count == k:
            return s
    return None


def set_active_schema(name: str) -> None:
    """Activate a registered schema and rebind the module-level aliases.

    See the module docstring caveat: `from skeleton import KEYPOINT_NAMES`
    bindings made before the swap are not updated; prefer `get_active_schema()`.
    """
    global _ACTIVE, KEYPOINT_NAMES, SKELETON_CONNECTIONS, SYMMETRY_PAIRS, NUM_KEYPOINTS
    if name not in SCHEMAS:
        raise KeyError(f"unknown keypoint schema '{name}'; known: {sorted(SCHEMAS)}")
    _ACTIVE = SCHEMAS[name]
    KEYPOINT_NAMES = _ACTIVE.keypoint_names
    SKELETON_CONNECTIONS = _ACTIVE.connections
    SYMMETRY_PAIRS = _ACTIVE.symmetry_pairs
    NUM_KEYPOINTS = _ACTIVE.count


def connections_for(active_names: list[str]) -> list[tuple[int, int]]:
    """Remap the active schema's connections to indices within `active_names`.

    Only links whose *both* endpoints are present in the subset survive. This is
    the single source of truth for the subset-remap the overlays did inline.
    """
    schema = _ACTIVE
    name_to_new = {n: i for i, n in enumerate(active_names)}
    out: list[tuple[int, int]] = []
    for a, b in schema.connections:
        na = schema.keypoint_names.get(a)
        nb = schema.keypoint_names.get(b)
        if na in name_to_new and nb in name_to_new:
            out.append((name_to_new[na], name_to_new[nb]))
    return out


# ── Active-schema aliases (back-compat; reflect the default schema) ─────────────

KEYPOINT_NAMES: dict[int, str] = _ACTIVE.keypoint_names
SKELETON_CONNECTIONS: list[tuple[int, int]] = _ACTIVE.connections
SYMMETRY_PAIRS: list[tuple[int, int]] = _ACTIVE.symmetry_pairs
NUM_KEYPOINTS: int = _ACTIVE.count
