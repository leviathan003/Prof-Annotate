"""
bytemark/config/skeleton.py
19-keypoint skeleton connectivity for yolo11s segpose.
!! Replace SKELETON_CONNECTIONS with your provided definition !!
"""

KEYPOINT_NAMES: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
    17: "left_foot",
    18: "right_foot",
}

# !! REPLACE WITH YOUR PROVIDED CONNECTIVITY DEFINITION !!
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (12, 14),
    (14, 16),
    (16, 18),
]

VISIBILITY_NOT_LABELED = 0
VISIBILITY_LABELED_HIDDEN = 1
VISIBILITY_LABELED_VISIBLE = 2

SYMMETRY_PAIRS: list[tuple[int, int]] = [
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

NUM_KEYPOINTS = len(KEYPOINT_NAMES)
assert NUM_KEYPOINTS == 19
