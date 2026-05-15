"""
bytemark/config/skeleton.py
19-keypoint skeleton for yolo11s segpose — corrected ordering and connectivity.
"""

KEYPOINT_NAMES: dict[int, str] = {
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

SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    # Face
    (0, 1),  # nose → left_eye
    (0, 2),  # nose → right_eye
    (1, 3),  # left_eye → left_mouth
    (2, 4),  # right_eye → right_mouth
    (1, 5),  # left_eye → left_ear
    (2, 6),  # right_eye → right_ear
    # Ear → shoulder
    (5, 7),  # left_ear → left_shoulder
    (6, 8),  # right_ear → right_shoulder
    # Shoulder girdle
    (7, 8),  # left_shoulder ↔ right_shoulder
    # Arms
    (7, 9),  # left_shoulder → left_elbow
    (8, 10),  # right_shoulder → right_elbow
    (9, 11),  # left_elbow → left_wrist
    (10, 12),  # right_elbow → right_wrist
    # Torso
    (7, 13),  # left_shoulder → left_hip
    (8, 14),  # right_shoulder → right_hip
    (13, 14),  # left_hip ↔ right_hip
    # Legs
    (13, 15),  # left_hip → left_knee
    (14, 16),  # right_hip → right_knee
    (15, 17),  # left_knee → left_ankle
    (16, 18),  # right_knee → right_ankle
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
