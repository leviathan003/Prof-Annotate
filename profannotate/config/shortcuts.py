"""
profannotate/config/shortcuts.py
All keyboard shortcuts in one place. Never hardcode Qt.Key_* elsewhere.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

DRAW_BBOX = Qt.Key.Key_B
DRAW_KEYPOINT = Qt.Key.Key_K
DRAW_SEGMENTATION = Qt.Key.Key_S

SAVE_ANNOTATION = QKeySequence.StandardKey.Save
UNDO = QKeySequence.StandardKey.Undo
TRIGGER_AUTO_ANNOTATE = QKeySequence("Ctrl+Y")
DELETE_POINT = Qt.Key.Key_Delete

NUDGE_UP = Qt.Key.Key_Up
NUDGE_DOWN = Qt.Key.Key_Down
NUDGE_LEFT = Qt.Key.Key_Left
NUDGE_RIGHT = Qt.Key.Key_Right

NEXT_IMAGE = Qt.Key.Key_D
PREV_IMAGE = Qt.Key.Key_A
OPEN_FOLDER = QKeySequence("Ctrl+O")
CONFIRM = Qt.Key.Key_Return
CANCEL = Qt.Key.Key_Escape
HELP = Qt.Key.Key_F1

TOGGLE_BBOX_VISIBILITY = QKeySequence("Ctrl+1")
TOGGLE_KEYPOINT_VISIBILITY = QKeySequence("Ctrl+2")
TOGGLE_SEGMENTATION_VISIBILITY = QKeySequence("Ctrl+3")

AUTOANNOTATE_HUMAN_WARNING = (
    "Auto-annotations are currently optimised for human subjects only.\n"
    "A generic multi-class model is coming soon — patience, Annotator."
)

SHORTCUT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Dataset",
        [
            ("Create New Dataset", "Ctrl+Shift+N"),
            ("Open Dataset Folder", "Ctrl+O"),
            ("Open Single Image", "Ctrl+F"),
            ("Save Annotation", "Ctrl+S"),
            ("Undo", "Ctrl+Z"),
            ("Bulk Keypoint Removal", "Ctrl+Del"),
        ],
    ),
    (
        "Drawing Modes",
        [
            ("Draw BBox", "B"),
            ("Draw Keypoints (bbox first)", "K"),
            ("Draw Segmentation (bbox first)", "S"),
            ("Auto-annotate Current Image", "Ctrl+Shift+A  /  Ctrl+Y"),
            ("Close Segmentation Mask", "Double-click"),
            ("Skip Current Keypoint", "Right-click"),
            ("Undo Last Segmentation Point", "Ctrl+Z (in S mode)"),
        ],
    ),
    (
        "Navigation",
        [
            ("Next Image", "D  /  →"),
            ("Previous Image", "A  /  ←"),
            ("Pan Canvas", "Middle Mouse"),
            ("Zoom Canvas", "Scroll Wheel"),
        ],
    ),
    (
        "Selection & Editing",
        [
            ("Select Annotation", "Left Click"),
            ("Nudge Selected Point", "Arrow Keys"),
            ("Delete Selected Point / Instance", "Del"),
            ("Deselect / Exit Mode", "Esc"),
            ("Accept Auto-annotate Diff", "Enter"),
            ("Reject Auto-annotate Diff", "Esc"),
        ],
    ),
    (
        "View Toggles",
        [
            ("Toggle BBox Visibility", "Ctrl+1"),
            ("Toggle Keypoint Visibility", "Ctrl+2"),
            ("Toggle Segmentation Visibility", "Ctrl+3"),
        ],
    ),
]

SHORTCUT_LABELS: dict[str, str] = {
    label: keys for _, items in SHORTCUT_GROUPS for label, keys in items
}
