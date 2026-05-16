"""
bytemark/config/shortcuts.py
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

SHORTCUT_LABELS: dict[str, str] = {
    "Draw BBox": "B",
    "Draw Keypoints": "K",
    "Draw Segmentation": "S",
    "Save Annotation": "Ctrl+S",
    "Undo": "Ctrl+Z",
    "Auto-annotate Image": "Ctrl+Shift+A",
    "Delete Selected Point": "Del",
    "Nudge Point": "Arrow Keys",
    "Next Image": "D",
    "Previous Image": "A",
    "Open Folder": "Ctrl+O",
    "Confirm Dialog": "Enter",
    "Cancel / Close": "Esc",
    "Help": "F1",
    "Toggle BBox View": "Ctrl+1",
    "Toggle Keypoint View": "Ctrl+2",
    "Toggle Segmentation View": "Ctrl+3",
}
