"""
bytemark/config/constants.py
All app-wide constants. No magic numbers anywhere else in the codebase.
"""

from pathlib import Path

APP_NAME = "Prof Annotate"
APP_VERSION = "1.0.0"
APP_DOCS_URL = "https://profannotate.readthedocs.io"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = _PACKAGE_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
ICONS_DIR = ASSETS_DIR / "icons"
STYLES_DIR = ASSETS_DIR / "styles"
MODELS_DIR = _PACKAGE_ROOT / "models"

APP_CACHE_DIR = Path.home() / ".profannotate"
SESSION_CACHE_DIR = APP_CACHE_DIR / "sessions"
PREFS_FILE = APP_CACHE_DIR / "preferences.json"
LAYOUT_FILE = APP_CACHE_DIR / "layout.json"

ONNX_MODEL_FILENAME = "yolo11n_segpose.onnx"
ONNX_MODEL_PATH = MODELS_DIR / ONNX_MODEL_FILENAME
MODEL_INPUT_SIZE = (640, 640)
MODEL_CONF_THRESHOLD = 0.25
MODEL_IOU_THRESHOLD = 0.45
NUM_KEYPOINTS = 19

ONNX_PROVIDERS_PRIORITY = [
    "CUDAExecutionProvider",
    "DirectMLExecutionProvider",
    "CPUExecutionProvider",
]

YOLO_VERSION = "yolo11"
YOLO_LABEL_EXT = ".txt"
YOLO_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
YOLO_TRAIN_DIR = "train"
YOLO_VAL_DIR = "val"
YOLO_IMAGES_SUBDIR = "images"
YOLO_LABELS_SUBDIR = "labels"
DATA_YAML_FILENAME = "data.yaml"

MODALITY_BBOX = "bbox"
MODALITY_KEYPOINTS = "keypoints"
MODALITY_SEGMENTATION = "segmentation"
ALL_MODALITIES = [MODALITY_BBOX, MODALITY_KEYPOINTS, MODALITY_SEGMENTATION]

UNDO_HISTORY_SIZE = 10

WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 520
SIDEBAR_MIN_WIDTH = 130
SIDEBAR_DEFAULT_WIDTH = 200
JSON_PANEL_MIN_WIDTH = 150
JSON_PANEL_DEFAULT_WIDTH = 220
STATS_PANEL_MIN_HEIGHT = 100
STATUS_BAR_HEIGHT = 22

SPLITTER_SIDEBAR_STRETCH = 1
SPLITTER_CANVAS_STRETCH = 5
SPLITTER_JSON_STRETCH = 2

CANVAS_ZOOM_STEP = 0.1
CANVAS_ZOOM_MIN = 0.1
CANVAS_ZOOM_MAX = 20.0
CANVAS_SCROLL_SENSITIVITY = 1.15
KEYPOINT_SNAP_RADIUS_PX = 8
POLYGON_CLOSE_RADIUS_PX = 10
ARROW_KEY_NUDGE_PX = 1

COLOR_ANNOTATED = "#00FF88"
COLOR_PARTIAL = "#FFD700"
COLOR_UNANNOTATED = "#FF4444"

CANVAS_BORDER_ANNOTATED = COLOR_ANNOTATED
CANVAS_BORDER_UNSAVED = COLOR_PARTIAL
CANVAS_BORDER_UNANNOTATED = COLOR_UNANNOTATED

BBOX_COLOR = "#00CFFF"
KEYPOINT_COLOR = "#FFD700"
SKELETON_COLOR = "#FF8800"
SEGMENTATION_COLOR = "#CC44FF"
SEGMENTATION_FILL_ALPHA = 60

DIFF_OLD_COLOR = "#FF4444"
DIFF_NEW_COLOR = "#00FF88"
DIFF_OLD_ALPHA = 120
DIFF_NEW_ALPHA = 120

STATS_RELOAD_INTERVAL_MS = 750
JSON_RELOAD_INTERVAL_MS = 250

MERGED_DATASET_FORMAT = "{name1}_{name2}_merged_{datetime}"

AUTOANNOTATE_HUMAN_WARNING = (
    "Auto-annotations are currently optimised for human subjects only.\n"
    "A generic multi-class model is coming soon."
)

STATUS_FIELD_ORDER = [
    "filename",
    "dimensions",
    "corrupted",
    "annotated",
    "annotated_by",
    "annotation_date",
]

# ── BBox-overlay handle IDs (used by bbox_overlay.py + canvas.py) ────────────
HANDLE_NONE = -1
HANDLE_MOVE = 0
HANDLE_TL = 1
HANDLE_TC = 2
HANDLE_TR = 3
HANDLE_ML = 4
HANDLE_MR = 5
HANDLE_BL = 6
HANDLE_BC = 7
HANDLE_BR = 8

# ── Dataset-open scenarios (used by validator + main_window) ─────────────────
SCENARIO_EMPTY = "empty"
SCENARIO_IMAGES_ONLY_FLAT = "images_only_flat"
SCENARIO_LABELS_ONLY = "labels_only"
SCENARIO_STRUCTURED_ALL_EMPTY = "structured_all_empty"
SCENARIO_STRUCTURED_LABELS_EMPTY = "structured_labels_empty"
SCENARIO_STRUCTURED_ONE_SPLIT = "structured_one_split"
SCENARIO_OK = "ok"
