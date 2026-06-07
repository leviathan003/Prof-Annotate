# Prof Annotate

A terminal-styled GUI annotation tool for YOLO training data, featuring an arcane mascot (Prof. Annotate) and a rich annotation workflow.

---

## Features

- **Bounding box, keypoint, and segmentation mask** annotation
- **Auto-annotation** via a bundled YOLO11 SegPose ONNX model (human subjects)
- **Dataset wizard** — create, merge, and structure datasets from raw images
- **Bulk keypoint removal** across entire datasets (Ctrl+Del)
- **Git integration** — shows last annotation commit per image
- **Autosave / session recovery** — unsaved work survives crashes
- **Animated mascot** (Prof. Annotate) with guided first-run tutorial
- Adaptive UI for screen sizes from 800×540 (tablet) to 4K desktop

---

## Installation

### From source

```bash
git clone <repo>
cd profannotate
pip install -e .
```

**Requirements:** Python ≥ 3.10, PySide6 ≥ 6.6, onnxruntime ≥ 1.17, numpy ≥ 1.26, opencv-python-headless ≥ 4.9, Pillow ≥ 10.2, PyYAML ≥ 6.0, gitpython ≥ 3.1.

GPU acceleration (optional):
```bash
pip install onnxruntime-gpu
```

### Run

```bash
python main.py
# or, after pip install -e .
bytemark
```

### AppImage (Linux)

```bash
bash build/build_appimage.sh
./ProfAnnotate-x86_64.AppImage
```

---

## Dataset Structure

ByteMark expects YOLO11 format:

```
dataset_root/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`data.yaml` is auto-generated or updated on open. Opening a flat folder of images will prompt you to restructure automatically.

---

## Label File Format

Standard YOLO11 with optional combined fields per line:

```
<class_id> <cx> <cy> <w> <h> [kx ky kv ...] [sx sy ...]
```

- `cx cy w h` — normalized bounding box (0–1)
- `kx ky kv` — keypoint x, y (normalized), visibility (0/1/2) × N keypoints
- `sx sy` — segmentation polygon points (normalized) × M points

---

## Keyboard Shortcuts

| Action | Key |
|---|---|
| Draw BBox | `B` |
| Draw Keypoints | `K` (bbox required) |
| Draw Segmentation | `S` (bbox required) |
| Auto-annotate image | `Ctrl+Shift+A` or `Ctrl+Y` |
| Save annotation | `Ctrl+S` |
| Undo | `Ctrl+Z` |
| Undo seg point | `Ctrl+Z` (in S mode) |
| Close seg polygon | Double-click |
| Skip keypoint | Right-click |
| Next image | `D` or `→` |
| Previous image | `A` or `←` |
| Delete point/instance | `Del` |
| Nudge selected point | Arrow keys |
| Pan canvas | Middle mouse |
| Zoom canvas | Scroll wheel |
| Deselect / exit mode | `Esc` |
| Accept diff preview | `Enter` |
| Reject diff preview | `Esc` |
| Toggle BBox visibility | `Ctrl+1` |
| Toggle Keypoint visibility | `Ctrl+2` |
| Toggle Segmentation visibility | `Ctrl+3` |
| Open folder | `Ctrl+O` |
| Open single image | `Ctrl+F` |
| Create new dataset | `Ctrl+Shift+N` |
| Bulk keypoint removal | `Ctrl+Del` |

---

## Annotation Workflow

### Basic annotation

1. **Open** a dataset folder (`Ctrl+O`) or a single image (`Ctrl+F`).
2. Select an image from the **File Explorer** (left panel).
3. Press `B` → draw a bounding box by click-drag.
4. Press `K` → place keypoints sequentially (right-click to skip, `Ctrl+Z` to undo last).
5. Press `S` → click polygon points; double-click to close the mask.
6. Press `Ctrl+S` to save.

### Auto-annotation

Press `Ctrl+Y` (or `Ctrl+Shift+A`) on any image to run the ONNX model. A diff overlay shows old vs new — press `Enter` to accept or `Esc` to reject.

> **Note:** The current model is optimised for human subjects only.

### Creating a dataset

Click **Create New Dataset** (or `Ctrl+Shift+N`):
1. Select one or more source directories.
2. Choose which labels to carry over.
3. Select keypoints to annotate.
4. Choose auto-annotate or manual.
5. Set train/val split ratio (default 80/20).
6. Confirm — the wizard copies, shuffles, and writes `data.yaml`.

### Bulk keypoint removal

Press `Ctrl+Del` with a dataset open. Check the keypoints to remove, confirm. All label files are rewritten and `data.yaml` is updated. The operation is added to the global undo stack.

---

## UI Panels

| Panel | Description |
|---|---|
| **File Explorer** (left top) | Train/val tree. Green = annotated, yellow = unsaved, red = unannotated. |
| **Dataset Stats** (left bottom) | Live counts: total, train/val split, annotated %, corrupted, per-class distribution. |
| **Canvas** (center) | Image + overlay rendering. Annotation drawing and selection. |
| **Modality Toggles** (above canvas) | Show/hide BBox, Keypoints, Segmentation per image. |
| **data.yaml Editor** (right top) | Editable YAML with syntax highlighting. `*` = unsaved. Ctrl+S to save; autosaves on 2s debounce. |
| **Prof.'s Workshop** (right middle) | Animated mascot. Replaced by "absent" state when a dialog is open. |
| **JSON Editor** (right bottom) | Live JSON view of the selected annotation instance. Directly editable. |
| **Status Bar** (bottom) | Filename, dimensions, corrupted flag, annotated flag, last git commit. |

---

## data.yaml

Auto-generated structure:

```yaml
nc: 1
names: [object]
path: /absolute/path/to/dataset
train: images/train
val: images/val
kpt_shape: [19, 3]

# Keypoint ordering — index → name (positional; do not reorder)
keypoint_names: [nose, left_eye, right_eye, ...]
```

Edit directly in the YAML editor panel. Changes are picked up by the annotation engine immediately after save.

---

## Keypoints

Default skeleton: 19 keypoints (human body).

| Index | Name | Index | Name |
|---|---|---|---|
| 0 | nose | 10 | right_elbow |
| 1 | left_eye | 11 | left_wrist |
| 2 | right_eye | 12 | right_wrist |
| 3 | left_mouth | 13 | left_hip |
| 4 | right_mouth | 14 | right_hip |
| 5 | left_ear | 15 | left_knee |
| 6 | right_ear | 16 | right_knee |
| 7 | left_shoulder | 17 | left_ankle |
| 8 | right_shoulder | 18 | right_ankle |
| 9 | left_elbow | | |

Keypoint subsets can be chosen per-dataset via the wizard or the keypoint selection dialog.

Visibility values: `0` = not labeled, `1` = labeled hidden, `2` = labeled visible.

---

## Session Recovery

Unsaved annotations are written to `~/.profannotate/sessions/` on a 600ms debounce. On next open of the same dataset, the session is restored automatically.

---

## Layout Persistence

Splitter positions and window geometry are saved to `~/.profannotate/layout.json` on close and restored on next launch.

---

## Git Integration

If the dataset root is inside a git repository, the status bar shows the author, date, and hash of the last commit that touched each image's label file (read-only — ByteMark never commits).

---

## Building

### PyInstaller

```bash
pyinstaller build/profannotate.spec
```

### AppImage (Linux)

```bash
bash build/build_appimage.sh
```

Requires: conda environment active, `nuitka`, `patchelf`, `wget`.

---

## Project Structure

```
profannotate/
├── main.py                  # Entry point
├── assets/                  # Fonts, icons, QSS theme
├── models/                  # ONNX model file
├── src/
│   ├── config/
│   │   ├── constants.py     # All app constants
│   │   ├── shortcuts.py     # Keyboard shortcut definitions
│   │   └── skeleton.py      # 19-kpt skeleton definition
│   ├── core/
│   │   ├── annotation/      # models, parser, writer, undo
│   │   ├── dataset/         # loader, validator, wizard worker, yaml_handler
│   │   ├── inference/       # ONNX engine, postprocess, filter
│   │   ├── git/             # read-only git log reader
│   │   └── recovery/        # autosave / session restore
│   └── ui/
│       ├── main_window.py
│       ├── tutorial.py
│       ├── prof_annotate.py # Mascot assets + presence manager
│       ├── prof_watcher.py  # Dialog presence tracking
│       ├── dialogs/         # All popup dialogs
│       ├── drawing/         # BBox, keypoint, segmentation tools
│       ├── overlays/        # QGraphicsItem overlays
│       └── widgets/         # Canvas, file explorer, editors, panels
└── tests/
```

---

## License

GPL v2 — see `LICENSE`.
```
