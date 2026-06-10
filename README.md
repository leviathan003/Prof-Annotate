```
# Prof Annotate

A terminal-styled GUI annotation tool for YOLO training data, featuring an arcane mascot (Prof. Annotate) and a rich annotation workflow.

---

## Requirements

- Python >= 3.10
- pip >= 23
- Git
- Linux (AppImage build), Windows, or macOS
- (Optional) NVIDIA GPU + CUDA 11.8+ for GPU-accelerated inference

---

## Quick Start

```bash
git clone <repo-url> profannotate
cd profannotate
bash scripts/setup.sh
```

Then launch:

```bash
source .venv/bin/activate
python main.py

# or if installed as a CLI command
bytemark
```

---

## Manual Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install dependencies

CPU-only (default):
```bash
pip install -e .
```

GPU (CUDA) — replaces the CPU onnxruntime:
```bash
pip install -e .
pip uninstall -y onnxruntime
pip install onnxruntime-gpu>=1.17.0
```

### 4. Place the ONNX model

Download or copy yolo11n_segpose.onnx into the models/ directory:

```
profannotate/
└── models/
    └── yolo11n_segpose.onnx
```

The application will start without it, but auto-annotation will not work.

### 5. Run

```bash
python main.py
# or
bytemark
```

---

## Development Setup

```bash
pip install -e ".[dev]"
```

Installs pytest, pytest-qt, black, and ruff.

Run tests:
```bash
pytest
```

Format / lint:
```bash
black .
ruff check .
```

---

## Dataset Structure

Prof Annotate expects YOLO11 format:

```
dataset_root/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

data.yaml is auto-generated on open. Opening a flat folder of images will prompt you to restructure automatically.

---

## Label File Format

```
<class_id> <cx> <cy> <w> <h> [kx ky kv ...] [sx sy ...]
```

- cx cy w h   — normalized bounding box (0-1)
- kx ky kv    — keypoint x, y (normalized), visibility (0/1/2) x N keypoints
- sx sy       — segmentation polygon points (normalized) x M points

---

## Keyboard Shortcuts

Action                          Key
-------                         ---
Draw BBox                       B
Draw Keypoints                  K (bbox required)
Draw Segmentation               S (bbox required)
Auto-annotate image             Ctrl+Shift+A or Ctrl+Y
Save annotation                 Ctrl+S
Undo                            Ctrl+Z
Undo seg point                  Ctrl+Z (in S mode)
Close seg polygon               Double-click
Skip keypoint                   Right-click
Next image                      D or ->
Previous image                  A or <-
Delete point / instance         Del
Nudge selected point            Arrow keys
Pan canvas                      Middle mouse
Zoom canvas                     Scroll wheel
Deselect / exit mode            Esc
Accept diff preview             Enter
Reject diff preview             Esc
Toggle BBox visibility          Ctrl+1
Toggle Keypoint visibility      Ctrl+2
Toggle Segmentation visibility  Ctrl+3
Open folder                     Ctrl+O
Open single image               Ctrl+F
Create new dataset              Ctrl+Shift+N
Bulk keypoint removal           Ctrl+Del

---

## Annotation Workflow

### Basic annotation

1. Open a dataset folder (Ctrl+O) or single image (Ctrl+F).
2. Select an image from the File Explorer (left panel).
3. Press B -> click-drag to draw a bounding box.
4. Press K -> place keypoints sequentially. Right-click to skip; Ctrl+Z to undo last point.
5. Press S -> click polygon points; double-click to close.
6. Press Ctrl+S to save.

### Auto-annotation

Press Ctrl+Y on any image. A diff overlay shows old vs new annotations.
Press Enter to accept or Esc to reject.

Note: The current model is optimised for human subjects only.

### Creating a dataset

Click Create New Dataset (Ctrl+Shift+N):
1. Select one or more source directories.
2. Choose which labels to carry over (multi-source merge).
3. Select keypoints to include.
4. Choose auto-annotate or manual.
5. Set train/val split ratio (default 80/20).
6. Confirm — files are copied, shuffled, and data.yaml is written.

### Bulk keypoint removal

Press Ctrl+Del with a dataset open. Check the keypoints to remove and confirm.
All label files are rewritten, data.yaml is updated, and the operation is
pushed onto the undo stack.

---

## UI Panels

Panel                           Description
-----                           -----------
File Explorer (left top)        Train/val tree. Green = annotated, yellow = unsaved, red = unannotated.
Dataset Stats (left bottom)     Live counts: total, train/val split, annotated %, corrupted, per-class.
Canvas (center)                 Image + overlay rendering, drawing, and selection.
Modality Toggles (above canvas) Show/hide BBox, Keypoints, Segmentation independently.
data.yaml Editor (right top)    Editable YAML with syntax highlighting. * = unsaved. Autosaves on 2s debounce.
Prof.'s Workshop (right middle) Animated mascot. Shows absent state while any dialog is open.
JSON Editor (right bottom)      Live JSON of the selected annotation instance. Directly editable.
Status Bar (bottom)             Filename, dimensions, corrupted flag, annotated flag, last git commit.

---

## data.yaml

Auto-generated structure:

```
nc: 1
names: [object]
path: /absolute/path/to/dataset
train: images/train
val: images/val
kpt_shape: [19, 3]

# Keypoint ordering — index -> name (positional; do not reorder)
keypoint_names: [nose, left_eye, right_eye, ...]
```

---

## Keypoints

Default: 19-keypoint human body skeleton.

Index   Name                Index   Name
-----   ----                -----   ----
0       nose                10      right_elbow
1       left_eye            11      left_wrist
2       right_eye           12      right_wrist
3       left_mouth          13      left_hip
4       right_mouth         14      right_hip
5       left_ear            15      left_knee
6       right_ear           16      right_knee
7       left_shoulder       17      left_ankle
8       right_shoulder      18      right_ankle
9       left_elbow

Visibility: 0 = not labeled, 1 = labeled hidden, 2 = labeled visible.

Keypoint subsets can be chosen per-dataset via the wizard or the keypoint selection dialog.

---

## Session Recovery

Unsaved annotations are debounce-written to ~/.profannotate/sessions/.
On next open of the same dataset the session is restored automatically.

---

## Layout Persistence

Splitter positions and window geometry are saved to ~/.profannotate/layout.json on close.

---

## Git Integration

If the dataset is inside a git repo, the status bar shows the author, date, and hash
of the last commit touching each label file (read-only — Prof Annotate never commits).

---

## Building

### PyInstaller

```bash
pip install pyinstaller
pyinstaller build/profannotate.spec
```

### AppImage (Linux)

```bash
conda activate <your_env>
bash build/build_appimage.sh
```

Requires: nuitka, patchelf, wget installed in the environment.

---

## Project Structure

```
profannotate/
├── main.py
├── scripts/
│   └── setup.sh
├── assets/                 Fonts, icons, QSS theme
├── models/                 ONNX model (yolo11n_segpose.onnx)
├── src/
│   ├── config/             constants, shortcuts, skeleton
│   ├── core/
│   │   ├── annotation/     models, parser, writer, undo
│   │   ├── dataset/        loader, validator, yaml_handler, wizard worker
│   │   ├── inference/      ONNX engine, postprocess, filter
│   │   ├── git/            read-only git log reader
│   │   └── recovery/       autosave / session restore
│   └── ui/
│       ├── main_window.py
│       ├── tutorial.py
│       ├── prof_annotate.py
│       ├── prof_watcher.py
│       ├── dialogs/
│       ├── drawing/
│       ├── overlays/
│       └── widgets/
└── tests/
```

---

## License

GPL v2 — see LICENSE.
```
