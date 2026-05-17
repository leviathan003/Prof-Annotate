"""
bytemark/ui/main_window.py
Root window — coordinates all panels, dialogs, workers, and signals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from bytemark.config.constants import (
    APP_DOCS_URL,
    APP_NAME,
    APP_VERSION,
    JSON_PANEL_DEFAULT_WIDTH,
    LAYOUT_FILE,
    SIDEBAR_DEFAULT_WIDTH,
    SPLITTER_CANVAS_STRETCH,
    SPLITTER_JSON_STRETCH,
    SPLITTER_SIDEBAR_STRETCH,
    YOLO_IMAGE_EXTS,
    YOLO_TRAIN_DIR,
)
from bytemark.core.annotation.models import ImageAnnotations, Modality
from bytemark.core.dataset.loader import DatasetIndex, ImageEntry, load_dataset
from bytemark.core.dataset.validator import (
    SCENARIO_EMPTY,
    SCENARIO_IMAGES_ONLY_FLAT,
    SCENARIO_LABELS_ONLY,
    SCENARIO_STRUCTURED_ALL_EMPTY,
    SCENARIO_STRUCTURED_LABELS_EMPTY,
    SCENARIO_STRUCTURED_ONE_SPLIT,
    diagnose_dataset,
)
from bytemark.core.dataset.yaml_handler import generate_yaml
from bytemark.core.git.reader import find_repo_root, get_last_annotation_commit, is_git_repo
from bytemark.core.recovery.autosave import clear_session, load_session, save_session
from bytemark.ui.dialogs.autoannotate_prompt import AutoAnnotatePrompt
from bytemark.ui.dialogs.dataset_wizard import DatasetWizard
from bytemark.ui.dialogs.error_dialog import ErrorDialog
from bytemark.ui.widgets.canvas import AnnotationCanvas
from bytemark.ui.widgets.file_explorer import FileExplorer
from bytemark.ui.widgets.json_editor import JsonEditor
from bytemark.ui.widgets.modality_selector import ModalitySelector
from bytemark.ui.widgets.stats_panel import StatsPanel
from bytemark.ui.widgets.status_bar import StatusBar
from bytemark.ui.widgets.yaml_editor import YamlEditor
from bytemark.utils.logger import setup_logging

# ── Background workers ────────────────────────────────────────────────────────


class _ReshuffleWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log_line = Signal(str, str)

    def __init__(self, root: Path, dest: Path) -> None:
        super().__init__()
        self._root = root
        self._dest = dest

    def run(self) -> None:
        try:
            from bytemark.core.dataset.validator import reshuffle_into_yolo_format

            self.log_line.emit("Scanning source directory for images...", "active")
            images = [
                p
                for p in self._root.rglob("*")
                if p.is_file() and p.suffix.lower() in YOLO_IMAGE_EXTS
            ]
            self.log_line.emit(
                f"Found {len(images)} image(s) — preparing output structure...", "done"
            )
            self.log_line.emit("Shuffling and splitting into train / val...", "active")
            result = reshuffle_into_yolo_format(self._root, self._dest)
            self.log_line.emit("Train / val split complete...", "done")
            self.log_line.emit("Generating data.yaml and finalising...", "active")
            generate_yaml(result)
            self.log_line.emit("Dataset ready. Loading now...", "done")
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class _DatasetIndexLoader(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, root: Path, flat: bool = False) -> None:
        super().__init__()
        self._root = root
        self._flat = flat

    def run(self) -> None:
        try:
            if self._flat:
                from bytemark.core.dataset.loader import load_flat_dataset

                self.finished.emit(load_flat_dataset(self._root))
            else:
                self.finished.emit(load_dataset(self._root))
        except Exception as exc:
            self.failed.emit(str(exc))


class _BulkAutoAnnotateWorker(QObject):
    progress = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, root: Path, modalities: set) -> None:
        super().__init__()
        self._root = root
        self._modalities = modalities

    def run(self) -> None:
        try:
            self._execute()
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _execute(self) -> None:
        from bytemark.core.annotation.models import ImageAnnotations
        from bytemark.core.annotation.writer import write_label_file
        from bytemark.core.inference.engine import InferenceEngine
        from bytemark.core.inference.filter import filter_by_modality
        from bytemark.core.inference.postprocess import postprocess
        from bytemark.utils.image import derive_label_path, image_dimensions, load_image_rgb

        engine = InferenceEngine()
        engine.load()
        for img_path in sorted(self._root.rglob("*")):
            if img_path.suffix.lower() not in YOLO_IMAGE_EXTS:
                continue
            rgb = load_image_rgb(img_path)
            if rgb is None:
                continue
            dims = image_dimensions(img_path)
            if dims is None:
                continue
            w, h = dims
            raw = engine.run(rgb)
            anns = postprocess(raw, w, h)
            filtered = filter_by_modality(anns, self._modalities)
            lbl_path = derive_label_path(img_path)
            img_ann = ImageAnnotations(
                image_path=str(img_path),
                label_path=str(lbl_path),
                instances=filtered,
            )
            write_label_file(img_ann)
            self.progress.emit(img_path.name)
        engine.unload()


class _ReshuffleProgressDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        frame.setFixedWidth(520)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(32, 28, 32, 28)
        inner.setSpacing(16)

        title = QLabel("Patience, Annotator.")
        title.setObjectName("dialog_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(title)

        body = QLabel(
            "The reshuffle command has been received and is underway.\n"
            "This may take a moment — the best work is never rushed."
        )
        body.setObjectName("dialog_body")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        inner.addWidget(body)

        log_frame = QFrame()
        log_frame.setObjectName("exec_log_frame")
        self._log_layout = QVBoxLayout(log_frame)
        self._log_layout.setContentsMargins(12, 8, 12, 8)
        self._log_layout.setSpacing(3)
        inner.addWidget(log_frame)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def add_log(self, message: str, state: str = "active") -> QLabel:
        lbl = QLabel(message)
        lbl.setObjectName("exec_log_line")
        lbl.setProperty("state", state)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)
        self._log_layout.addWidget(lbl)
        return lbl

    def update_log(self, lbl: QLabel, state: str) -> None:
        lbl.setProperty("state", state)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)


# ── Main window ───────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        setup_logging()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")

        self._dataset_root: Optional[Path] = None
        self._dataset_index: Optional[DatasetIndex] = None
        self._current_entry: Optional[ImageEntry] = None
        self._current_idx: int = 0
        self._dirty_map: dict[str, ImageAnnotations] = {}
        self._is_git_repo: bool = False
        self._git_root: Optional[Path] = None
        self._pending_gen_yaml: bool = False
        self._bg_thread: Optional[QThread] = None
        self._load_thread: Optional[QThread] = None
        self._dataset_undo_stack: list[dict] = []

        self._build_ui()
        self._connect_signals()
        self._restore_layout()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_top_bar())

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(3)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self._file_explorer = FileExplorer()
        self._stats_panel = StatsPanel()
        self._stats_panel.setMinimumHeight(180)
        left_layout.addWidget(self._file_explorer, stretch=2)
        left_layout.addWidget(self._stats_panel, stretch=1)
        left_widget.setMinimumWidth(160)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self._modality_selector = ModalitySelector()
        self._canvas = AnnotationCanvas()
        center_layout.addWidget(self._modality_selector)
        center_layout.addWidget(self._canvas, stretch=1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self._json_editor = JsonEditor()
        self._yaml_editor = YamlEditor()
        self._yaml_editor.setMaximumHeight(200)
        right_layout.addWidget(self._json_editor, stretch=2)
        right_layout.addWidget(self._yaml_editor, stretch=1)
        right_widget.setMinimumWidth(160)

        self._splitter.addWidget(left_widget)
        self._splitter.addWidget(center_widget)
        self._splitter.addWidget(right_widget)
        self._splitter.setStretchFactor(0, SPLITTER_SIDEBAR_STRETCH)
        self._splitter.setStretchFactor(1, SPLITTER_CANVAS_STRETCH)
        self._splitter.setStretchFactor(2, SPLITTER_JSON_STRETCH)
        self._splitter.setSizes([SIDEBAR_DEFAULT_WIDTH, 999, JSON_PANEL_DEFAULT_WIDTH])

        root_layout.addWidget(self._splitter, stretch=1)

        self._status_bar = StatusBar()
        self.setStatusBar(self._status_bar)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("top_bar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(12)

        self._create_btn = QPushButton("Create New Dataset")
        self._create_btn.setObjectName("create_dataset_btn")
        layout.addWidget(self._create_btn)

        self._kpt_edit_btn = QPushButton("Edit Keypoints")
        self._kpt_edit_btn.setObjectName("kpt_edit_btn")
        self._kpt_edit_btn.setEnabled(False)
        layout.addWidget(self._kpt_edit_btn)

        layout.addStretch()

        title = QLabel(APP_NAME)
        title.setObjectName("app_title")
        layout.addWidget(title)

        layout.addStretch()

        self._help_btn = QPushButton("Help")
        self._help_btn.setObjectName("help_btn")
        layout.addWidget(self._help_btn)

        return bar

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._create_btn.clicked.connect(self._on_create_dataset)
        self._help_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(APP_DOCS_URL)))
        self._kpt_edit_btn.clicked.connect(self._on_bulk_kpt_edit)

        self._file_explorer.file_selected.connect(self._on_entry_selected)
        self._file_explorer.open_folder_requested.connect(self._on_open_folder)

        self._canvas.annotations_changed.connect(self._on_annotations_changed)
        self._canvas.instance_selected.connect(
            lambda ann, idx: self._json_editor.show_instance(ann, idx)
        )
        self._canvas.instance_deselected.connect(self._json_editor.clear_selection)
        self._canvas.save_requested.connect(self._on_save)
        self._canvas.image_loaded.connect(self._on_image_loaded)
        self._canvas.auto_annotate_triggered.connect(self._on_auto_annotate_single)
        self._canvas.navigate_next.connect(self._navigate_next)
        self._canvas.navigate_prev.connect(self._navigate_prev)
        self._canvas.undo_requested.connect(self._on_global_undo)

        self._modality_selector.modalities_changed.connect(self._canvas.set_visible_modalities)
        self._json_editor.annotation_edited.connect(self._on_json_edited)
        self._yaml_editor.yaml_saved.connect(lambda _: None)

        open_sc = QShortcut(QKeySequence("Ctrl+O"), self)
        open_sc.activated.connect(self._on_open_folder)

        save_sc = QShortcut(QKeySequence("Ctrl+S"), self)
        save_sc.activated.connect(
            lambda: self._canvas._save() if self._canvas._annotations else None
        )

        file_sc = QShortcut(QKeySequence("Ctrl+F"), self)
        file_sc.activated.connect(self._on_open_file)

        ctrl1 = QShortcut(QKeySequence("Ctrl+1"), self)
        ctrl1.activated.connect(
            lambda: self._modality_selector.set_modality_visible(
                Modality.BBOX,
                not (Modality.BBOX in self._modality_selector.active_modalities()),
            )
        )
        ctrl2 = QShortcut(QKeySequence("Ctrl+2"), self)
        ctrl2.activated.connect(
            lambda: self._modality_selector.set_modality_visible(
                Modality.KEYPOINTS,
                not (Modality.KEYPOINTS in self._modality_selector.active_modalities()),
            )
        )
        ctrl3 = QShortcut(QKeySequence("Ctrl+3"), self)
        ctrl3.activated.connect(
            lambda: self._modality_selector.set_modality_visible(
                Modality.SEGMENTATION,
                not (Modality.SEGMENTATION in self._modality_selector.active_modalities()),
            )
        )

    # ── Folder / dataset opening ──────────────────────────────────────────────

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Dataset Root", str(Path.home()))
        if not folder:
            return
        self._open_dataset(Path(folder))

    def _on_open_file(self) -> None:
        from bytemark.utils.image import derive_label_path

        exts = " ".join(f"*{e}" for e in sorted(YOLO_IMAGE_EXTS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Single Image", str(Path.home()), f"Images ({exts})"
        )
        if not path:
            return

        from bytemark.core.dataset.loader import DatasetIndex, ImageEntry

        img_path = Path(path)
        lbl_path = derive_label_path(img_path)
        entry = ImageEntry(
            image_path=img_path,
            label_path=lbl_path,
            split=YOLO_TRAIN_DIR,
            is_corrupted=False,
            has_label=lbl_path.exists() and lbl_path.stat().st_size > 0,
        )
        index = DatasetIndex(root=img_path.parent)
        index.entries.append(entry)

        self._dataset_root = img_path.parent
        self._dataset_index = index
        self._is_git_repo = False
        self._git_root = None
        self._dirty_map = {}

        self._file_explorer.load_index(index)
        self._stats_panel.set_index(index)
        self._yaml_editor.load(img_path.parent / "data.yaml")
        self._load_entry_by_index(0)

    def _open_dataset(self, root: Path) -> None:  # noqa: C901
        subdirs = [p.name for p in root.iterdir() if p.is_dir()]
        unexpected = [d for d in subdirs if d not in ("images", "labels")]
        if unexpected:
            ErrorDialog(
                "Annotator, the directory structure here is not what I expected.\n\n"
                f"Unexpected folder(s) found: {', '.join(sorted(unexpected))}\n\n"
                "A proper dataset root must contain only:\n\n"
                "  root/\n"
                "  ├── images/\n"
                "  │     ├── train/\n"
                "  │     └── val/\n"
                "  └── labels/\n"
                "        ├── train/\n"
                "        └── val/\n\n"
                "Please correct the structure and return.",
                self,
            ).exec()
            return

        from bytemark.ui.dialogs.confirm_dialog import ConfirmDialog

        diag = diagnose_dataset(root)

        if diag.scenario == SCENARIO_EMPTY:
            ErrorDialog(
                "The folder appears to be empty, Annotator.\n\n"
                "No images or annotation files were found within. "
                "Perhaps verify the path and try again.",
                self,
            ).exec()
            return

        if diag.scenario == SCENARIO_LABELS_ONLY:
            ErrorDialog(
                "Annotation files are present, but the images are nowhere to be found, Annotator.\n\n"
                "Please select the directory that contains both the images and their labels.",
                self,
            ).exec()
            return

        if diag.scenario == SCENARIO_STRUCTURED_ALL_EMPTY:
            ErrorDialog(
                "The dataset structure exists, but the image directories are empty, Annotator.\n\n"
                "Populate images/train and images/val before proceeding — "
                "there is little to annotate without source material.",
                self,
            ).exec()
            return

        if diag.scenario == SCENARIO_IMAGES_ONLY_FLAT:
            dlg = ConfirmDialog(
                "No Dataset Structure Detected",
                f"I see {diag.flat_image_count} image(s) here, Annotator, "
                f"but no YOLO layout to speak of — no images/train, no images/val.\n\n"
                f"Shall I organise these into a proper dataset structure for you? "
                f"It is, I assure you, the wiser path forward.",
                "> Yes, structure the dataset",
                "No, leave as-is",
                self,
            )
            if dlg.exec() != dlg.DialogCode.Accepted:
                return
            wizard = DatasetWizard([root], root.parent, self)
            wizard.dataset_ready.connect(lambda p: self._open_dataset(Path(p)))
            wizard.exec()
            return

        if diag.scenario == SCENARIO_STRUCTURED_ONE_SPLIT:
            active = diag.active_split
            count = diag.train_image_count if active == YOLO_TRAIN_DIR else diag.val_image_count
            dlg = ConfirmDialog(
                "Incomplete Train / Val Split",
                f"Only the '{active}' split holds images ({count} image(s)), Annotator. "
                f"A single-split dataset is rather lopsided for training.\n\n"
                f"Shall I reshuffle everything into a balanced train/val split? "
                f"Choosing 'No' will open the images as they stand.",
                "> Yes, reshuffle and balance",
                "No, open as-is",
                self,
            )
            if dlg.exec() == dlg.DialogCode.Accepted:
                self._start_reshuffle(root)
            else:
                if not (root / "data.yaml").exists():
                    generate_yaml(root)
                self._start_index_load(root, flat=False, gen_yaml=False)
            return

        if diag.scenario == SCENARIO_STRUCTURED_LABELS_EMPTY:
            dlg = ConfirmDialog(
                "No Annotations Found",
                f"The dataset carries {diag.total_structured_images} image(s) "
                f"but not a single annotation, Annotator. A blank canvas awaits.\n\n"
                f"Shall I run the auto-annotator across all images now? "
                f"You may always refine the results manually afterwards — "
                f"the machine lays the groundwork, the expert perfects it.",
                "> Yes, auto-annotate",
                "No, I will annotate manually",
                self,
            )
            if dlg.exec() == dlg.DialogCode.Accepted:
                self._start_bulk_auto_annotate(root)
            else:
                if not (root / "data.yaml").exists():
                    generate_yaml(root)
                self._start_index_load(root, flat=False, gen_yaml=False)
            return

        if not (root / "data.yaml").exists():
            generate_yaml(root)
        self._start_index_load(root, flat=False, gen_yaml=False)

    def _start_reshuffle(self, root: Path) -> None:
        from datetime import datetime

        dest = root.parent / (root.name + "_reshuffled")
        if dest.exists():
            dest = (
                root.parent / f"{root.name}_reshuffled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        self._reshuffle_dlg = _ReshuffleProgressDialog(self)
        self._reshuffle_log_map: dict[str, QLabel] = {}
        self._reshuffle_dlg.show()

        self._bg_thread = QThread()
        self._bg_worker = _ReshuffleWorker(root, dest)
        self._bg_worker.moveToThread(self._bg_thread)
        self._bg_thread.started.connect(self._bg_worker.run)
        self._bg_worker.log_line.connect(self._on_reshuffle_log, Qt.ConnectionType.QueuedConnection)
        self._bg_worker.finished.connect(
            self._on_reshuffle_done, Qt.ConnectionType.QueuedConnection
        )
        self._bg_worker.failed.connect(self._on_bg_error, Qt.ConnectionType.QueuedConnection)
        self._bg_worker.finished.connect(self._bg_thread.quit)
        self._bg_worker.failed.connect(self._bg_thread.quit)
        self._bg_thread.finished.connect(self._bg_thread.deleteLater)
        self._bg_thread.start()

    def _on_reshuffle_log(self, message: str, state: str) -> None:
        if not hasattr(self, "_reshuffle_dlg") or not hasattr(self, "_reshuffle_log_map"):
            return
        if message not in self._reshuffle_log_map:
            lbl = self._reshuffle_dlg.add_log(message, state)
            self._reshuffle_log_map[message] = lbl
        else:
            self._reshuffle_dlg.update_log(self._reshuffle_log_map[message], state)

    def _on_reshuffle_done(self, new_root: Path) -> None:
        if hasattr(self, "_reshuffle_dlg") and self._reshuffle_dlg is not None:
            self._reshuffle_dlg.accept()
            self._reshuffle_dlg = None
        self._start_index_load(new_root, flat=False, gen_yaml=False)

    def _start_bulk_auto_annotate(self, root: Path) -> None:
        from bytemark.ui.dialogs.modality_prompt import ModalityPrompt

        prompt = ModalityPrompt(
            "Select Modalities to Auto-Annotate", show_warning=True, parent=self
        )
        if prompt.exec() != prompt.DialogCode.Accepted:
            if not (root / "data.yaml").exists():
                generate_yaml(root)
            self._start_index_load(root, flat=False, gen_yaml=False)
            return

        modalities = prompt.selected_modalities()
        if not modalities:
            if not (root / "data.yaml").exists():
                generate_yaml(root)
            self._start_index_load(root, flat=False, gen_yaml=False)
            return

        self._set_loading(True, "Auto-annotating dataset — this may take a while…")
        self._aa_thread = QThread()
        self._aa_worker = _BulkAutoAnnotateWorker(root, modalities)
        self._aa_worker.moveToThread(self._aa_thread)
        self._aa_thread.started.connect(self._aa_worker.run)
        self._aa_worker.progress.connect(
            lambda name: self._status_bar.showMessage(f"⟳  Annotating {name}…"),
            Qt.ConnectionType.QueuedConnection,
        )
        self._aa_worker.finished.connect(
            lambda: self._on_bulk_annotate_done(root),
            Qt.ConnectionType.QueuedConnection,
        )
        self._aa_worker.failed.connect(self._on_bg_error, Qt.ConnectionType.QueuedConnection)
        self._aa_worker.finished.connect(self._aa_thread.quit)
        self._aa_worker.failed.connect(self._aa_thread.quit)
        self._aa_thread.finished.connect(self._aa_thread.deleteLater)
        self._aa_thread.start()

    def _on_bulk_annotate_done(self, root: Path) -> None:
        self._set_loading(False)
        generate_yaml(root)
        self._start_index_load(root, flat=False, gen_yaml=False)

    def _start_index_load(self, root: Path, flat: bool, gen_yaml: bool) -> None:
        self._dataset_root = root
        self._pending_gen_yaml = gen_yaml
        self._set_loading(True, "Indexing dataset...")
        self._load_thread = QThread()
        self._load_worker = _DatasetIndexLoader(root, flat=flat)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(
            self._on_index_loaded, Qt.ConnectionType.QueuedConnection
        )
        self._load_worker.failed.connect(self._on_bg_error, Qt.ConnectionType.QueuedConnection)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.start()

    def _on_index_loaded(self, index: DatasetIndex) -> None:
        self._set_loading(False)
        self._dataset_index = index
        self._file_explorer.load_index(index)
        self._stats_panel.set_index(index)

        yaml_path = self._dataset_root / "data.yaml"
        if self._pending_gen_yaml and not yaml_path.exists():
            generate_yaml(self._dataset_root)
        self._yaml_editor.load(yaml_path)

        self._is_git_repo = is_git_repo(self._dataset_root)
        self._git_root = find_repo_root(self._dataset_root) if self._is_git_repo else None

        self._kpt_edit_btn.setEnabled(True)

        recovered = load_session(self._dataset_root)
        if recovered:
            self._dirty_map = recovered

        if index.entries:
            self._load_entry_by_index(0)

    def _on_bg_error(self, msg: str) -> None:
        self._set_loading(False)
        if hasattr(self, "_reshuffle_dlg") and self._reshuffle_dlg is not None:
            try:
                self._reshuffle_dlg.reject()
            except RuntimeError:
                pass
            self._reshuffle_dlg = None
        ErrorDialog(
            f"Something went wrong, Annotator. The details are as follows:\n\n{msg}",
            self,
        ).exec()

    def _set_loading(self, loading: bool, message: str = "") -> None:
        self._canvas.setEnabled(not loading)
        self._file_explorer.setEnabled(not loading)
        if loading:
            self._status_bar.showMessage(f"⟳  {message}")
        else:
            self._status_bar.clearMessage()

    def _on_create_dataset(self) -> None:
        from bytemark.ui.dialogs.confirm_dialog import ConfirmDialog

        folders: list[Path] = []

        first = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", str(Path.home()))
        if not first:
            return
        folders.append(Path(first))

        while True:
            dlg = ConfirmDialog(
                "Add Another Source?",
                f"You have selected {len(folders)} source(s) so far.\n\n"
                "Shall we include another source directory, Annotator? "
                "Multiple sources will be randomly mixed into a single dataset.",
                "> Yes, add another source",
                "No, proceed with these",
                self,
            )
            if dlg.exec() != dlg.DialogCode.Accepted:
                break
            extra = QFileDialog.getExistingDirectory(
                self, "Select Additional Dataset Directory", str(Path.home())
            )
            if extra:
                folders.append(Path(extra))

        wizard = DatasetWizard(folders, folders[0].parent, self)
        wizard.dataset_ready.connect(lambda p: self._open_dataset(Path(p)))
        wizard.exec()

    # ── Entry navigation ──────────────────────────────────────────────────────

    def _on_entry_selected(self, entry: ImageEntry) -> None:
        try:
            idx = self._dataset_index.entries.index(entry)
            self._load_entry_by_index(idx)
        except (ValueError, AttributeError):
            pass

    def _load_entry_by_index(self, idx: int) -> None:
        if self._dataset_index is None:
            return
        entries = self._dataset_index.entries
        if not entries:
            return
        idx = max(0, min(idx, len(entries) - 1))
        self._current_idx = idx
        self._current_entry = entries[idx]
        self._canvas.set_nav_label(idx + 1, len(entries))
        self._canvas.load_entry(self._current_entry)

    def _navigate_next(self) -> None:
        if self._dataset_index:
            self._load_entry_by_index(self._current_idx + 1)

    def _navigate_prev(self) -> None:
        if self._dataset_index:
            self._load_entry_by_index(self._current_idx - 1)

    # ── Canvas callbacks ──────────────────────────────────────────────────────

    def _on_image_loaded(self, path: str, w: int, h: int, corrupted: bool, annotated: bool) -> None:
        filename = Path(path).name
        self._status_bar.update_image_info(filename, w, h, corrupted, annotated)
        if self._is_git_repo and self._current_entry:
            commit = get_last_annotation_commit(self._current_entry.label_path, self._git_root)
            self._status_bar.update_git_info(commit)
        else:
            self._status_bar.update_git_info(None)

    def _on_annotations_changed(self, ann: ImageAnnotations) -> None:
        self._dirty_map[ann.image_path] = ann
        self._json_editor.set_annotations(ann)
        if self._current_entry:
            self._file_explorer.mark_unsaved(str(self._current_entry.image_path))
        if self._dataset_root:
            save_session(self._dataset_root, self._dirty_map)

    def _on_save(self, ann: ImageAnnotations) -> None:
        self._dirty_map.pop(ann.image_path, None)
        if self._current_entry:
            self._file_explorer.mark_saved(str(self._current_entry.image_path))
        if not self._dirty_map and self._dataset_root:
            clear_session(self._dataset_root)
        self._stats_panel.set_index(self._dataset_index)
        if self._json_editor._selected_idx is not None:
            self._json_editor._refresh_display(force=True)

    # ── Global undo ───────────────────────────────────────────────────────────

    def _on_global_undo(self) -> None:
        if self._dataset_undo_stack:
            originals = self._dataset_undo_stack.pop()
            for path, content in originals.items():
                Path(path).write_text(content, encoding="utf-8")
            if self._dataset_root:
                self._yaml_editor.load(self._dataset_root / "data.yaml")
            if self._current_entry:
                self._canvas.load_entry(self._current_entry)
            self._status_bar.showMessage("Dataset operation undone.")
        else:
            self._canvas._undo_action()

    # ── Bulk keypoint edit ────────────────────────────────────────────────────

    def _on_bulk_kpt_edit(self) -> None:
        if self._dataset_root is None:
            return
        from bytemark.ui.dialogs.kpt_bulk_edit_dialog import KptBulkEditDialog

        dlg = KptBulkEditDialog(self._dataset_root, self)
        dlg.dataset_updated.connect(self._on_bulk_kpt_done)
        dlg.exec()

    def _on_bulk_kpt_done(self, originals: dict) -> None:
        self._dataset_undo_stack.append(originals)
        self._start_index_load(self._dataset_root, flat=False, gen_yaml=False)

    # ── JSON editing ──────────────────────────────────────────────────────────

    def _on_json_edited(self, json_text: str) -> None:
        import json as _json

        try:
            data = _json.loads(json_text)
            from bytemark.core.annotation.models import (
                Annotation,
                BBox,
                ImageAnnotations,
                Keypoint,
                SegmentationMask,
            )

            if self._current_entry is None or self._canvas._annotations is None:
                return
            selected_idx = self._json_editor._selected_idx
            if selected_idx is None:
                return
            ann = self._canvas._annotations
            if selected_idx >= len(ann.instances):
                return

            a = Annotation(class_id=data.get("class", 0))
            if "bbox" in data:
                b = data["bbox"]
                a.bbox = BBox(b["cx"], b["cy"], b["w"], b["h"])
            if "keypoints" in data:
                a.keypoints = [
                    Keypoint(k["x"], k["y"], k.get("v", 2)) if k is not None else None
                    for k in data["keypoints"]
                ]
            if "mask" in data:
                a.mask = SegmentationMask(points=[tuple(p) for p in data["mask"]["points"]])

            ann.instances[selected_idx] = a
            self._canvas.set_annotations(ann)
        except Exception:
            pass

    # ── Auto-annotate single image ────────────────────────────────────────────

    def _on_auto_annotate_single(self) -> None:
        if self._current_entry is None:
            return
        prompt = AutoAnnotatePrompt(self)
        if prompt.exec() != prompt.DialogCode.Accepted:
            return
        modalities = prompt.selected_modalities()
        if not modalities:
            return

        from PySide6.QtCore import QObject, QThread
        from PySide6.QtCore import Signal as Sig

        from bytemark.core.inference.engine import InferenceEngine
        from bytemark.core.inference.filter import filter_by_modality
        from bytemark.core.inference.postprocess import postprocess
        from bytemark.utils.image import image_dimensions, load_image_rgb

        class _Worker(QObject):
            done = Sig(list)
            fail = Sig(str)

            def __init__(self, entry, mods):
                super().__init__()
                self._entry = entry
                self._mods = mods

            def run(self):
                try:
                    rgb = load_image_rgb(self._entry.image_path)
                    dims = image_dimensions(self._entry.image_path)
                    if rgb is None or dims is None:
                        self.fail.emit("Could not load image.")
                        return
                    w, h = dims
                    engine = InferenceEngine()
                    engine.load()
                    raw = engine.run(rgb)
                    anns = postprocess(raw, w, h)
                    filt = filter_by_modality(anns, self._mods)
                    engine.unload()
                    self.done.emit(filt)
                except Exception as e:
                    self.fail.emit(str(e))

        self._ai_thread = QThread()
        self._ai_worker = _Worker(self._current_entry, modalities)
        self._ai_worker.moveToThread(self._ai_thread)
        self._ai_thread.started.connect(self._ai_worker.run)
        self._ai_worker.done.connect(self._on_auto_done, Qt.ConnectionType.QueuedConnection)
        self._ai_worker.fail.connect(
            lambda msg: ErrorDialog(
                f"The auto-annotator encountered a problem, Annotator:\n\n{msg}", self
            ).exec(),
            Qt.ConnectionType.QueuedConnection,
        )
        self._ai_worker.done.connect(self._ai_thread.quit)
        self._ai_worker.fail.connect(self._ai_thread.quit)
        self._ai_thread.finished.connect(self._ai_thread.deleteLater)
        self._ai_thread.start()

    def _on_auto_done(self, new_annotations: list) -> None:
        if self._canvas._annotations is None:
            return
        old = list(self._canvas._annotations.instances)
        self._canvas.show_diff(old, new_annotations)

    # ── Layout persistence ────────────────────────────────────────────────────

    def _restore_layout(self) -> None:
        if LAYOUT_FILE.exists():
            try:
                data = json.loads(LAYOUT_FILE.read_text())
                if "splitter" in data:
                    self._splitter.setSizes(data["splitter"])
                if "geometry" in data:
                    self.restoreGeometry(bytes.fromhex(data["geometry"]))
            except Exception:
                pass

    def _save_layout(self) -> None:
        LAYOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "splitter": self._splitter.sizes(),
            "geometry": self.saveGeometry().toHex().data().decode(),
        }
        LAYOUT_FILE.write_text(json.dumps(data))

    def closeEvent(self, event) -> None:
        if self._dataset_root and self._dirty_map:
            save_session(self._dataset_root, self._dirty_map)
        self._save_layout()
        self._stats_panel.clear()
        super().closeEvent(event)
