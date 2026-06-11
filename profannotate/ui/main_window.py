"""
profannotate/ui/main_window.py
Root window — coordinates all panels, dialogs, workers, and signals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
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

from profannotate.config.constants import (
    APP_DOCS_URL,
    APP_NAME,
    APP_VERSION,
    JSON_PANEL_MIN_WIDTH,
    LAYOUT_FILE,
    SIDEBAR_MIN_WIDTH,
    SPLITTER_CANVAS_STRETCH,
    SPLITTER_JSON_STRETCH,
    SPLITTER_SIDEBAR_STRETCH,
    STATS_PANEL_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    YOLO_IMAGE_EXTS,
    YOLO_TRAIN_DIR,
    YOLO_VAL_DIR,
)
from profannotate.core.annotation.models import ImageAnnotations, Modality
from profannotate.core.dataset.loader import DatasetIndex, ImageEntry, load_dataset
from profannotate.core.dataset.validator import (
    SCENARIO_EMPTY,
    SCENARIO_IMAGES_ONLY_FLAT,
    SCENARIO_LABELS_ONLY,
    SCENARIO_STRUCTURED_ALL_EMPTY,
    SCENARIO_STRUCTURED_LABELS_EMPTY,
    SCENARIO_STRUCTURED_ONE_SPLIT,
    diagnose_dataset,
)
from profannotate.core.dataset.yaml_handler import generate_yaml
from profannotate.core.git.reader import find_repo_root, get_last_annotation_commit, is_git_repo
from profannotate.core.recovery.autosave import clear_session, load_session, save_session
from profannotate.ui.dialogs.autoannotate_prompt import AutoAnnotatePrompt
from profannotate.ui.dialogs.dataset_wizard import DatasetWizard
from profannotate.ui.dialogs.error_dialog import ErrorDialog
from profannotate.ui.widgets.canvas import AnnotationCanvas
from profannotate.ui.widgets.file_explorer import FileExplorer
from profannotate.ui.widgets.json_editor import JsonEditor
from profannotate.ui.widgets.modality_selector import ModalitySelector
from profannotate.ui.widgets.prof_widget import ProfWidget
from profannotate.ui.widgets.stats_panel import StatsPanel
from profannotate.ui.widgets.status_bar import StatusBar
from profannotate.ui.widgets.yaml_editor import YamlEditor
from profannotate.utils.logger import setup_logging
from profannotate.utils.ui_scaling import (
    horizontal_splitter_sizes,
    right_splitter_sizes,
    screen_for,
)

# ── Background workers ────────────────────────────────────────────────────────


class _SingleAutoAnnotateWorker(QObject):
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, entry, modalities: set, active_kpt_names: list[str] | None = None) -> None:
        super().__init__()
        self._entry = entry
        self._modalities = modalities
        self._active_kpt_names = active_kpt_names

    def run(self) -> None:
        try:
            from profannotate.core.inference.engine import InferenceEngine
            from profannotate.core.inference.filter import filter_by_modality
            from profannotate.core.inference.postprocess import postprocess
            from profannotate.ui.dialogs.dataset_wizard import _reindex_keypoints_for_selection
            from profannotate.utils.image import image_dimensions, load_image_rgb

            rgb = load_image_rgb(self._entry.image_path)
            dims = image_dimensions(self._entry.image_path)
            if rgb is None or dims is None:
                self.failed.emit("Could not load image, Annotator.")
                return
            w, h = dims
            engine = InferenceEngine()
            engine.load()
            raw = engine.run(rgb)
            anns = postprocess(raw, w, h)
            filtered = filter_by_modality(anns, self._modalities)
            engine.unload()
            if self._active_kpt_names:
                filtered = _reindex_keypoints_for_selection(filtered, self._active_kpt_names)
            self.done.emit(filtered)
        except Exception as exc:
            self.failed.emit(str(exc))


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
            from profannotate.core.dataset.validator import reshuffle_into_yolo_format

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
    chunk_ready = Signal(object, bool)  # (partial_index, is_final)
    status = Signal(str)
    read_only = Signal()
    failed = Signal(str)

    def __init__(self, root: Path, flat: bool = False) -> None:
        super().__init__()
        self._root = root
        self._flat = flat

    def run(self) -> None:
        try:
            if self._flat:
                from profannotate.core.dataset.loader import load_flat_dataset

                idx = load_flat_dataset(self._root)
                self.chunk_ready.emit(idx, True)
            else:
                from profannotate.core.annotation.writer import (
                    dataset_writable,
                    materialize_empty_labels,
                )
                from profannotate.core.dataset.loader import stream_dataset

                # Guarantee a labels tree + an empty .txt per image before
                # indexing, so the dataset always carries the mapped label
                # files (filled later by auto-annotate or manual saves). On
                # read-only media we skip creation and warn the annotator —
                # viewing still works.
                if dataset_writable(self._root):
                    created = materialize_empty_labels(self._root)
                    if created:
                        self.status.emit(f"Prepared labels folder — {created} new empty file(s).")
                else:
                    self.read_only.emit()

                for partial_index, is_final in stream_dataset(self._root):
                    self.chunk_ready.emit(partial_index, is_final)
        except Exception as exc:
            self.failed.emit(str(exc))


class _BulkAutoAnnotateWorker(QObject):
    progress = Signal(str)
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        root: Path,
        modalities: set,
        active_kpt_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._root = root
        self._modalities = modalities
        self._active_kpt_names = active_kpt_names

    def run(self) -> None:
        try:
            self._execute()
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _execute(self) -> None:
        from profannotate.core.annotation.models import ImageAnnotations
        from profannotate.core.annotation.writer import (
            label_path_for_image,
            materialize_empty_labels,
            write_label_file,
        )
        from profannotate.core.inference.engine import InferenceEngine
        from profannotate.core.inference.filter import filter_by_modality
        from profannotate.core.inference.postprocess import postprocess
        from profannotate.ui.dialogs.dataset_wizard import _reindex_keypoints_for_selection
        from profannotate.utils.image import image_dimensions, load_image_rgb

        # Lay down the labels tree + an empty .txt per image up front, so the
        # folder exists immediately regardless of how many detections follow.
        materialize_empty_labels(self._root)
        self.status.emit("Created labels folder…")

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
            if self._active_kpt_names:
                filtered = _reindex_keypoints_for_selection(filtered, self._active_kpt_names)
            lbl_path = label_path_for_image(self._root, img_path)
            img_ann = ImageAnnotations(
                image_path=str(img_path),
                label_path=str(lbl_path),
                instances=filtered,
            )
            write_label_file(img_ann)
            self.progress.emit(img_path.name)
        engine.unload()


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
        self._stats_panel.setMinimumHeight(STATS_PANEL_MIN_HEIGHT)
        left_layout.addWidget(self._file_explorer, stretch=2)
        left_layout.addWidget(self._stats_panel, stretch=1)
        left_widget.setMinimumWidth(SIDEBAR_MIN_WIDTH)

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
        self._prof_widget = ProfWidget()
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        # Top → bottom: data.yaml · Prof.'s workshop · JSON editor.
        self._right_splitter.addWidget(self._yaml_editor)
        self._right_splitter.addWidget(self._prof_widget)
        self._right_splitter.addWidget(self._json_editor)
        self._right_splitter.setStretchFactor(0, 2)  # YAML
        self._right_splitter.setStretchFactor(1, 2)  # Prof. (stretches with window)
        self._right_splitter.setStretchFactor(2, 5)  # JSON
        self._right_splitter.setCollapsible(0, False)
        self._right_splitter.setCollapsible(1, False)
        self._right_splitter.setCollapsible(2, False)
        # Sizes are applied in `_apply_proportional_sizes` once the window
        # has a concrete geometry (deferred to first showEvent).
        right_layout.addWidget(self._right_splitter)
        right_widget.setMinimumWidth(JSON_PANEL_MIN_WIDTH)

        self._splitter.addWidget(left_widget)
        self._splitter.addWidget(center_widget)
        self._splitter.addWidget(right_widget)
        self._splitter.setStretchFactor(0, SPLITTER_SIDEBAR_STRETCH)
        self._splitter.setStretchFactor(1, SPLITTER_CANVAS_STRETCH)
        self._splitter.setStretchFactor(2, SPLITTER_JSON_STRETCH)
        # Seed with screen-aware widths so the layout is reasonable even
        # before `_apply_proportional_sizes` runs on showEvent.
        screen = screen_for(self)
        seed_w = min(screen.width, max(WINDOW_MIN_WIDTH, screen.width - 80))
        sidebar_w, canvas_w, json_w = horizontal_splitter_sizes(seed_w)
        self._splitter.setSizes([sidebar_w, canvas_w, json_w])
        seed_h = min(screen.height, max(520, screen.height - 80))
        yaml_h, prof_h, json_h = right_splitter_sizes(seed_h)
        self._right_splitter.setSizes([yaml_h, prof_h, json_h])

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

        layout.addStretch()

        title = QLabel(APP_NAME)
        title.setObjectName("app_title")
        layout.addWidget(title)

        layout.addStretch()

        self._tutorial_btn = QPushButton("Tutorial")
        self._tutorial_btn.setObjectName("help_btn")
        layout.addWidget(self._tutorial_btn)

        self._keybindings_btn = QPushButton("Keybindings")
        self._keybindings_btn.setObjectName("help_btn")
        layout.addWidget(self._keybindings_btn)

        self._help_btn = QPushButton("Help")
        self._help_btn.setObjectName("help_btn")
        layout.addWidget(self._help_btn)

        return bar

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._create_btn.clicked.connect(self._on_create_dataset)
        self._tutorial_btn.clicked.connect(self._replay_tutorial)
        self._keybindings_btn.clicked.connect(self._show_keybindings)
        self._help_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(APP_DOCS_URL)))

        # Ctrl+Del — bulk keypoint removal (replaces toolbar button)
        kpt_del_sc = QShortcut(QKeySequence("Ctrl+Del"), self)
        kpt_del_sc.activated.connect(self._on_bulk_kpt_edit)

        new_dataset_sc = QShortcut(QKeySequence("Ctrl+Shift+N"), self)
        new_dataset_sc.activated.connect(self._on_create_dataset)

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
        from profannotate.utils.image import derive_label_path

        exts = " ".join(f"*{e}" for e in sorted(YOLO_IMAGE_EXTS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Single Image", str(Path.home()), f"Images ({exts})"
        )
        if not path:
            return

        from profannotate.core.dataset.loader import DatasetIndex, ImageEntry

        img_path = Path(path)
        lbl_path = derive_label_path(img_path)
        # Detect the split from the file path so val/ files don't land under train/.
        parent_names = {p.name.lower() for p in img_path.parents}
        if YOLO_VAL_DIR in parent_names:
            split = YOLO_VAL_DIR
        else:
            split = YOLO_TRAIN_DIR
        entry = ImageEntry(
            image_path=img_path,
            label_path=lbl_path,
            split=split,
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

    def _open_dataset(self, root: Path, from_wizard: bool = False) -> None:  # noqa: C901
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

        from profannotate.ui.dialogs.confirm_dialog import ConfirmDialog

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
            wizard.dataset_ready.connect(lambda p: self._open_dataset(Path(p), from_wizard=True))
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
            # The wizard already collected kpt + auto/manual choices and may have
            # legitimately produced a labels-free dataset (manual mode). Don't
            # re-prompt — just open it.
            if from_wizard:
                self._start_index_load(root, flat=False, gen_yaml=False)
                return
            # Labels folder is empty for this dataset — always ask the annotator
            # which kpts to use, regardless of whether data.yaml already records
            # a set. Existing names (if any) are preselected in the dialog.
            if not self._ensure_kpt_config_for_root(root, force=True):
                return
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

        self._progress_dlg = self._make_progress_dialog(
            title="Reshuffling the dataset, Annotator.",
            subtitle=(
                "I will balance the splits and seal the new data.yaml. "
                "Watch the runes — they record every step."
            ),
        )
        self._progress_dlg.show()

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
        if getattr(self, "_progress_dlg", None) is None:
            return
        self._progress_dlg.status(message, state)

    def _on_reshuffle_done(self, new_root: Path) -> None:
        self._close_progress_dialog()
        self._start_index_load(new_root, flat=False, gen_yaml=False)

    def _start_bulk_auto_annotate(self, root: Path) -> None:
        from profannotate.ui.dialogs.modality_prompt import ModalityPrompt

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

        self._progress_dlg = self._make_progress_dialog(
            title="Auto-annotating the dataset, Annotator.",
            subtitle=(
                "I shall conjure labels for every image in turn. "
                "This may take a while — patience is the truest craft."
            ),
        )
        self._progress_dlg.add_log("Loading the inference engine…", "active")
        self._progress_dlg.show()

        self._aa_thread = QThread()
        from profannotate.core.dataset.yaml_handler import load_yaml as _load_yaml_for_kpts

        yaml_data = _load_yaml_for_kpts(root)
        bulk_kpt_names = yaml_data.get("keypoint_names")
        if not (isinstance(bulk_kpt_names, list) and bulk_kpt_names):
            bulk_kpt_names = None
        self._aa_worker = _BulkAutoAnnotateWorker(root, modalities, bulk_kpt_names)
        self._aa_worker.moveToThread(self._aa_thread)
        self._aa_thread.started.connect(self._aa_worker.run)
        self._aa_worker.progress.connect(
            self._on_bulk_annotate_progress,
            Qt.ConnectionType.QueuedConnection,
        )
        self._aa_worker.status.connect(
            self._on_index_status,
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

    def _on_bulk_annotate_progress(self, name: str) -> None:
        if getattr(self, "_progress_dlg", None) is None:
            return
        # Mark the previous "loading engine" line as done on the first progress.
        self._progress_dlg.update_log("Loading the inference engine…", "done")
        self._progress_dlg.status(f"Annotating  {name}", "active")

    def _on_bulk_annotate_done(self, root: Path) -> None:
        if getattr(self, "_progress_dlg", None) is not None:
            self._progress_dlg.status("All images annotated — sealing labels…", "done")
        self._close_progress_dialog()
        generate_yaml(root)
        self._start_index_load(root, flat=False, gen_yaml=False)

    def _start_index_load(self, root: Path, flat: bool, gen_yaml: bool) -> None:
        self._dataset_root = root
        self._pending_gen_yaml = gen_yaml
        self._dataset_index = None
        self._current_entry = None
        self._current_idx = 0
        self._dataset_read_only = False

        if getattr(self, "_progress_dlg", None) is None:
            self._progress_dlg = self._make_progress_dialog(
                title="Reading the dataset, Annotator.",
                subtitle="Streaming entries — the index will populate progressively.",
            )
            self._progress_dlg.show()
        self._progress_dlg.status("Indexing dataset…", "active")

        self._load_thread = QThread()
        self._load_worker = _DatasetIndexLoader(root, flat=flat)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.chunk_ready.connect(
            self._on_chunk_ready, Qt.ConnectionType.QueuedConnection
        )
        self._load_worker.status.connect(
            self._on_index_status, Qt.ConnectionType.QueuedConnection
        )
        self._load_worker.read_only.connect(
            self._on_dataset_read_only, Qt.ConnectionType.QueuedConnection
        )
        self._load_worker.failed.connect(self._on_bg_error, Qt.ConnectionType.QueuedConnection)
        self._load_worker.chunk_ready.connect(
            lambda _, is_final: self._load_thread.quit() if is_final else None,
            Qt.ConnectionType.QueuedConnection,
        )
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.start()

    def _on_index_status(self, message: str) -> None:
        if getattr(self, "_progress_dlg", None) is None:
            return
        self._progress_dlg.add_log(message, "done")

    def _on_dataset_read_only(self) -> None:
        # Flag now; surface the message once indexing finishes and the progress
        # dialog has closed (avoids stacking a modal over the progress overlay).
        self._dataset_read_only = True
        if getattr(self, "_progress_dlg", None) is not None:
            self._progress_dlg.add_log("Dataset is read-only — labels can't be saved.", "error")

    def _on_chunk_ready(self, index: "DatasetIndex", is_final: bool) -> None:
        self._dataset_index = index

        if getattr(self, "_progress_dlg", None) is not None:
            self._progress_dlg.status(f"Indexed {index.total} images…", "active")

        if not is_final:
            # First chunk: show UI immediately
            if self._current_entry is None and index.entries:
                self._file_explorer.load_index(index)
                self._stats_panel.set_index(index)
                self._yaml_editor.load(self._dataset_root / "data.yaml")
                self._load_entry_by_index(0)
            elif index.total % 1000 < 100:
                # Refresh every ~1000 entries after that
                self._file_explorer.load_index(index)
                self._stats_panel.set_index(index)
            return

        # Final chunk
        if getattr(self, "_progress_dlg", None) is not None:
            self._progress_dlg.status(f"Indexed {index.total} images…", "done")
        self._close_progress_dialog()

        yaml_path = self._dataset_root / "data.yaml"
        if self._pending_gen_yaml and not yaml_path.exists():
            generate_yaml(self._dataset_root)

        if index.kpt_config_synthesized and index.annotated_count == 0:
            self._prompt_kpt_selection_for_fresh_dataset()

        self._yaml_editor.load(yaml_path)
        self._is_git_repo = is_git_repo(self._dataset_root)
        self._git_root = find_repo_root(self._dataset_root) if self._is_git_repo else None

        recovered = load_session(self._dataset_root)
        if recovered:
            self._dirty_map = recovered

        self._file_explorer.load_index(index)
        self._stats_panel.set_index(index)

        if self._current_entry is None and index.entries:
            self._load_entry_by_index(0)

        if getattr(self, "_dataset_read_only", False):
            self._dataset_read_only = False
            ErrorDialog(
                "This dataset rests on read-only ground, Annotator.\n\n"
                "I cannot conjure label files here, nor will your annotations "
                "persist when saved. You may view and mark up freely, but to "
                "commit your work, copy the dataset to a writable location first.",
                self,
            ).exec()

    def _prompt_kpt_selection_for_fresh_dataset(self) -> None:
        from profannotate.core.dataset.yaml_handler import load_yaml, save_yaml
        from profannotate.ui.dialogs.kpt_selection_dialog import KptSelectionDialog

        dlg = KptSelectionDialog(self, preselected=self._dataset_index.active_keypoint_names)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        names = dlg.selected_names()
        if not names:
            return
        self._dataset_index.active_keypoint_names = names
        self._dataset_index.kpt_config_synthesized = False
        data = load_yaml(self._dataset_root)
        data["kpt_shape"] = [len(names), 3]
        data["keypoint_names"] = names
        save_yaml(self._dataset_root, data)

    def _ensure_kpt_config_for_root(self, root: Path, force: bool = False) -> bool:
        """Make sure data.yaml at `root` has a keypoint_names entry.

        If `force` is True the prompt is always shown (used for empty-labels
        datasets where the annotator must explicitly decide regardless of any
        existing yaml). Otherwise an existing recorded keypoint_names short-
        circuits the dialog.
        Returns False if the user cancels (caller should abort).
        """
        from profannotate.core.dataset.yaml_handler import generate_yaml, load_yaml, save_yaml
        from profannotate.ui.dialogs.kpt_selection_dialog import KptSelectionDialog

        yaml_path = root / "data.yaml"
        existing = load_yaml(root) if yaml_path.exists() else {}
        names = existing.get("keypoint_names")
        if not force and isinstance(names, list) and names:
            return True

        preselected = names if isinstance(names, list) and names else None
        dlg = KptSelectionDialog(self, preselected=preselected)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return False
        chosen = dlg.selected_names()
        if not chosen:
            return False
        if not yaml_path.exists():
            generate_yaml(root)
        data = load_yaml(root)
        data["kpt_shape"] = [len(chosen), 3]
        data["keypoint_names"] = chosen
        save_yaml(root, data)
        return True

    def _on_bg_error(self, msg: str) -> None:
        self._set_loading(False)
        self._close_progress_dialog()
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

    # ── Debounced autosave ───────────────────────────────────────────────────

    def _schedule_autosave(self) -> None:
        """Restart the debounce timer. Actual session write happens 600 ms
        after the last annotation change — so a continuous drag/edit batch
        produces exactly one write instead of dozens."""
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtCore import QTimer

        timer = getattr(self, "_autosave_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setTimerType(_Qt.TimerType.CoarseTimer)
            timer.setInterval(600)
            timer.timeout.connect(self._flush_autosave)
            self._autosave_timer = timer
        timer.start()

    def _flush_autosave(self) -> None:
        if self._dataset_root and self._dirty_map:
            save_session(self._dataset_root, self._dirty_map)

    # ── Prof. progress-dialog helpers ────────────────────────────────────────

    def _make_progress_dialog(self, title: str, subtitle: str):
        from profannotate.ui.dialogs.prof_progress_dialog import ProfProgressDialog

        return ProfProgressDialog(title=title, subtitle=subtitle, parent=self)

    def _close_progress_dialog(self) -> None:
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is None:
            return
        self._progress_dlg = None
        try:
            dlg.accept()
        except RuntimeError:
            pass

    def _show_keybindings(self) -> None:
        from profannotate.ui.dialogs.keybindings_dialog import KeybindingsDialog

        KeybindingsDialog(self).exec()

    def _ensure_tutorial(self) -> "object":
        if getattr(self, "_tutorial", None) is None:
            from profannotate.ui.tutorial import TutorialWalkthrough

            self._tutorial = TutorialWalkthrough(self)
        return self._tutorial

    def maybe_show_first_run_tutorial(self) -> None:
        """Public entry-point called from main.py once the splash dismisses.
        Shows the tutorial only if the annotator hasn't seen it yet."""
        from profannotate.utils.prefs import get_pref

        if get_pref("tutorial_seen", False):
            return
        self._ensure_tutorial().start()

    def _replay_tutorial(self) -> None:
        """Triggered by the 'Tutorial' top-bar button."""
        self._ensure_tutorial().start()

    def _on_create_dataset(self) -> None:
        from profannotate.ui.dialogs.confirm_dialog import ConfirmDialog

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
        wizard.dataset_ready.connect(lambda p: self._open_dataset(Path(p), from_wizard=True))
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
        if (
            idx == self._current_idx
            and self._current_entry is not None
            and self._current_entry is entries[idx]
        ):
            return
        self._autosave_current()
        self._current_idx = idx
        self._current_entry = entries[idx]
        self._canvas.set_nav_label(idx + 1, len(entries))
        nk = self._dataset_index.num_keypoints
        names = (
            self._dataset_index.active_keypoint_names
            if self._dataset_index.active_keypoint_names
            else None
        )
        self._canvas.load_entry(self._current_entry, num_keypoints=nk, active_kpt_names=names)
        self._schedule_prefetch_window()

    def _entry_pos_in_split(self, entry: ImageEntry) -> tuple[list[ImageEntry], int] | None:
        """Return (split_entries_list, position_within_split) for `entry`."""
        if self._dataset_index is None:
            return None
        split_list = (
            self._dataset_index.train_entries
            if entry.split == YOLO_TRAIN_DIR
            else self._dataset_index.val_entries
        )
        if not split_list:
            return None
        try:
            return split_list, split_list.index(entry)
        except ValueError:
            return None

    def _step_within_split(self, delta: int) -> None:
        if self._dataset_index is None or self._current_entry is None:
            return
        info = self._entry_pos_in_split(self._current_entry)
        if info is None:
            return
        split_list, pos = info
        n = len(split_list)
        next_entry = split_list[(pos + delta) % n]
        try:
            global_idx = self._dataset_index.entries.index(next_entry)
        except ValueError:
            return
        self._load_entry_by_index(global_idx)

    def _navigate_next(self) -> None:
        self._step_within_split(1)

    def _navigate_prev(self) -> None:
        self._step_within_split(-1)

    def _schedule_prefetch_window(self, half: int = 20) -> None:
        """Tell the canvas to load the ±`half` neighbours within the current
        split into its RGB cache. Wraps around at the split boundaries so a
        sliding window keeps following the user's position."""
        if self._dataset_index is None or self._current_entry is None:
            return
        info = self._entry_pos_in_split(self._current_entry)
        if info is None:
            return
        split_list, pos = info
        n = len(split_list)
        paths: list[str] = []
        for d in range(1, half + 1):
            paths.append(str(split_list[(pos + d) % n].image_path))
            paths.append(str(split_list[(pos - d) % n].image_path))
        self._canvas.prefetch_paths(paths)

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
            # Debounce: the previous code called save_session on every
            # annotation tick (e.g. every mouse-move while dragging a kpt),
            # which JSON-serialised + wrote the entire dirty map each time.
            # Defer the actual write so we only do it once the user pauses.
            self._schedule_autosave()

    def _on_save(self, ann: ImageAnnotations) -> None:
        has_label = len(ann.instances) > 0
        self._dirty_map.pop(ann.image_path, None)
        if self._current_entry:
            self._file_explorer.mark_saved(str(self._current_entry.image_path), has_label=has_label)
        if not self._dirty_map and self._dataset_root:
            clear_session(self._dataset_root)
        self._stats_panel.set_index(self._dataset_index)
        if self._json_editor._selected_idx is not None:
            self._json_editor._refresh_display(force=True)

    # ── AutoSave Helper ───────────────────────────────────────────────────────

    def _autosave_current(self) -> None:
        if self._current_entry is None:
            return
        if self._canvas._annotations is None:
            return
        if not self._canvas._dirty:
            return
        from profannotate.core.annotation.writer import write_label_file

        ann = self._canvas._annotations
        write_label_file(ann)
        self._canvas._dirty = False
        self._canvas._undo.clear()
        self._canvas._update_border()
        has_label = len(ann.instances) > 0
        self._file_explorer.mark_saved(str(self._current_entry.image_path), has_label=has_label)
        self._dirty_map.pop(ann.image_path, None)
        if not self._dirty_map and self._dataset_root:
            clear_session(self._dataset_root)
        self._stats_panel.set_index(self._dataset_index)
        self._status_bar.showMessage(f"✓  Auto-saved {self._current_entry.image_path.name}", 2000)

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

    # ── Bulk keypoint removal (Ctrl+Del) ──────────────────────────────────────

    def _on_bulk_kpt_edit(self) -> None:
        if self._dataset_root is None:
            self._status_bar.showMessage(
                "No dataset loaded — open a dataset before using bulk keypoint removal.", 3000
            )
            return
        from profannotate.ui.dialogs.kpt_bulk_edit_dialog import KptBulkEditDialog

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
            from profannotate.core.annotation.models import (
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

        img_name = self._current_entry.image_path.name
        self._progress_dlg = self._make_progress_dialog(
            title="Auto-annotating this image, Annotator.",
            subtitle=(
                "I shall examine the pixels and propose annotations. "
                "You may accept or reject the result when I am done."
            ),
        )
        self._progress_dlg.add_log("Loading the inference engine…", "active")
        self._progress_dlg.add_log(f"Examining  {img_name}", "active")
        self._progress_dlg.show()

        self._ai_thread = QThread()
        active_kpt_names = (
            self._dataset_index.active_keypoint_names
            if self._dataset_index and self._dataset_index.active_keypoint_names
            else None
        )
        self._ai_worker = _SingleAutoAnnotateWorker(
            self._current_entry, modalities, active_kpt_names
        )
        self._ai_worker.moveToThread(self._ai_thread)
        self._ai_thread.started.connect(self._ai_worker.run)
        self._ai_worker.done.connect(self._on_auto_done, Qt.ConnectionType.QueuedConnection)
        self._ai_worker.failed.connect(
            self._on_auto_annotate_failed, Qt.ConnectionType.QueuedConnection
        )
        self._ai_worker.done.connect(self._ai_thread.quit)
        self._ai_worker.failed.connect(self._ai_thread.quit)
        self._ai_thread.finished.connect(self._ai_thread.deleteLater)
        self._ai_thread.start()

    def _on_auto_annotate_failed(self, msg: str) -> None:
        self._close_progress_dialog()
        ErrorDialog(f"The auto-annotator encountered a problem, Annotator:\n\n{msg}", self).exec()

    def _on_auto_done(self, new_annotations: list) -> None:
        if getattr(self, "_progress_dlg", None) is not None:
            self._progress_dlg.status("Loading the inference engine…", "done")
            if self._current_entry is not None:
                self._progress_dlg.status(
                    f"Examining  {self._current_entry.image_path.name}", "done"
                )
        self._close_progress_dialog()
        if self._canvas._annotations is None:
            return
        old = list(self._canvas._annotations.instances)
        self._canvas.show_diff(old, new_annotations)

    # ── Layout persistence ────────────────────────────────────────────────────

    def _restore_layout(self) -> None:
        self._layout_restored_from_disk = False
        if LAYOUT_FILE.exists():
            try:
                data = json.loads(LAYOUT_FILE.read_text())
                if "splitter" in data:
                    self._splitter.setSizes(data["splitter"])
                    self._layout_restored_from_disk = True
                if "right_splitter" in data:
                    self._right_splitter.setSizes(data["right_splitter"])
                if "geometry" in data:
                    self.restoreGeometry(bytes.fromhex(data["geometry"]))
            except Exception:
                pass

    def _apply_proportional_sizes(self) -> None:
        """Recompute splitter sizes from the actual window size. Skipped if
        the user had saved layout from a previous session — we honour their
        last-known split instead of overwriting it."""
        if getattr(self, "_layout_restored_from_disk", False):
            return
        sidebar_w, canvas_w, json_w = horizontal_splitter_sizes(self.width())
        self._splitter.setSizes([sidebar_w, canvas_w, json_w])
        yaml_h, prof_h, json_h = right_splitter_sizes(self.height())
        self._right_splitter.setSizes([yaml_h, prof_h, json_h])

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        # Only run on the first real show — subsequent shows shouldn't
        # forcibly re-divide the splitters the user has dragged.
        if not getattr(self, "_first_show_done", False):
            self._first_show_done = True
            self._apply_proportional_sizes()

    def _save_layout(self) -> None:
        LAYOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "splitter": self._splitter.sizes(),
            "right_splitter": self._right_splitter.sizes(),
            "geometry": self.saveGeometry().toHex().data().decode(),
        }
        LAYOUT_FILE.write_text(json.dumps(data))

    def closeEvent(self, event) -> None:
        # Cancel any pending debounced autosave — we're about to write
        # synchronously below.
        timer = getattr(self, "_autosave_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._autosave_current()
        # Flush any pending data.yaml edits the user hasn't Ctrl+S'd.
        self._yaml_editor.flush_if_dirty()
        if self._dataset_root and self._dirty_map:
            save_session(self._dataset_root, self._dirty_map)
        self._save_layout()
        self._stats_panel.clear()
        super().closeEvent(event)
