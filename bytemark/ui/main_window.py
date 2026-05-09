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
    QApplication,
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
    SESSION_CACHE_DIR,
    SIDEBAR_DEFAULT_WIDTH,
    SPLITTER_CANVAS_STRETCH,
    SPLITTER_JSON_STRETCH,
    SPLITTER_SIDEBAR_STRETCH,
)
from bytemark.core.annotation.models import ImageAnnotations, Modality
from bytemark.core.dataset.loader import DatasetIndex, ImageEntry, load_dataset
from bytemark.core.dataset.validator import validate_dataset
from bytemark.core.dataset.yaml_handler import generate_yaml
from bytemark.core.git.reader import (
    find_repo_root,
    get_last_annotation_commit,
    is_git_repo,
)
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
    finished = Signal(object)  # Path
    failed = Signal(str)

    def __init__(self, root: Path, dest: Path) -> None:
        super().__init__()
        self._root = root
        self._dest = dest

    def run(self) -> None:
        try:
            from bytemark.core.dataset.validator import reshuffle_into_yolo_format

            self.finished.emit(reshuffle_into_yolo_format(self._root, self._dest))
        except Exception as exc:
            self.failed.emit(str(exc))


class _DatasetIndexLoader(QObject):
    finished = Signal(object)  # DatasetIndex
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

        # Left: file explorer + stats
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

        # Center: modality selector + canvas
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self._modality_selector = ModalitySelector()
        self._canvas = AnnotationCanvas()
        center_layout.addWidget(self._modality_selector)
        center_layout.addWidget(self._canvas, stretch=1)

        # Right: json editor + yaml editor
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

        self._file_explorer.file_selected.connect(self._on_entry_selected)
        self._file_explorer.open_folder_requested.connect(self._on_open_folder)

        self._canvas.annotations_changed.connect(self._on_annotations_changed)
        self._canvas.save_requested.connect(self._on_save)
        self._canvas.image_loaded.connect(self._on_image_loaded)
        self._canvas.auto_annotate_triggered.connect(self._on_auto_annotate_single)
        self._canvas.navigate_next.connect(self._navigate_next)
        self._canvas.navigate_prev.connect(self._navigate_prev)

        self._modality_selector.modalities_changed.connect(self._canvas.set_visible_modalities)

        self._json_editor.annotation_edited.connect(self._on_json_edited)
        self._yaml_editor.yaml_saved.connect(lambda _: None)

        open_sc = QShortcut(QKeySequence("Ctrl+O"), self)
        open_sc.activated.connect(self._on_open_folder)

        ctrl1 = QShortcut(QKeySequence("Ctrl+1"), self)
        ctrl1.activated.connect(
            lambda: self._modality_selector.set_modality_visible(
                Modality.BBOX, not (Modality.BBOX in self._modality_selector.active_modalities())
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

    def _open_dataset(self, root: Path) -> None:
        result = validate_dataset(root)

        if not result.is_valid:
            from bytemark.ui.dialogs.confirm_dialog import ConfirmDialog

            issues_text = "\n".join(f"• {i}" for i in result.issues)
            dlg = ConfirmDialog(
                "Non-standard Dataset Format",
                f"Issues found:\n{issues_text}\n\nRearrange into YOLO format?",
                "> Yes, rearrange",
                "No, open as-is",
                self,
            )
            if dlg.exec() == dlg.DialogCode.Accepted:
                self._start_reshuffle(root)
            else:
                self._start_index_load(root, flat=True, gen_yaml=False)
            return

        if not result.has_yaml:
            generate_yaml(root)
        self._start_index_load(root, flat=False, gen_yaml=False)

    def _start_reshuffle(self, root: Path) -> None:
        import tempfile

        dest = Path(tempfile.mkdtemp()) / root.name
        self._set_loading(True, "Rearranging dataset into YOLO format...")
        self._bg_thread = QThread()
        self._bg_worker = _ReshuffleWorker(root, dest)
        self._bg_worker.moveToThread(self._bg_thread)
        self._bg_thread.started.connect(self._bg_worker.run)
        self._bg_worker.finished.connect(
            self._on_reshuffle_done, Qt.ConnectionType.QueuedConnection
        )
        self._bg_worker.failed.connect(self._on_bg_error, Qt.ConnectionType.QueuedConnection)
        self._bg_worker.finished.connect(self._bg_thread.quit)
        self._bg_worker.failed.connect(self._bg_thread.quit)
        self._bg_thread.finished.connect(self._bg_thread.deleteLater)
        self._bg_thread.start()

    def _on_reshuffle_done(self, new_root: Path) -> None:
        generate_yaml(new_root)
        self._start_index_load(new_root, flat=False, gen_yaml=False)

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

        recovered = load_session(self._dataset_root)
        if recovered:
            self._dirty_map = recovered

        if index.entries:
            self._load_entry_by_index(0)

    def _on_bg_error(self, msg: str) -> None:
        self._set_loading(False)
        ErrorDialog(msg, self).exec()

    def _set_loading(self, loading: bool, message: str = "") -> None:
        self._canvas.setEnabled(not loading)
        self._file_explorer.setEnabled(not loading)
        if loading:
            self._status_bar.showMessage(f"⟳  {message}")
        else:
            self._status_bar.clearMessage()

    def _on_create_dataset(self) -> None:
        folders = []
        while True:
            folder = QFileDialog.getExistingDirectory(
                self, "Select Dataset Directory", str(Path.home())
            )
            if not folder:
                break
            folders.append(Path(folder))
            from bytemark.ui.dialogs.confirm_dialog import ConfirmDialog

            more = ConfirmDialog(
                "Add another directory?",
                "Do you wish to add another source directory?",
                "> Yes, add more",
                "No, proceed",
                self,
            )
            if more.exec() != more.DialogCode.Accepted:
                break

        if not folders:
            return

        output_parent = Path(
            QFileDialog.getExistingDirectory(self, "Select Output Directory", str(Path.home()))
        )
        if not output_parent:
            return

        wizard = DatasetWizard(folders, output_parent, self)
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

            if self._current_entry is None:
                return
            ann = ImageAnnotations(
                image_path=str(self._current_entry.image_path),
                label_path=str(self._current_entry.label_path),
            )
            for inst_data in data.get("instances", []):
                a = Annotation(class_id=inst_data.get("class", 0))
                if "bbox" in inst_data:
                    b = inst_data["bbox"]
                    a.bbox = BBox(b["cx"], b["cy"], b["w"], b["h"])
                if "keypoints" in inst_data:
                    a.keypoints = [
                        Keypoint(k["x"], k["y"], k.get("v", 2)) for k in inst_data["keypoints"]
                    ]
                if "mask" in inst_data:
                    a.mask = SegmentationMask(
                        points=[tuple(p) for p in inst_data["mask"]["points"]]
                    )
                ann.add_instance(a)
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
            lambda msg: ErrorDialog(msg, self).exec(), Qt.ConnectionType.QueuedConnection
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
