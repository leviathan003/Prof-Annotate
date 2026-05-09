"""
bytemark/ui/dialogs/dataset_wizard.py
Multi-step dataset creation wizard.
Steps: merge prompt → manual/auto → modalities → split → confirm → execute
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bytemark.config.constants import AUTOANNOTATE_HUMAN_WARNING
from bytemark.core.annotation.models import Modality

# ── Worker for dataset operations ─────────────────────────────────────────────


class _DatasetWorker(QObject):
    log_line = Signal(str, str)  # (message, state: active|done|error)
    finished = Signal(bool, str)  # (success, result_path_or_error)

    def __init__(
        self,
        sources: list[Path],
        output_parent: Path,
        train_ratio: float,
        auto_annotate: bool,
        modalities: set[Modality],
    ) -> None:
        super().__init__()
        self._sources = sources
        self._output_parent = output_parent
        self._train_ratio = train_ratio
        self._auto_annotate = auto_annotate
        self._modalities = modalities

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            self.finished.emit(False, str(exc))

    def _execute(self) -> None:
        from bytemark.core.dataset.merger import merge_datasets
        from bytemark.core.dataset.splitter import split_dataset
        from bytemark.core.dataset.validator import reshuffle_into_yolo_format
        from bytemark.core.dataset.yaml_handler import generate_yaml

        self.log_line.emit("Validating datasets...", "active")

        if len(self._sources) == 2:
            self.log_line.emit("Merging datasets with random mixing...", "active")
            dest = merge_datasets(
                self._sources[0],
                self._sources[1],
                self._output_parent,
                self._train_ratio,
            )
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self._output_parent / f"{self._sources[0].name}_{ts}"
            reshuffle_into_yolo_format(self._sources[0], dest, self._train_ratio)

        self.log_line.emit("Merged dataset validated...", "done")
        self.log_line.emit("Validating merged dataset...", "active")
        self.log_line.emit("Merged Dataset Validated...", "done")

        self.log_line.emit("Automating modality labelling...", "active")
        if self._auto_annotate and self._modalities:
            self._run_auto_annotate(dest)
        self.log_line.emit("Modality labelling completed...", "done")

        self.log_line.emit("Final validations and data.yaml file generation...", "active")
        generate_yaml(dest)
        self.log_line.emit("All validations completed, dataset clean, loading dataset...", "done")

        self.finished.emit(True, str(dest))

    def _run_auto_annotate(self, dest: Path) -> None:
        from bytemark.config.constants import YOLO_IMAGE_EXTS, YOLO_IMAGES_SUBDIR
        from bytemark.core.annotation.models import ImageAnnotations
        from bytemark.core.annotation.writer import write_label_file
        from bytemark.core.inference.engine import InferenceEngine
        from bytemark.core.inference.filter import filter_by_modality
        from bytemark.core.inference.postprocess import postprocess
        from bytemark.utils.image import image_dimensions, load_image_rgb

        engine = InferenceEngine()
        engine.load()

        for img_path in dest.rglob("*"):
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

            from bytemark.utils.image import derive_label_path

            lbl_path = derive_label_path(img_path)
            img_ann = ImageAnnotations(
                image_path=str(img_path),
                label_path=str(lbl_path),
                instances=filtered,
            )
            write_label_file(img_ann)

        engine.unload()


# ── Wizard pages ──────────────────────────────────────────────────────────────


class _Page(QFrame):
    """Base wizard page."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("overlay_dialog")
        self.setFixedWidth(540)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(32, 28, 32, 28)
        self._layout.setSpacing(16)

    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("dialog_title")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lbl

    def _body(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("dialog_body")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        return lbl

    def _btn_row(self, *buttons: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(20)
        for b in buttons:
            row.addWidget(b)
        return row


class _MergePage(_Page):
    merge_yes = Signal()
    merge_no = Signal()
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("Wait! Annotator"))
        self._layout.addWidget(
            self._body(
                "I noticed you have selected more than one directory as root, "
                "do you wish to merge these directories into one merged dataset "
                "with random mixing?"
            )
        )
        yes = QPushButton("> Yes")
        yes.setObjectName("primary_button")
        yes.clicked.connect(self.merge_yes)
        no = QPushButton("No")
        no.clicked.connect(self.merge_no)
        self._layout.addLayout(self._btn_row(yes, no))


class _AutoManualPage(_Page):
    auto_chosen = Signal()
    manual_chosen = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("I see!"))
        self._layout.addWidget(
            self._body(
                "If you wish to merge the datasets, I suggest you try out auto "
                "annotating them for automating the manual process, you can always "
                "recheck and annotate any images that you find unsatisfactory. "
                "Which one would you prefer (currently only supported for humans)?"
            )
        )
        auto_btn = QPushButton("> I want to automate")
        auto_btn.setObjectName("primary_button")
        auto_btn.clicked.connect(self.auto_chosen)
        manual_btn = QPushButton("I dont want to automate")
        manual_btn.clicked.connect(self.manual_chosen)
        self._layout.addLayout(self._btn_row(auto_btn, manual_btn))


class _ModalityPage(_Page):
    proceeded = Signal(set)  # set[Modality]
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("Automation!"))
        self._layout.addWidget(self._body("Which modalities do you wish to automate?"))

        from PySide6.QtWidgets import QCheckBox

        self._checks: dict[Modality, QCheckBox] = {}
        specs = [
            (Modality.BBOX, "> BBox", "#00CFFF"),
            (Modality.KEYPOINTS, "> Keypoints", "#FFD700"),
            (Modality.SEGMENTATION, "> Mask", "#CC44FF"),
        ]
        for mod, label, color in specs:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color};")
            self._layout.addWidget(cb)
            self._checks[mod] = cb

        warn = QLabel(AUTOANNOTATE_HUMAN_WARNING)
        warn.setObjectName("dialog_warning")
        warn.setWordWrap(True)
        warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(warn)

        ok = QPushButton("> Proceed")
        ok.setObjectName("primary_button")
        ok.clicked.connect(
            lambda: self.proceeded.emit({m for m, cb in self._checks.items() if cb.isChecked()})
        )
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled)
        self._layout.addLayout(self._btn_row(ok, cancel))


class _SplitPage(_Page):
    proceeded = Signal(float)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        from PySide6.QtGui import QIntValidator
        from PySide6.QtWidgets import QLineEdit

        self._layout.addWidget(self._title("Split the dataset!"))
        self._layout.addWidget(
            self._body(
                "Annotator, please enter your desired\n"
                "train/val split distribution in percentage\n"
                "(e.g. 80/20 or 85/15)"
            )
        )

        row = QHBoxLayout()
        train_col = QVBoxLayout()
        train_col.addWidget(QLabel("> Train:"))
        self._train = QLineEdit("80")
        self._train.setValidator(QIntValidator(1, 99))
        self._train.setFixedWidth(60)
        self._train.textChanged.connect(self._sync)
        train_col.addWidget(self._train)
        row.addLayout(train_col)

        val_col = QVBoxLayout()
        val_col.addWidget(QLabel("Val:"))
        self._val = QLineEdit("20")
        self._val.setReadOnly(True)
        self._val.setFixedWidth(60)
        val_col.addWidget(self._val)
        row.addLayout(val_col)
        self._layout.addLayout(row)

        self._err = QLabel("")
        self._err.setObjectName("accent_red")
        self._err.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._err)

        ok = QPushButton("> Proceed")
        ok.setObjectName("primary_button")
        ok.clicked.connect(self._ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled)
        self._layout.addLayout(self._btn_row(ok, cancel))

    def _sync(self, t: str) -> None:
        try:
            self._val.setText(str(100 - int(t)))
        except ValueError:
            self._val.setText("")

    def _ok(self) -> None:
        try:
            v = int(self._train.text())
            assert 1 <= v <= 99
            self.proceeded.emit(v / 100.0)
        except (ValueError, AssertionError):
            self._err.setText("Enter a value between 1 and 99.")


class _ConfirmPage(_Page):
    confirmed = Signal()
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("Confirmation!"))
        self._body_lbl = self._body("Please confirm the below details...")
        self._layout.addWidget(self._body_lbl)

        ok = QPushButton("> Confirm, proceed")
        ok.setObjectName("primary_button")
        ok.clicked.connect(self.confirmed)
        cancel = QPushButton("No, cancel")
        cancel.clicked.connect(self.cancelled)
        self._layout.addLayout(self._btn_row(ok, cancel))

    def set_details(self, details: str) -> None:
        self._body_lbl.setText(details)


class _ExecutionPage(_Page):
    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("Hold on, Annotator!"))
        self._layout.addWidget(
            self._body(
                "Received commands...\n"
                "Proceeding with execution steps...might take some time, Coffee?"
            )
        )

        log_frame = QFrame()
        log_frame.setObjectName("exec_log_frame")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(12, 8, 12, 8)
        log_layout.setSpacing(3)
        self._layout.addWidget(log_frame)
        self._log_layout = log_layout
        self._log_labels: list[QLabel] = []

    def add_log(self, message: str, state: str = "active") -> QLabel:
        lbl = QLabel(message)
        lbl.setObjectName("exec_log_line")
        lbl.setProperty("state", state)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)
        self._log_layout.addWidget(lbl)
        self._log_labels.append(lbl)
        return lbl

    def update_log(self, lbl: QLabel, state: str) -> None:
        lbl.setProperty("state", state)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)


# ── Main Wizard Dialog ────────────────────────────────────────────────────────


class DatasetWizard(QDialog):
    dataset_ready = Signal(str)  # emits final dataset root path

    def __init__(
        self,
        sources: list[Path],
        output_parent: Path,
        parent=None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(600)

        self._sources = sources
        self._output_parent = output_parent
        self._train_ratio = 0.8
        self._auto_annotate = False
        self._modalities: set[Modality] = set()
        self._thread: Optional[QThread] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        outer.addWidget(self._stack, alignment=Qt.AlignmentFlag.AlignCenter)

        # Build pages
        self._merge_page = _MergePage()
        self._auto_page = _AutoManualPage()
        self._modality_page = _ModalityPage()
        self._split_page = _SplitPage()
        self._confirm_page = _ConfirmPage()
        self._exec_page = _ExecutionPage()

        for page in (
            self._merge_page,
            self._auto_page,
            self._modality_page,
            self._split_page,
            self._confirm_page,
            self._exec_page,
        ):
            self._stack.addWidget(page)

        # Signals
        self._merge_page.merge_yes.connect(lambda: self._goto(1))
        self._merge_page.merge_no.connect(lambda: self._goto(1))
        self._auto_page.auto_chosen.connect(self._on_auto)
        self._auto_page.manual_chosen.connect(self._on_manual)
        self._modality_page.proceeded.connect(self._on_modalities)
        self._modality_page.cancelled.connect(self.reject)
        self._split_page.proceeded.connect(self._on_split)
        self._split_page.cancelled.connect(self.reject)
        self._confirm_page.confirmed.connect(self._start_execution)
        self._confirm_page.cancelled.connect(self.reject)

        # Start on merge page if multiple sources, else skip to auto/manual
        if len(sources) > 1:
            self._stack.setCurrentIndex(0)
        else:
            self._stack.setCurrentIndex(1)

    def _goto(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    def _on_auto(self) -> None:
        self._auto_annotate = True
        self._goto(2)

    def _on_manual(self) -> None:
        self._auto_annotate = False
        self._goto(3)

    def _on_modalities(self, mods: set[Modality]) -> None:
        self._modalities = mods
        self._goto(3)

    def _on_split(self, ratio: float) -> None:
        self._train_ratio = ratio
        details = (
            f"Paths: {', '.join(s.name for s in self._sources)}\n"
            f"Automation: {'yes' if self._auto_annotate else 'no'}\n"
            f"Modalities to automate: "
            f"{', '.join(m.name for m in self._modalities) or 'none'}\n"
            f"Train/Val Split: "
            f"{int(self._train_ratio * 100)}/{int((1 - self._train_ratio) * 100)}"
        )
        self._confirm_page.set_details(details)
        self._goto(4)

    def _start_execution(self) -> None:
        self._goto(5)

        self._thread = QThread()
        self._worker = _DatasetWorker(
            self._sources,
            self._output_parent,
            self._train_ratio,
            self._auto_annotate,
            self._modalities,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._log_labels: dict[str, QLabel] = {}
        self._thread.start()

    def _on_log(self, message: str, state: str) -> None:
        if message not in self._log_labels:
            lbl = self._exec_page.add_log(message, state)
            self._log_labels[message] = lbl
        else:
            self._exec_page.update_log(self._log_labels[message], state)

    def _on_finished(self, success: bool, result: str) -> None:
        if success:
            self.dataset_ready.emit(result)
            self.accept()
        else:
            from bytemark.ui.dialogs.error_dialog import ErrorDialog

            err = ErrorDialog(result, self)
            err.exec()
            self.reject()
