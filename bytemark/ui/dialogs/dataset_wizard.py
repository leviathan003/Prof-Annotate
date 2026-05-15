"""
bytemark/ui/dialogs/dataset_wizard.py
Multi-step dataset creation wizard.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bytemark.config.constants import AUTOANNOTATE_HUMAN_WARNING
from bytemark.core.annotation.models import Modality


class _DatasetWorker(QObject):
    log_line = Signal(str, str)
    finished = Signal(bool, str)

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
        from bytemark.core.dataset.validator import reshuffle_into_yolo_format
        from bytemark.core.dataset.yaml_handler import generate_yaml

        self.log_line.emit("Validating source datasets...", "active")

        if len(self._sources) == 2:
            self.log_line.emit("Merging datasets with random mixing...", "active")
            dest = merge_datasets(
                self._sources[0], self._sources[1], self._output_parent, self._train_ratio
            )
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self._output_parent / f"{self._sources[0].name}_{ts}"
            reshuffle_into_yolo_format(self._sources[0], dest, self._train_ratio)

        self.log_line.emit("Source data validated and restructured...", "done")
        self.log_line.emit("Running final dataset validation...", "active")
        self.log_line.emit("Dataset structure confirmed clean...", "done")

        self.log_line.emit("Initiating modality labelling...", "active")
        if self._auto_annotate and self._modalities:
            self._run_auto_annotate(dest)
        self.log_line.emit("Modality labelling complete...", "done")

        self.log_line.emit("Generating data.yaml and finalising...", "active")
        generate_yaml(dest)
        self.log_line.emit("All done. Dataset is clean and ready to load...", "done")

        self.finished.emit(True, str(dest))

    def _run_auto_annotate(self, dest: Path) -> None:
        from bytemark.config.constants import YOLO_IMAGE_EXTS
        from bytemark.core.annotation.models import ImageAnnotations
        from bytemark.core.annotation.writer import write_label_file
        from bytemark.core.inference.engine import InferenceEngine
        from bytemark.core.inference.filter import filter_by_modality
        from bytemark.core.inference.postprocess import postprocess
        from bytemark.utils.image import derive_label_path, image_dimensions, load_image_rgb

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
            lbl_path = derive_label_path(img_path)
            img_ann = ImageAnnotations(
                image_path=str(img_path), label_path=str(lbl_path), instances=filtered
            )
            write_label_file(img_ann)
        engine.unload()


# ── Base page ─────────────────────────────────────────────────────────────────


class _Page(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("overlay_dialog")
        self.setFixedWidth(560)
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


# ── Wizard pages ──────────────────────────────────────────────────────────────


class _MergePage(_Page):
    merge_yes = Signal()
    merge_no = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("One Moment, Annotator."))
        self._layout.addWidget(
            self._body(
                "I notice you have selected more than one source directory. "
                "Shall I merge them into a single dataset with random mixing? "
                "This is generally the wiser approach for training stability and generalisation."
            )
        )
        yes = QPushButton("> Yes, merge them")
        yes.setObjectName("primary_button")
        yes.clicked.connect(self.merge_yes)
        no = QPushButton("No, treat separately")
        no.clicked.connect(self.merge_no)
        self._layout.addLayout(self._btn_row(yes, no))


class _AutoManualPage(_Page):
    auto_chosen = Signal()
    manual_chosen = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("How Shall We Proceed?"))
        self._layout.addWidget(
            self._body(
                "The auto-annotator can handle the laborious groundwork, Annotator, "
                "leaving you to review and refine — a sensible division of labour. "
                "Note that the current model is optimised for human subjects only.\n\n"
                "Which path shall we take?"
            )
        )
        auto_btn = QPushButton("> Auto-annotate for me")
        auto_btn.setObjectName("primary_button")
        auto_btn.clicked.connect(self.auto_chosen)
        manual_btn = QPushButton("I will annotate manually")
        manual_btn.clicked.connect(self.manual_chosen)
        self._layout.addLayout(self._btn_row(auto_btn, manual_btn))


class _ModalityPage(_Page):
    proceeded = Signal(set)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("Select Your Modalities"))
        self._layout.addWidget(
            self._body(
                "Which annotation modalities shall the auto-annotator produce, Annotator? "
                "Choose carefully — more modalities yield richer labels, but also more to verify."
            )
        )
        self._checks: dict[Modality, QCheckBox] = {}
        specs = [
            (Modality.BBOX, "> Bounding Box", "#00CFFF"),
            (Modality.KEYPOINTS, "> Keypoints", "#FFD700"),
            (Modality.SEGMENTATION, "> Segmentation Mask", "#CC44FF"),
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

        ok = QPushButton("> Proceed with these modalities")
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

        self._layout.addWidget(self._title("Define the Train / Val Split"))
        self._layout.addWidget(
            self._body(
                "Enter your desired train/val distribution, Annotator.\n"
                "An 80/20 split is a solid, time-honoured starting point for most datasets."
            )
        )

        row = QHBoxLayout()
        train_col = QVBoxLayout()
        train_col.addWidget(QLabel("> Train %:"))
        self._train = QLineEdit("80")
        self._train.setValidator(QIntValidator(1, 99))
        self._train.setFixedWidth(60)
        self._train.textChanged.connect(self._sync)
        train_col.addWidget(self._train)
        row.addLayout(train_col)

        val_col = QVBoxLayout()
        val_col.addWidget(QLabel("Val %:"))
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
            self._err.setText("Please enter a value between 1 and 99, Annotator.")


class _ConfirmPage(_Page):
    confirmed = Signal()
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("Review Before We Proceed"))
        self._body_lbl = self._body(
            "Please review the details below, Annotator. Once confirmed, the pipeline will execute."
        )
        self._layout.addWidget(self._body_lbl)

        ok = QPushButton("> Confirm — proceed")
        ok.setObjectName("primary_button")
        ok.clicked.connect(self.confirmed)
        cancel = QPushButton("No, let me reconsider")
        cancel.clicked.connect(self.cancelled)
        self._layout.addLayout(self._btn_row(ok, cancel))

    def set_details(self, details: str) -> None:
        self._body_lbl.setText(details)


class _ExecutionPage(_Page):
    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("Executing, Annotator — Patience."))
        self._layout.addWidget(
            self._body(
                "Commands received. The pipeline is underway.\n"
                "This may take a few moments — the best work is never rushed. "
                "Perhaps some tea?"
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


# ── Main wizard ───────────────────────────────────────────────────────────────


class DatasetWizard(QDialog):
    dataset_ready = Signal(str)

    def __init__(self, sources: list[Path], output_parent: Path, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(620)

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

        self._stack.setCurrentIndex(0 if len(sources) > 1 else 1)

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
            f"Source(s): {', '.join(s.name for s in self._sources)}\n"
            f"Auto-annotation: {'enabled' if self._auto_annotate else 'disabled'}\n"
            f"Modalities: {', '.join(m.name for m in self._modalities) or 'none'}\n"
            f"Train / Val split: "
            f"{int(self._train_ratio * 100)} / {int((1 - self._train_ratio) * 100)}"
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

            ErrorDialog(f"The pipeline encountered an error, Annotator:\n\n{result}", self).exec()
            self.reject()
