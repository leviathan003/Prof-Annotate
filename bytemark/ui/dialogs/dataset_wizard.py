"""
bytemark/ui/dialogs/dataset_wizard.py
Multi-step dataset creation wizard — single or multi-source.
"""

from __future__ import annotations

import random
import shutil
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
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bytemark.config.constants import AUTOANNOTATE_HUMAN_WARNING
from bytemark.core.annotation.models import Modality

# ── Source analysis ───────────────────────────────────────────────────────────


def _analyze_source(path: Path) -> dict:
    from bytemark.config.constants import (
        YOLO_IMAGE_EXTS,
        YOLO_IMAGES_SUBDIR,
        YOLO_LABEL_EXT,
        YOLO_LABELS_SUBDIR,
        YOLO_TRAIN_DIR,
        YOLO_VAL_DIR,
    )

    images_dir = path / YOLO_IMAGES_SUBDIR
    labels_dir = path / YOLO_LABELS_SUBDIR
    is_structured = images_dir.exists() and (
        (images_dir / YOLO_TRAIN_DIR).exists() or (images_dir / YOLO_VAL_DIR).exists()
    )
    image_count = sum(
        1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in YOLO_IMAGE_EXTS
    )
    if labels_dir.exists():
        label_count = sum(1 for p in labels_dir.rglob(f"*{YOLO_LABEL_EXT}") if p.is_file())
    else:
        label_count = sum(1 for p in path.rglob(f"*{YOLO_LABEL_EXT}") if p.is_file())
    return {
        "path": path,
        "name": path.name,
        "is_structured": is_structured,
        "has_labels": label_count > 0,
        "image_count": image_count,
        "label_count": label_count,
    }


def _make_dest_name(sources: list[Path]) -> str:
    names = "_".join(s.name for s in sources)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{names}_{ts}"


# ── Worker ────────────────────────────────────────────────────────────────────


class _DatasetWorker(QObject):
    log_line = Signal(str, str)
    finished = Signal(bool, str)

    def __init__(
        self,
        sources: list[Path],
        source_infos: list[dict],
        label_decisions: dict,
        output_parent: Path,
        train_ratio: float,
        auto_annotate: bool,
        modalities: set,
    ) -> None:
        super().__init__()
        self._sources = sources
        self._source_infos = source_infos
        self._label_decisions = label_decisions
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
        from bytemark.config.constants import (
            YOLO_IMAGE_EXTS,
            YOLO_IMAGES_SUBDIR,
            YOLO_LABEL_EXT,
            YOLO_LABELS_SUBDIR,
            YOLO_TRAIN_DIR,
            YOLO_VAL_DIR,
        )
        from bytemark.core.dataset.yaml_handler import generate_yaml

        dest_name = _make_dest_name(self._sources)
        dest = self._output_parent / dest_name

        self.log_line.emit("Analysing source dataset(s)...", "active")

        pairs: list[tuple[Path, Optional[Path]]] = []
        for info in self._source_infos:
            src = info["path"]
            keep = self._label_decisions.get(str(src), True)
            images = [
                p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in YOLO_IMAGE_EXTS
            ]
            for img in images:
                lbl = None
                if keep:
                    lbl = self._find_label(img, src)
                    if lbl and not lbl.exists():
                        lbl = None
                pairs.append((img, lbl))

        if not pairs:
            raise ValueError("No images found in the selected source(s).")

        self.log_line.emit(
            f"Found {len(pairs)} image(s) across {len(self._sources)} source(s) — shuffling...",
            "done",
        )
        self.log_line.emit("Splitting into train / val...", "active")

        rng = random.Random()
        rng.shuffle(pairs)
        split_idx = max(1, int(len(pairs) * self._train_ratio))
        splits = {
            YOLO_TRAIN_DIR: pairs[:split_idx],
            YOLO_VAL_DIR: pairs[split_idx:],
        }

        seen: set[str] = set()
        for split_name, split_pairs in splits.items():
            img_out = dest / YOLO_IMAGES_SUBDIR / split_name
            lbl_out = dest / YOLO_LABELS_SUBDIR / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_path, lbl_path in split_pairs:
                stem = img_path.stem
                if stem in seen:
                    stem = f"{img_path.parent.name}_{stem}"
                base = stem
                counter = 1
                while stem in seen:
                    stem = f"{base}_{counter}"
                    counter += 1
                seen.add(stem)

                shutil.copy2(img_path, img_out / (stem + img_path.suffix))
                if lbl_path and lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_out / (stem + YOLO_LABEL_EXT))

        self.log_line.emit("Train / val split complete...", "done")

        if self._auto_annotate and self._modalities:
            self.log_line.emit("Running auto-annotation...", "active")
            self._run_auto_annotate(dest)
            self.log_line.emit("Auto-annotation complete...", "done")

        self.log_line.emit("Generating data.yaml and finalising...", "active")
        generate_yaml(dest)
        self.log_line.emit("All done. Dataset is clean and ready to load, Annotator.", "done")
        self.finished.emit(True, str(dest))

    def _find_label(self, img: Path, src: Path) -> Optional[Path]:
        from bytemark.config.constants import (
            YOLO_IMAGES_SUBDIR,
            YOLO_LABEL_EXT,
            YOLO_LABELS_SUBDIR,
        )

        try:
            parts = list(img.relative_to(src).parts)
            for i, p in enumerate(parts):
                if p.lower() == YOLO_IMAGES_SUBDIR:
                    parts[i] = YOLO_LABELS_SUBDIR
                    return (src / Path(*parts)).with_suffix(YOLO_LABEL_EXT)
        except Exception:
            pass
        return img.with_suffix(YOLO_LABEL_EXT)

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
            lbl_path = derive_label_path(img_path)
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                continue  # already labelled — skip
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
            img_ann = ImageAnnotations(
                image_path=str(img_path),
                label_path=str(lbl_path),
                instances=filtered,
            )
            write_label_file(img_ann)
        engine.unload()


# ── Base page ─────────────────────────────────────────────────────────────────


class _Page(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("overlay_dialog")
        self.setFixedWidth(580)
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


# ── Label handling page (multi-source only) ───────────────────────────────────


class _LabelHandlingPage(_Page):
    proceeded = Signal(dict)
    cancelled = Signal()

    def __init__(self, source_infos: list[dict]) -> None:
        super().__init__()
        self._infos = source_infos
        self._checks: dict[str, QCheckBox] = {}
        self._build()

    def _build(self) -> None:
        self._layout.addWidget(self._title("Label Handling, Annotator"))

        any_labels = any(i["has_labels"] for i in self._infos)
        all_labeled = all(i["has_labels"] for i in self._infos)

        if not any_labels:
            summary = (
                "None of the selected sources carry annotation files — "
                "all will contribute images only. Nothing to decide here, Annotator."
            )
        elif any_labels and not all_labeled:
            summary = (
                "A mixed situation, Annotator — some sources carry annotations, "
                "some do not.\n\n"
                "Decide below which labels to carry into the merged dataset. "
                "Sources without labels will contribute images only."
            )
        else:
            summary = (
                "All sources carry annotations. Decide which labels to keep, "
                "Annotator — deselecting a source discards its annotations "
                "while still including its images."
            )
        self._layout.addWidget(self._body(summary))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(min(40 + 58 * len(self._infos), 240))
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(10)

        for info in self._infos:
            key = str(info["path"])
            has_lbl = info["has_labels"]

            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(f"  {info['name']}")
            cb.setChecked(has_lbl)
            cb.setEnabled(has_lbl)

            parts = [f"{info['image_count']} images"]
            if has_lbl:
                parts += [
                    f"{info['label_count']} labels",
                    "structured" if info["is_structured"] else "flat",
                ]
            else:
                parts.append("no labels")
            detail = QLabel("  ·  ".join(parts))
            detail.setObjectName("dimmed")

            row.addWidget(cb)
            row.addStretch()
            row.addWidget(detail)
            inner_layout.addWidget(row_widget)
            self._checks[key] = cb

        scroll.setWidget(inner)
        self._layout.addWidget(scroll)

        ok = QPushButton("> Proceed with these choices")
        ok.setObjectName("primary_button")
        ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled)
        self._layout.addLayout(self._btn_row(ok, cancel))

    def _on_ok(self) -> None:
        self.proceeded.emit({k: cb.isChecked() for k, cb in self._checks.items()})


# ── Auto / manual page ────────────────────────────────────────────────────────


class _AutoManualPage(_Page):
    auto_chosen = Signal()
    manual_chosen = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._layout.addWidget(self._title("How Shall We Proceed?"))
        self._layout.addWidget(
            self._body(
                "The auto-annotator can handle the laborious groundwork, Annotator, "
                "leaving you to review and refine — a sensible division of labour.\n\n"
                "Note that the current model is optimised for human subjects only. "
                "Already-labelled images will not be overwritten.\n\n"
                "Which path shall we take?"
            )
        )
        auto_btn = QPushButton("> Auto-annotate for me")
        auto_btn.setObjectName("primary_button")
        auto_btn.clicked.connect(self.auto_chosen)
        manual_btn = QPushButton("I will annotate manually")
        manual_btn.clicked.connect(self.manual_chosen)
        self._layout.addLayout(self._btn_row(auto_btn, manual_btn))


# ── Modality page ─────────────────────────────────────────────────────────────


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


# ── Split page ────────────────────────────────────────────────────────────────


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


# ── Confirm page ──────────────────────────────────────────────────────────────


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


# ── Execution page ────────────────────────────────────────────────────────────


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

    _PAGE_LABELS = 0
    _PAGE_AUTO = 1
    _PAGE_MOD = 2
    _PAGE_SPLIT = 3
    _PAGE_CONFIRM = 4
    _PAGE_EXEC = 5

    def __init__(self, sources: list[Path], output_parent: Path, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(640)

        self._sources = sources
        self._output_parent = output_parent
        self._source_infos: list[dict] = [_analyze_source(s) for s in sources]

        # Default: keep labels wherever they exist
        self._label_decisions: dict[str, bool] = {
            str(i["path"]): i["has_labels"] for i in self._source_infos
        }
        self._train_ratio = 0.8
        self._auto_annotate = False
        self._modalities: set[Modality] = set()
        self._thread: Optional[QThread] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        outer.addWidget(self._stack, alignment=Qt.AlignmentFlag.AlignCenter)

        multi = len(sources) > 1
        any_labels = any(i["has_labels"] for i in self._source_infos)
        self._show_label_page = multi and any_labels

        # Build pages — label page is real only when needed
        if self._show_label_page:
            self._label_page = _LabelHandlingPage(self._source_infos)
            self._label_page.proceeded.connect(self._on_label_decisions)
            self._label_page.cancelled.connect(self.reject)
        else:
            self._label_page = _Page()  # inert placeholder

        self._auto_page = _AutoManualPage()
        self._modality_page = _ModalityPage()
        self._split_page = _SplitPage()
        self._confirm_page = _ConfirmPage()
        self._exec_page = _ExecutionPage()

        for page in (
            self._label_page,
            self._auto_page,
            self._modality_page,
            self._split_page,
            self._confirm_page,
            self._exec_page,
        ):
            self._stack.addWidget(page)

        self._auto_page.auto_chosen.connect(self._on_auto)
        self._auto_page.manual_chosen.connect(self._on_manual)
        self._modality_page.proceeded.connect(self._on_modalities)
        self._modality_page.cancelled.connect(self.reject)
        self._split_page.proceeded.connect(self._on_split)
        self._split_page.cancelled.connect(self.reject)
        self._confirm_page.confirmed.connect(self._start_execution)
        self._confirm_page.cancelled.connect(self.reject)

        start = self._PAGE_LABELS if self._show_label_page else self._PAGE_AUTO
        self._stack.setCurrentIndex(start)

    def _goto(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    def _on_label_decisions(self, decisions: dict) -> None:
        self._label_decisions = decisions
        self._goto(self._PAGE_AUTO)

    def _on_auto(self) -> None:
        self._auto_annotate = True
        self._goto(self._PAGE_MOD)

    def _on_manual(self) -> None:
        self._auto_annotate = False
        self._modalities = set()
        self._goto(self._PAGE_SPLIT)

    def _on_modalities(self, mods: set) -> None:
        self._modalities = mods
        self._goto(self._PAGE_SPLIT)

    def _on_split(self, ratio: float) -> None:
        self._train_ratio = ratio

        keeping = [
            info["name"]
            for info in self._source_infos
            if self._label_decisions.get(str(info["path"]), False)
        ]
        discarding = [
            info["name"]
            for info in self._source_infos
            if not self._label_decisions.get(str(info["path"]), False)
        ]

        lines = [
            f"Source(s): {', '.join(s.name for s in self._sources)}",
            f"Auto-annotation: {'enabled' if self._auto_annotate else 'disabled'}",
        ]
        if self._auto_annotate and self._modalities:
            lines.append(f"Modalities: {', '.join(m.name for m in self._modalities)}")
        if keeping:
            lines.append(f"Labels kept from: {', '.join(keeping)}")
        if discarding:
            lines.append(f"Labels discarded from: {', '.join(discarding)}")
        lines.append(
            f"Train / Val split: "
            f"{int(self._train_ratio * 100)} / {int((1 - self._train_ratio) * 100)}"
        )
        lines.append(f"Output naming: {' + '.join(s.name for s in self._sources)}_<timestamp>")

        self._confirm_page.set_details("\n".join(lines))
        self._goto(self._PAGE_CONFIRM)

    def _start_execution(self) -> None:
        self._goto(self._PAGE_EXEC)
        self._thread = QThread()
        self._worker = _DatasetWorker(
            sources=self._sources,
            source_infos=self._source_infos,
            label_decisions=self._label_decisions,
            output_parent=self._output_parent,
            train_ratio=self._train_ratio,
            auto_annotate=self._auto_annotate,
            modalities=self._modalities,
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
