"""
bytemark/ui/dialogs/kpt_bulk_edit_dialog.py
Zero out selected keypoints across all label files in a dataset.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from bytemark.config.constants import NUM_KEYPOINTS
from bytemark.config.skeleton import KEYPOINT_NAMES


class _BulkKptWorker(QObject):
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, root: Path, zero_indices: set[int]) -> None:
        super().__init__()
        self._root = root
        self._indices = zero_indices

    def run(self) -> None:
        try:
            from bytemark.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR

            lbl_dir = self._root / YOLO_LABELS_SUBDIR
            files = list(lbl_dir.rglob(f"*{YOLO_LABEL_EXT}"))
            self.log.emit(f"Processing {len(files)} label file(s)...")
            originals = {}
            for lbl in files:
                originals[lbl] = lbl.read_text(encoding="utf-8")
                self._process(lbl)

            from bytemark.core.dataset.yaml_handler import load_yaml, save_yaml

            yaml_path = self._root / "data.yaml"
            originals[yaml_path] = (
                yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
            )
            data = load_yaml(self._root)
            data["zeroed_keypoints"] = sorted(self._indices)
            data["zeroed_keypoint_names"] = [
                KEYPOINT_NAMES[i] for i in sorted(self._indices) if i in KEYPOINT_NAMES
            ]
            save_yaml(self._root, data)
            self.log.emit("Complete. data.yaml has been updated accordingly.")
            self.finished.emit(originals)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _process(self, path: Path) -> None:
        pose_fields = 3 * NUM_KEYPOINTS
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            parts = line.strip().split()
            n = len(parts)
            is_pose = n == 1 + 4 + pose_fields
            is_combined = n > 1 + 4 + pose_fields and (n - 1 - 4 - pose_fields) % 2 == 0
            if is_pose or is_combined:
                new = list(parts[:5])
                for i in range(NUM_KEYPOINTS):
                    off = 5 + i * 3
                    new += ["0", "0", "0"] if i in self._indices else list(parts[off : off + 3])
                new += list(parts[5 + pose_fields :])
                out.append(" ".join(new))
            else:
                out.append(line)
        path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


class KptBulkEditDialog(QDialog):
    dataset_updated = Signal(dict)

    def __init__(self, dataset_root: Path, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._root = dataset_root

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        frame.setFixedWidth(540)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(12)

        t = QLabel("Bulk Keypoint Zeroing")
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(t)

        b = QLabel(
            "Select the keypoints to zero out across all label files in this dataset, "
            "Annotator.\n\n"
            "Their x, y, and visibility will be set to 0 throughout. "
            "The data.yaml will be updated to reflect the change.\n\n"
            "This cannot be undone with Ctrl+Z "
            "once the deletions are done, the originals are gone."
        )
        b.setObjectName("dialog_body")
        b.setWordWrap(True)
        b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(b)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(240)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        kw = QWidget()
        kl = QVBoxLayout(kw)
        kl.setSpacing(4)
        self._checks: dict[int, QCheckBox] = {}
        for idx in range(NUM_KEYPOINTS):
            name = KEYPOINT_NAMES.get(idx, str(idx))
            cb = QCheckBox(f"  {idx:02d}  {name}")
            cb.setChecked(False)
            kl.addWidget(cb)
            self._checks[idx] = cb
        scroll.setWidget(kw)
        inner.addWidget(scroll)

        self._status = QLabel("")
        self._status.setObjectName("dialog_body")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("> Zero Out Selected Keypoints")
        self._run_btn.setObjectName("primary_button")
        self._run_btn.clicked.connect(self._run)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(cancel_btn)
        inner.addLayout(btn_row)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def _run(self) -> None:
        indices = {idx for idx, cb in self._checks.items() if cb.isChecked()}
        if not indices:
            self._status.setText("Please select at least one keypoint to proceed, Annotator.")
            return

        names = ", ".join(KEYPOINT_NAMES.get(i, str(i)) for i in sorted(indices))

        from bytemark.ui.dialogs.confirm_dialog import ConfirmDialog

        dlg = ConfirmDialog(
            "Permanent Dataset Modification — Are You Certain?",
            f"Annotator, you are about to zero out the following keypoint(s) "
            f"across every label file in this dataset:\n\n"
            f"{names}\n\n"
            f"This is a substantive, dataset-wide change. "
            f"It cannot be reversed with Ctrl+Z, once deleted "
            f"the originals are permanently gone.\n\n"
            f"Proceed with the care befitting an expert.",
            "> Yes, I understand — proceed",
            "No, let me reconsider",
            self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        self._run_btn.setEnabled(False)
        self._status.setText("Working...")
        self._thread = QThread()
        self._worker = _BulkKptWorker(self._root, indices)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._status.setText)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(
            lambda e: self._status.setText(f"Something went wrong, Annotator: {e}")
        )
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_done(self, originals: dict) -> None:
        self.dataset_updated.emit(originals)
        self.accept()
