"""
profannotate/ui/dialogs/kpt_bulk_edit_dialog.py
Remove selected keypoints across all label files in a dataset.
Triggered via Ctrl+Del shortcut — no toolbar button.
Handles datasets that have already had keypoints removed (multi-round safe).
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

from profannotate.config.constants import NUM_KEYPOINTS
from profannotate.config.skeleton import KEYPOINT_NAMES


class _BulkKptWorker(QObject):
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        root: Path,
        remove_indices: set[int],
        current_kpt_names: list[str],
    ) -> None:
        super().__init__()
        self._root = root
        self._indices = remove_indices
        self._current_kpt_names = current_kpt_names

    def run(self) -> None:
        try:
            from profannotate.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR

            lbl_dir = self._root / YOLO_LABELS_SUBDIR
            files = list(lbl_dir.rglob(f"*{YOLO_LABEL_EXT}"))
            self.log.emit(f"Processing {len(files)} label file(s)...")
            originals: dict = {}
            for lbl in files:
                originals[lbl] = lbl.read_text(encoding="utf-8")
                self._process(lbl)

            from profannotate.core.dataset.yaml_handler import load_yaml, save_yaml

            yaml_path = self._root / "data.yaml"
            originals[yaml_path] = (
                yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
            )
            data = load_yaml(self._root)

            remaining_names = [
                name for i, name in enumerate(self._current_kpt_names) if i not in self._indices
            ]
            # Only update kpt_shape and keypoint_names — leave nc/names/path/train/val untouched
            data["kpt_shape"] = [len(remaining_names), 3]
            data["keypoint_names"] = remaining_names
            # Clean up legacy keys from older zeroing approach
            data.pop("zeroed_keypoints", None)
            data.pop("zeroed_keypoint_names", None)
            save_yaml(self._root, data)

            self.log.emit(
                f"Done. {len(self._indices)} keypoint(s) removed. "
                f"{len(remaining_names)} remaining. data.yaml updated."
            )
            self.finished.emit(originals)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _process(self, path: Path) -> None:
        n_kpts = len(self._current_kpt_names)
        pose_fields = 3 * n_kpts
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            parts = line.strip().split()
            n = len(parts)
            # Identify pose or combined lines using CURRENT keypoint count
            is_pose = n == 1 + 4 + pose_fields
            is_combined = n > 1 + 4 + pose_fields and (n - 1 - 4 - pose_fields) % 2 == 0
            if is_pose or is_combined:
                new = list(parts[:5])  # class_id + bbox (4)
                for i in range(n_kpts):
                    if i not in self._indices:
                        off = 5 + i * 3
                        new += list(parts[off : off + 3])
                # Preserve trailing seg points (combined format)
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

        # Read current active keypoints from data.yaml (supports multi-round deletion)
        from profannotate.core.dataset.yaml_handler import load_yaml

        yaml_data = load_yaml(dataset_root)
        if (
            "keypoint_names" in yaml_data
            and isinstance(yaml_data["keypoint_names"], list)
            and yaml_data["keypoint_names"]
        ):
            self._active_kpt_names: list[str] = yaml_data["keypoint_names"]
        else:
            self._active_kpt_names = [KEYPOINT_NAMES[i] for i in range(NUM_KEYPOINTS)]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        from profannotate.ui.dialogs._prof_layout import screen_aware_size

        chosen_w = screen_aware_size(frame, preferred_w=580, min_w=360, parent=parent)
        frame.setMinimumWidth(chosen_w)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(12)

        t = QLabel("Bulk Keypoint Removal")
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(t)

        b = QLabel(
            "Select the keypoints to permanently remove from every label file in this dataset, "
            "Annotator.\n\n"
            "Their fields will be stripped entirely. "
            "Only kpt_shape and keypoint_names in data.yaml will be updated — "
            "all other dataset config is preserved.\n\n"
            "This cannot be undone with Ctrl+Z — once removed, the originals are gone."
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
        for idx, name in enumerate(self._active_kpt_names):
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
        self._run_btn = QPushButton("> Remove Selected Keypoints")
        self._run_btn.setObjectName("primary_button")
        self._run_btn.setDefault(True)
        self._run_btn.setAutoDefault(True)
        self._run_btn.clicked.connect(self._run)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(cancel_btn)
        inner.addLayout(btn_row)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        self._run_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _run(self) -> None:
        indices = {idx for idx, cb in self._checks.items() if cb.isChecked()}
        if not indices:
            self._status.setText("Please select at least one keypoint to proceed, Annotator.")
            return

        names = ", ".join(self._active_kpt_names[i] for i in sorted(indices))

        from profannotate.ui.dialogs.confirm_dialog import ConfirmDialog

        dlg = ConfirmDialog(
            "Permanent Dataset Modification — Are You Certain?",
            f"Annotator, you are about to permanently remove:\n\n{names}\n\n"
            f"These fields will be stripped from every annotation file. "
            f"data.yaml will be updated with the new keypoint count.\n\n"
            f"This cannot be reversed with Ctrl+Z.",
            "> Yes, remove them permanently",
            "No, let me reconsider",
            self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        self._run_btn.setEnabled(False)
        self._status.setText("Working...")
        self._thread = QThread()
        self._worker = _BulkKptWorker(self._root, indices, self._active_kpt_names)
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
