"""
bytemark/ui/dialogs/modality_prompt.py
Checkbox prompt for selecting annotation modalities.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from bytemark.config.constants import AUTOANNOTATE_HUMAN_WARNING
from bytemark.core.annotation.models import Modality


class ModalityPrompt(QDialog):
    def __init__(self, title: str, show_warning: bool = False, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        frame.setFixedWidth(480)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(14)

        t = QLabel(title)
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(t)

        if show_warning:
            w = QLabel(AUTOANNOTATE_HUMAN_WARNING)
            w.setObjectName("dialog_warning")
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w.setWordWrap(True)
            inner.addWidget(w)

        # Checkboxes
        self._checks: dict[Modality, QCheckBox] = {}
        specs = [
            (Modality.BBOX, "> BBox", "#00CFFF"),
            (Modality.KEYPOINTS, "> Keypoints", "#FFD700"),
            (Modality.SEGMENTATION, "> Mask", "#CC44FF"),
        ]
        for modality, label, color in specs:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color};")
            inner.addWidget(cb)
            self._checks[modality] = cb

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("> Proceed")
        ok_btn.setObjectName("primary_button")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        inner.addLayout(btn_row)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def selected_modalities(self) -> set[Modality]:
        return {m for m, cb in self._checks.items() if cb.isChecked()}
