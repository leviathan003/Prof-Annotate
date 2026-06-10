"""
profannotate/ui/dialogs/modality_prompt.py
Checkbox prompt for selecting annotation modalities — Prof. presents the
choice and adapts to the active screen.
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

from profannotate.config.constants import AUTOANNOTATE_HUMAN_WARNING
from profannotate.core.annotation.models import Modality
from profannotate.ui.dialogs._prof_layout import build_prof_column, screen_aware_size


class ModalityPrompt(QDialog):
    def __init__(self, title: str, show_warning: bool = False, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        chosen_w = screen_aware_size(frame, preferred_w=600, min_w=340, parent=parent)
        frame.setMinimumWidth(chosen_w)

        outer_h = QHBoxLayout(frame)
        outer_h.setContentsMargins(22, 20, 22, 20)
        outer_h.setSpacing(18)

        prof_col, _ = build_prof_column(parent, size="compact")
        outer_h.addWidget(prof_col)

        inner = QVBoxLayout()
        inner.setSpacing(12)

        t = QLabel(title)
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignLeft)
        t.setWordWrap(True)
        inner.addWidget(t)

        if show_warning:
            w = QLabel(AUTOANNOTATE_HUMAN_WARNING)
            w.setObjectName("dialog_warning")
            w.setAlignment(Qt.AlignmentFlag.AlignLeft)
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

        inner.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("arcane_button_dim")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("> Proceed")
        ok_btn.setObjectName("arcane_button")
        ok_btn.setDefault(True)
        ok_btn.setAutoDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        inner.addLayout(btn_row)
        self._focus_target = ok_btn

        outer_h.addLayout(inner, stretch=1)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def selected_modalities(self) -> set[Modality]:
        return {m for m, cb in self._checks.items() if cb.isChecked()}

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        self._focus_target.setFocus(Qt.FocusReason.OtherFocusReason)
