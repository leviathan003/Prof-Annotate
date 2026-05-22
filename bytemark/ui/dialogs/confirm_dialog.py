"""
bytemark/ui/dialogs/confirm_dialog.py
Prof.-flavored confirmation dialog. Adapts to the active screen.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from bytemark.ui.dialogs._prof_layout import build_prof_column, screen_aware_size


class ConfirmDialog(QDialog):
    def __init__(
        self,
        title: str,
        body: str,
        confirm_text: str = "> Confirm, proceed",
        cancel_text: str = "No, cancel",
        parent=None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        chosen_w = screen_aware_size(frame, preferred_w=620, min_w=340, parent=parent)
        frame.setMinimumWidth(chosen_w)

        outer_h = QHBoxLayout(frame)
        outer_h.setContentsMargins(22, 20, 22, 20)
        outer_h.setSpacing(18)

        # ── Prof. portrait column ────────────────────────────────────────────
        prof_col, _ = build_prof_column(parent, size="compact")
        outer_h.addWidget(prof_col)

        # ── Right column: title + scrollable body + buttons ──────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        t = QLabel(title)
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right.addWidget(t)

        b = QLabel(body)
        b.setObjectName("dialog_body")
        b.setAlignment(Qt.AlignmentFlag.AlignLeft)
        b.setWordWrap(True)
        b.setContentsMargins(0, 4, 0, 4)
        b.setMinimumWidth(200)

        scroll = QScrollArea()
        scroll.setObjectName("prof_speech_scroll")
        scroll.setWidget(b)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(110)
        scroll.setMaximumHeight(360)
        right.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch(1)

        cancel_btn = QPushButton(cancel_text)
        cancel_btn.setObjectName("arcane_button_dim")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)

        confirm_btn = QPushButton(confirm_text)
        confirm_btn.setObjectName("arcane_button")
        confirm_btn.setDefault(True)
        confirm_btn.setAutoDefault(True)
        confirm_btn.clicked.connect(self.accept)
        self._focus_target = confirm_btn

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(confirm_btn)
        right.addLayout(btn_row)

        outer_h.addLayout(right, stretch=1)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

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
