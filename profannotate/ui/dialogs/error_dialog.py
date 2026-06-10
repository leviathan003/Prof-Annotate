"""
profannotate/ui/dialogs/error_dialog.py
Prof.-flavored error overlay dialog. Adapts to the active screen.
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

from profannotate.ui.dialogs._prof_layout import build_prof_column, screen_aware_size


class ErrorDialog(QDialog):
    def __init__(self, message: str, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog_error")
        chosen_w = screen_aware_size(frame, preferred_w=600, min_w=340, parent=parent)
        frame.setMinimumWidth(chosen_w)

        outer_h = QHBoxLayout(frame)
        outer_h.setContentsMargins(22, 20, 22, 20)
        outer_h.setSpacing(18)

        # ── Prof. portrait column (concerned variant — same art, red name) ──
        prof_col, _ = build_prof_column(
            parent, size="compact", name_text="PROF. ANNOTATE"
        )
        outer_h.addWidget(prof_col)

        # ── Right column ─────────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        title = QLabel("A disturbance, Annotator.")
        title.setObjectName("dialog_title_error")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right.addWidget(title)

        msg = QLabel(message)
        msg.setObjectName("dialog_body")
        msg.setAlignment(Qt.AlignmentFlag.AlignLeft)
        msg.setWordWrap(True)
        msg.setContentsMargins(0, 4, 0, 4)
        msg.setMinimumWidth(200)

        scroll = QScrollArea()
        scroll.setObjectName("prof_speech_scroll")
        scroll.setWidget(msg)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(110)
        scroll.setMaximumHeight(360)
        right.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn = QPushButton("> Acknowledged")
        btn.setObjectName("arcane_button_danger")
        btn.setDefault(True)
        btn.setAutoDefault(True)
        btn.clicked.connect(self.reject)
        btn_row.addWidget(btn)
        right.addLayout(btn_row)
        self._focus_target = btn

        outer_h.addLayout(right, stretch=1)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() in (
            Qt.Key.Key_Escape,
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            self.reject()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        self._focus_target.setFocus(Qt.FocusReason.OtherFocusReason)
