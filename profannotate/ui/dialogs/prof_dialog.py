"""
profannotate/ui/dialogs/prof_dialog.py
Reusable popup where Prof. Annotate speaks to the Annotator.

The dialog is intentionally lightweight — splash + tutorial draw on it.
Portrait on the left, speech + buttons on the right.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from profannotate.ui.dialogs._prof_layout import build_prof_column, screen_aware_size


class ProfDialog(QDialog):
    """Frameless overlay popup with the Professor's portrait + message.

    Parameters
    ----------
    message:
        The body of Prof.'s speech.
    title:
        Small title shown above the portrait (e.g. "Prof. Annotate").
    primary_label / primary_callback:
        The gold button. If callback is None the button only closes the dialog.
    secondary_label / secondary_callback:
        The dim button (e.g. "Skip"). Hidden if label is None.
    step_text:
        Optional small counter like "STEP 3 OF 9" shown above the speech.
    parent:
        Qt parent.
    """

    def __init__(
        self,
        message: str,
        title: str = "Prof. Annotate",
        primary_label: str = "> Continue",
        primary_callback: Optional[Callable[[], None]] = None,
        secondary_label: Optional[str] = None,
        secondary_callback: Optional[Callable[[], None]] = None,
        step_text: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._primary_cb = primary_callback
        self._secondary_cb = secondary_callback

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        chosen_w = screen_aware_size(frame, preferred_w=720, min_w=360, parent=parent)
        frame.setMinimumWidth(chosen_w)

        inner = QHBoxLayout(frame)
        inner.setContentsMargins(22, 20, 22, 20)
        inner.setSpacing(20)

        # ── Left column: portrait (size auto-scales with active screen) ──────
        prof_col, _ = build_prof_column(parent, size="full", name_text=title.upper())
        inner.addWidget(prof_col)

        # ── Right column: speech + buttons ───────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        if step_text:
            step = QLabel(step_text.upper())
            step.setObjectName("prof_step_counter")
            right.addWidget(step)

        speech_frame = QFrame()
        speech_frame.setObjectName("prof_speech_frame")
        speech_layout = QVBoxLayout(speech_frame)
        speech_layout.setContentsMargins(0, 0, 0, 0)
        speech_layout.setSpacing(0)

        speech = QLabel(message)
        speech.setObjectName("prof_speech")
        speech.setWordWrap(True)
        speech.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        speech.setTextFormat(Qt.TextFormat.PlainText)
        speech.setContentsMargins(14, 12, 14, 12)
        speech.setMinimumWidth(200)

        scroll = QScrollArea()
        scroll.setObjectName("prof_speech_scroll")
        scroll.setWidget(speech)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(140)
        scroll.setMaximumHeight(320)
        speech_layout.addWidget(scroll)
        right.addWidget(speech_frame, stretch=1)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)

        if secondary_label:
            sec_btn = QPushButton(secondary_label)
            sec_btn.setObjectName("arcane_button_dim")
            sec_btn.setAutoDefault(False)
            sec_btn.clicked.connect(self._on_secondary)
            button_row.addWidget(sec_btn)

        prim_btn = QPushButton(primary_label)
        prim_btn.setObjectName("arcane_button")
        prim_btn.setDefault(True)
        prim_btn.setAutoDefault(True)
        prim_btn.clicked.connect(self._on_primary)
        button_row.addWidget(prim_btn)
        self._focus_target = prim_btn

        right.addLayout(button_row)

        inner.addLayout(right, stretch=1)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_primary(self) -> None:
        if self._primary_cb is not None:
            self._primary_cb()
        self.accept()

    def _on_secondary(self) -> None:
        if self._secondary_cb is not None:
            self._secondary_cb()
        self.reject()

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() == Qt.Key.Key_Escape:
            self._on_secondary()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_primary()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        # Focus the primary button so Enter activates it and users see
        # which choice is the default.
        if hasattr(self, "_focus_target") and self._focus_target is not None:
            self._focus_target.setFocus(Qt.FocusReason.OtherFocusReason)
