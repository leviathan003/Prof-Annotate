"""
bytemark/ui/dialogs/prof_progress_dialog.py
Prof. Annotate at work — a frameless modal that shows Prof.'s animated
portrait, an arcane title, and a live log feed of what the background
worker is doing.

Used by every long-running flow: dataset reshuffle, bulk auto-annotation,
dataset indexing, single-image auto-annotation, etc.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.dialogs._prof_layout import (
    _screen_metrics,
    build_prof_column,
    screen_aware_size,
)
from src.ui.prof_annotate import PROF_FRAMES


class ProfProgressDialog(QDialog):
    """Live progress popup. Prof. is animated; the caller streams log lines.

    The dialog is intentionally non-cancellable — the surrounding code is
    responsible for aborting the underlying worker if needed.
    """

    def __init__(
        self,
        title: str = "Prof. is at work, Annotator.",
        subtitle: str = "Patience — the best work is never rushed.",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._frame_idx = 0
        self._log_map: dict[str, QLabel] = {}

        self._build_ui(title, subtitle, parent)
        self._start_animation()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self, title: str, subtitle: str, parent: Optional[QWidget]) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        chosen_w = screen_aware_size(frame, preferred_w=640, min_w=360, parent=parent)
        frame.setMinimumWidth(chosen_w)

        outer_h = QHBoxLayout(frame)
        outer_h.setContentsMargins(22, 22, 22, 22)
        outer_h.setSpacing(18)

        # ── Prof.'s animated portrait column ────────────────────────────────
        prof_col_widget, _ = build_prof_column(parent, size="compact")
        # Replace the static portrait inside the column with our animated label.
        # `build_prof_column` puts the QLabel first; grab it so we can swap text.
        labels = prof_col_widget.findChildren(QLabel)
        # Find the portrait label (object_name == "prof_portrait")
        self._portrait = next(
            (lbl for lbl in labels if lbl.objectName() == "prof_portrait"),
            labels[0],
        )
        # Seed the portrait with the first animation frame so it doesn't sit
        # on the static PROF_PORTRAIT_SMALL.
        self._portrait.setText(PROF_FRAMES[0])
        outer_h.addWidget(prof_col_widget)

        # ── Right column: title + subtitle + log feed ───────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        t = QLabel(title)
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignLeft)
        t.setWordWrap(True)
        right.addWidget(t)

        s = QLabel(subtitle)
        s.setObjectName("dialog_body")
        s.setAlignment(Qt.AlignmentFlag.AlignLeft)
        s.setWordWrap(True)
        right.addWidget(s)

        # ── Log feed (scrollable) ────────────────────────────────────────────
        self._log_frame = QFrame()
        self._log_frame.setObjectName("exec_log_frame")
        self._log_layout = QVBoxLayout(self._log_frame)
        self._log_layout.setContentsMargins(12, 10, 12, 10)
        self._log_layout.setSpacing(3)
        self._log_layout.addStretch(1)  # pushes new entries to the top

        scroll = QScrollArea()
        scroll.setObjectName("prof_speech_scroll")
        scroll.setWidget(self._log_frame)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        metrics = _screen_metrics(parent)
        scroll.setMinimumHeight(140)
        scroll.setMaximumHeight(max(160, int(metrics.height * 0.45)))
        self._scroll = scroll
        right.addWidget(scroll, stretch=1)

        outer_h.addLayout(right, stretch=1)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── Animation ────────────────────────────────────────────────────────────

    def _start_animation(self) -> None:
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self._tick)
        # Don't start yet — `showEvent` will kick it off, so a dialog that
        # is built but never shown costs zero CPU.

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        if hasattr(self, "_timer") and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: D401
        super().hideEvent(event)
        if hasattr(self, "_timer"):
            self._timer.stop()

    def _tick(self) -> None:
        self._frame_idx = (self._frame_idx + 1) % len(PROF_FRAMES)
        self._portrait.setText(PROF_FRAMES[self._frame_idx])

    # ── Log API ──────────────────────────────────────────────────────────────

    def add_log(self, message: str, state: str = "active") -> QLabel:
        """Append a new log entry. Returns the QLabel so the caller can later
        flip its state via `update_log`."""
        lbl = QLabel(message)
        lbl.setObjectName("exec_log_line")
        lbl.setProperty("state", state)
        lbl.setWordWrap(True)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)
        # Insert above the stretch so newest line is at the bottom of the list.
        self._log_layout.insertWidget(self._log_layout.count() - 1, lbl)
        self._log_map[message] = lbl
        # Auto-scroll to bottom after Qt processes the layout.
        QTimer.singleShot(0, self._scroll_to_bottom)
        return lbl

    def update_log(self, message_or_label, state: str) -> None:
        """Flip the state ("active"/"done"/"error") of an existing log entry.
        Accepts either the original message string or the QLabel returned
        from `add_log`."""
        lbl = (
            self._log_map.get(message_or_label)
            if isinstance(message_or_label, str)
            else message_or_label
        )
        if lbl is None:
            return
        lbl.setProperty("state", state)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)

    def status(self, message: str, state: str = "active") -> QLabel:
        """Convenience: either add or update the log entry depending on
        whether the message has been seen before."""
        if message in self._log_map:
            self.update_log(message, state)
            return self._log_map[message]
        return self.add_log(message, state)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def closeEvent(self, event) -> None:  # noqa: D401
        # Stop the animation when the dialog is dismissed so Qt can GC us.
        if hasattr(self, "_timer"):
            self._timer.stop()
        super().closeEvent(event)
