"""
bytemark/ui/widgets/prof_widget.py
Prof. Annotate's workshop — a small panel below data.yaml where Prof.
performs looping arcane "annotation" rituals.

Three loops drive the animation:
 - portrait frame cycle (~750 ms)
 - status line cycle (~3.2 s)
 - glyph feed scroll (~180 ms)

For low-end laptops/tablets we drive all three from a single ~180 ms
master timer with phase counters — three timers means three event-loop
wake-ups per cycle instead of one. The timer also stops outright while
the widget is hidden/minimised so an unfocused workshop costs zero CPU.

The portrait variant + glyph feed dimensions adapt to the widget's
allotted size — tiny on tablets, compact on laptops, full on desktops.
When a popup summons Prof. (see `prof_annotate.presence()`), the widget
swaps to an "absent" state with a small arcane note. On release he reappears.
"""

from __future__ import annotations

import random
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui.prof_annotate import (
    PROF_FRAMES,
    PROF_FRAMES_TINY,
    PROF_GLYPHS,
    PROF_TASKS,
    presence,
)
from src.utils.ui_scaling import portrait_column_width, prof_portrait_tier


class ProfWidget(QFrame):
    """Prof.'s workshop panel — animated portrait + status + glyph feed."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("prof_workshop")
        # Low minimum so the panel still works on a tablet-class screen.
        # The portrait variant will downgrade to tiny when needed.
        self.setMinimumHeight(130)

        self._frame_idx = 0
        self._task_idx = random.randrange(len(PROF_TASKS))
        # Glyph feed dimensions are recomputed on resize — seed conservatively.
        self._glyph_line_width = 22
        self._glyph_buffer: list[str] = []
        self._current_tier: Optional[str] = None

        self._build_ui()
        self._start_timers()

        presence().visibility_changed.connect(self._on_presence_changed)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QLabel("PROF. ANNOTATE'S WORKSHOP")
        header.setObjectName("section_header")
        outer.addWidget(header)

        # Stacked: present (working) ←→ absent (summoned)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_present_page())  # index 0
        self._stack.addWidget(self._build_absent_page())  # index 1
        self._stack.setCurrentIndex(0)
        outer.addWidget(self._stack, stretch=1)

    def _build_present_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(12)

        # ── Portrait column (width adjusted in `_apply_tier`) ────────────────
        self._portrait_label = QLabel(PROF_FRAMES[0])
        self._portrait_label.setObjectName("prof_portrait")
        self._portrait_label.setTextFormat(Qt.TextFormat.PlainText)
        self._portrait_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        layout.addWidget(self._portrait_label, alignment=Qt.AlignmentFlag.AlignTop)

        # ── Right column: status + glyph feed ────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        self._status_label = QLabel(f"▸  {PROF_TASKS[self._task_idx]}")
        self._status_label.setObjectName("prof_status")
        self._status_label.setWordWrap(True)
        right.addWidget(self._status_label)

        self._feed_frame = QFrame()
        self._feed_frame.setObjectName("prof_feed_frame")
        self._feed_layout = QVBoxLayout(self._feed_frame)
        self._feed_layout.setContentsMargins(8, 6, 8, 6)
        self._feed_layout.setSpacing(0)

        self._feed_labels: list[QLabel] = []
        # Build with placeholder lines — `_apply_tier` will replace these
        # with the correct count for the active screen size.
        self._rebuild_feed_lines(3)

        right.addWidget(self._feed_frame, stretch=1)

        layout.addLayout(right, stretch=1)
        return page

    def _build_absent_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)
        layout.addStretch(1)

        glyph = QLabel("✦       ✦       ✦")
        glyph.setObjectName("prof_absent_glyph")
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(glyph)

        note = QLabel("Prof. is otherwise engaged.")
        note.setObjectName("prof_absent_note")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note)

        layout.addStretch(1)
        return page

    # ── Tier / resize handling ───────────────────────────────────────────────

    def _rebuild_feed_lines(self, count: int) -> None:
        # Clear existing
        for lbl in self._feed_labels:
            self._feed_layout.removeWidget(lbl)
            lbl.deleteLater()
        self._feed_labels = []
        self._glyph_buffer = []
        for _ in range(max(1, count)):
            content = "".join(
                random.choice(PROF_GLYPHS + "    ") for _ in range(self._glyph_line_width)
            )
            self._glyph_buffer.append(content)
            lbl = QLabel(content)
            lbl.setObjectName("prof_feed")
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._feed_layout.addWidget(lbl)
            self._feed_labels.append(lbl)

    def _apply_tier(self) -> None:
        """Reconcile portrait variant + feed dimensions to the current size."""
        avail_w = self.width()
        avail_h = max(0, self.height() - 30)  # subtract header height
        # Reserve ~140 px for the status + glyph feed; rest goes to portrait,
        # capped by the largest natural portrait width (230). This lets the
        # compact portrait appear on standard laptops without giving the
        # full portrait a window-dominating column.
        portrait_budget_w = max(60, min(avail_w - 140, 230))
        portrait_budget_h = max(80, avail_h - 40)
        tier = prof_portrait_tier(
            available_height=portrait_budget_h,
            available_width=portrait_budget_w,
        )

        if tier == self._current_tier:
            # Still recompute the feed line width — width may have changed
            # even though the tier didn't.
            self._maybe_resize_feed_width(avail_w, portrait_budget_w)
            return
        self._current_tier = tier

        # Resize portrait column to the tier's natural width.
        col_w = min(portrait_column_width(tier), portrait_budget_w)
        self._portrait_label.setFixedWidth(max(60, col_w))

        # Pick the right animation frame set for this tier.
        frames = PROF_FRAMES_TINY if tier == "tiny" else PROF_FRAMES
        self._frames = frames
        self._frame_idx %= len(frames)
        self._portrait_label.setText(frames[self._frame_idx])

        # Recompute glyph feed line count from available height.
        # Each feed line ≈ 18 px; reserve 30 px for the status line.
        line_count = max(1, min(4, (avail_h - 40) // 18))
        # For tiny tier, prefer 1 line so the portrait dominates.
        if tier == "tiny":
            line_count = 1
        elif tier == "compact":
            line_count = min(line_count, 2)
        self._maybe_resize_feed_width(avail_w, col_w)
        self._rebuild_feed_lines(line_count)

    def _maybe_resize_feed_width(self, total_w: int, portrait_w: int) -> None:
        """Pick a glyph-line width that fills the right column without wrap."""
        right_w = max(60, total_w - portrait_w - 40)  # margins + spacing
        # Heuristic: ~7 px per monospace char in our QSS font sizing.
        new_w = max(8, min(40, right_w // 8))
        if new_w == self._glyph_line_width:
            return
        self._glyph_line_width = new_w
        # Re-seed buffers at the new width (will be regenerated on next tick).
        self._glyph_buffer = [
            "".join(random.choice(PROF_GLYPHS + "    ") for _ in range(new_w))
            for _ in self._glyph_buffer
        ]
        for i, lbl in enumerate(self._feed_labels):
            if i < len(self._glyph_buffer):
                lbl.setText(self._glyph_buffer[i])

    def resizeEvent(self, event) -> None:  # noqa: D401
        super().resizeEvent(event)
        # The first resizeEvent fires before show — defer until we have a
        # meaningful width.
        if self.width() > 10 and self.height() > 10:
            self._apply_tier()

    # ── Master timer ─────────────────────────────────────────────────────────
    # All three animation loops are driven from one 180 ms tick. Phase
    # counters keep portrait + status cycles on their slower intervals.

    _PORTRAIT_PHASE = 4  # 180 ms × 4 ≈ 720 ms
    _STATUS_PHASE = 18  # 180 ms × 18 ≈ 3240 ms

    def _start_timers(self) -> None:
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)  # cheap on low-end
        self._timer.setInterval(180)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _stop_timers(self) -> None:
        if hasattr(self, "_timer"):
            self._timer.stop()

    def _resume_timers(self) -> None:
        if hasattr(self, "_timer") and not self._timer.isActive():
            self._timer.start()

    # ── Tick handlers ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._phase += 1
        # Always scroll glyphs.
        self._tick_glyphs()
        if self._phase % self._PORTRAIT_PHASE == 0:
            self._tick_portrait()
        if self._phase % self._STATUS_PHASE == 0:
            self._tick_status()
        if self._phase >= self._STATUS_PHASE:
            self._phase = 0

    def _tick_portrait(self) -> None:
        frames = getattr(self, "_frames", PROF_FRAMES)
        self._frame_idx = (self._frame_idx + 1) % len(frames)
        self._portrait_label.setText(frames[self._frame_idx])

    def _tick_status(self) -> None:
        # Pick a new task that isn't the same as the current one.
        n = len(PROF_TASKS)
        if n <= 1:
            return
        new_idx = random.randrange(n - 1)
        if new_idx >= self._task_idx:
            new_idx += 1
        self._task_idx = new_idx
        self._status_label.setText(f"▸  {PROF_TASKS[self._task_idx]}")

    def _tick_glyphs(self) -> None:
        # Scroll each line left by one character, appending a fresh glyph
        # (or whitespace) at the tail to convey continuous activity.
        for i, line in enumerate(self._glyph_buffer):
            new_char = random.choice(PROF_GLYPHS + "    ")
            self._glyph_buffer[i] = line[1:] + new_char
            if i < len(self._feed_labels):
                self._feed_labels[i].setText(self._glyph_buffer[i])

    # ── Visibility — stop ticking when nobody can see us ─────────────────────

    def hideEvent(self, event) -> None:  # noqa: D401
        super().hideEvent(event)
        self._stop_timers()

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        # Only resume if Prof. is currently in the workshop (i.e. not summoned
        # away by a popup).
        if presence().is_in_workshop:
            self._resume_timers()

    # ── Presence handling ────────────────────────────────────────────────────

    def _on_presence_changed(self, is_present: bool) -> None:
        if is_present:
            self._stack.setCurrentIndex(0)
            self._resume_timers()
        else:
            self._stop_timers()
            self._stack.setCurrentIndex(1)
