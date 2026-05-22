"""
bytemark/ui/tutorial.py
Guided walkthrough — Prof. Annotate narrates ByteMark's surfaces.

A transparent overlay sits on top of MainWindow, dimming everything
except a single highlighted widget. The Professor's popup sits beside
the highlight and offers Next / Back / Skip controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.prof_annotate import (
    PROF_PORTRAIT_SMALL,
    TUTORIAL_INTRO,
    TUTORIAL_OUTRO,
    presence,
)


# ── Step model ───────────────────────────────────────────────────────────────


@dataclass
class TutorialStep:
    title: str  # short header, e.g. "THE CANVAS"
    message: str  # Prof.'s speech
    target_attr: Optional[str] = None  # name of the MainWindow attribute to highlight
    # `target_attr` is dotted, e.g. "_canvas" or "_file_explorer".


def _build_steps() -> list[TutorialStep]:
    return [
        TutorialStep(
            title="THE TOP BAR",
            message=(
                "Behold, Annotator — the upper rune-bar.\n\n"
                "• 'Create New Dataset' conjures fresh structure from raw images.\n"
                "• 'Keybindings' reveals every shortcut at your disposal.\n"
                "• 'Help' opens the chronicles online."
            ),
            target_attr="_create_btn",
        ),
        TutorialStep(
            title="THE FILE EXPLORER",
            message=(
                "On the left, your dataset's halls — train and val splits, "
                "every image neatly listed.\n\n"
                "Click an entry to summon it onto the canvas. Press Ctrl+O "
                "to open a folder, Ctrl+F for a single image."
            ),
            target_attr="_file_explorer",
        ),
        TutorialStep(
            title="THE STATS PANEL",
            message=(
                "Beneath the explorer, your dataset's vital signs — "
                "image counts, annotation coverage, and the balance "
                "between train and val. Watch it, Annotator."
            ),
            target_attr="_stats_panel",
        ),
        TutorialStep(
            title="MODALITY TOGGLES",
            message=(
                "Above the canvas: three sigils — BBOX, KPTS, SEG. "
                "Toggle which annotations appear.\n\n"
                "Bind them to Ctrl+1 / Ctrl+2 / Ctrl+3 for swift dismissal."
            ),
            target_attr="_modality_selector",
        ),
        TutorialStep(
            title="THE CANVAS",
            message=(
                "The arena of your craft.\n\n"
                "• B — draw a bounding box.\n"
                "• K — place keypoints (a bbox must rule them first).\n"
                "• S — trace a segmentation mask.\n"
                "• Middle-click to pan, scroll to zoom, Esc to retreat."
            ),
            target_attr="_canvas",
        ),
        TutorialStep(
            title="THE JSON ORACLE",
            message=(
                "Click any annotation and its essence appears here as JSON. "
                "Edit the runes directly to shape the instance — the canvas "
                "will follow your will."
            ),
            target_attr="_json_editor",
        ),
        TutorialStep(
            title="THE DATA.YAML SCROLL",
            message=(
                "Your dataset's manifest. Class names, keypoint definitions, "
                "splits — all recorded here.\n\n"
                "Ctrl+S commits your edits; idle hands trigger a quiet autosave."
            ),
            target_attr="_yaml_editor",
        ),
        TutorialStep(
            title="THE KEYBINDINGS TOME",
            message=(
                "Every binding lies here, Annotator. When the runes "
                "escape your memory, open this — the answers wait within."
            ),
            target_attr="_keybindings_btn",
        ),
    ]


# ── Overlay widget ───────────────────────────────────────────────────────────


class _TutorialOverlay(QWidget):
    """Sits on top of MainWindow. Paints a darkened wash with a 'window'
    cut around the highlighted widget, then draws a gold border around it."""

    def __init__(self, host: QMainWindow) -> None:
        super().__init__(host)
        self.setObjectName("tutorial_overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._target_rect: Optional[QRect] = None
        self.resize(host.size())

    def set_target_rect(self, rect: Optional[QRect]) -> None:
        self._target_rect = rect
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        full = QPainterPath()
        full.addRect(self.rect())

        if self._target_rect is not None and not self._target_rect.isNull():
            hole = QPainterPath()
            hole.addRect(self._target_rect)
            full = full.subtracted(hole)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(4, 2, 8, 210))
        painter.drawPath(full)

        if self._target_rect is not None and not self._target_rect.isNull():
            pen = QPen(QColor("#D4AF37"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._target_rect.adjusted(-1, -1, 0, 0))


# ── Floating speech panel ────────────────────────────────────────────────────


class _ProfSpeechPanel(QFrame):
    """Standalone (non-modal) Prof. card that floats over the overlay."""

    def __init__(
        self,
        host: QMainWindow,
        on_next: Callable[[], None],
        on_back: Callable[[], None],
        on_skip: Callable[[], None],
    ) -> None:
        super().__init__(host)
        self.setObjectName("overlay_dialog")
        self.setFixedWidth(500)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(16)

        # Compact portrait (left)
        portrait_col = QVBoxLayout()
        portrait_col.setSpacing(2)
        portrait = QLabel(PROF_PORTRAIT_SMALL)
        portrait.setObjectName("prof_portrait")
        portrait.setTextFormat(Qt.TextFormat.PlainText)
        portrait.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        portrait_col.addWidget(portrait)

        name = QLabel("PROF.\nANNOTATE")
        name.setObjectName("prof_name")
        name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        portrait_col.addWidget(name)
        portrait_col.addStretch(1)

        portrait_wrap = QWidget()
        portrait_wrap.setLayout(portrait_col)
        portrait_wrap.setFixedWidth(120)
        outer.addWidget(portrait_wrap)

        # Right column: step + speech + buttons
        right = QVBoxLayout()
        right.setSpacing(8)

        self._step_label = QLabel("")
        self._step_label.setObjectName("prof_step_counter")
        right.addWidget(self._step_label)

        self._title_label = QLabel("")
        self._title_label.setObjectName("dialog_title")
        right.addWidget(self._title_label)

        speech_frame = QFrame()
        speech_frame.setObjectName("prof_speech_frame")
        sf_layout = QVBoxLayout(speech_frame)
        sf_layout.setContentsMargins(0, 0, 0, 0)
        sf_layout.setSpacing(0)

        self._speech_label = QLabel("")
        self._speech_label.setObjectName("prof_speech")
        self._speech_label.setWordWrap(True)
        self._speech_label.setTextFormat(Qt.TextFormat.PlainText)
        self._speech_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._speech_label.setContentsMargins(12, 10, 12, 10)
        self._speech_label.setMinimumWidth(300)

        self._speech_scroll = QScrollArea()
        self._speech_scroll.setObjectName("prof_speech_scroll")
        self._speech_scroll.setWidget(self._speech_label)
        self._speech_scroll.setWidgetResizable(True)
        self._speech_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._speech_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._speech_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._speech_scroll.setMinimumHeight(120)
        self._speech_scroll.setMaximumHeight(260)
        sf_layout.addWidget(self._speech_scroll)
        right.addWidget(speech_frame, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._skip_btn = QPushButton("Skip Tutorial")
        self._skip_btn.setObjectName("arcane_button_dim")
        self._skip_btn.setAutoDefault(False)
        self._skip_btn.clicked.connect(on_skip)
        btn_row.addWidget(self._skip_btn)

        btn_row.addStretch(1)

        self._back_btn = QPushButton("< Back")
        self._back_btn.setObjectName("arcane_button_dim")
        self._back_btn.setAutoDefault(False)
        self._back_btn.clicked.connect(on_back)
        btn_row.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next >")
        self._next_btn.setObjectName("arcane_button")
        self._next_btn.setDefault(True)
        self._next_btn.setAutoDefault(True)
        self._next_btn.clicked.connect(on_next)
        btn_row.addWidget(self._next_btn)

        # The panel itself is a QFrame, not a QDialog — let Qt accept focus
        # so the Next button can hold it (and Enter activates the default).
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        right.addLayout(btn_row)

        outer.addLayout(right, stretch=1)

    def set_content(
        self,
        title: str,
        message: str,
        step_text: str,
        is_first: bool,
        is_last: bool,
    ) -> None:
        self._title_label.setText(title)
        self._speech_label.setText(message)
        self._step_label.setText(step_text)
        self._back_btn.setEnabled(not is_first)
        self._next_btn.setText("Finish >" if is_last else "Next >")
        # Reset scroll to the top whenever a new step's message lands.
        self._speech_scroll.verticalScrollBar().setValue(0)


# ── Walkthrough controller ───────────────────────────────────────────────────


class TutorialWalkthrough:
    """Orchestrates the overlay + speech panel through the steps."""

    def __init__(self, host: QMainWindow) -> None:
        self._host = host
        self._steps = _build_steps()
        self._idx = 0
        self._active = False

        self._overlay = _TutorialOverlay(host)
        self._panel = _ProfSpeechPanel(
            host,
            on_next=self._on_next,
            on_back=self._on_back,
            on_skip=self.cancel,
        )
        self._overlay.hide()
        self._panel.hide()

        # Reposition pieces when the host resizes/moves.
        self._filter = _HostFilter(self, host)
        host.installEventFilter(self._filter)

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._steps:
            return
        # Intro popup first, then dive into the highlighted steps.
        from src.ui.dialogs.prof_dialog import ProfDialog

        intro = ProfDialog(
            message=TUTORIAL_INTRO,
            title="Prof. Annotate",
            primary_label="> Begin the Tour",
            secondary_label="Skip",
            parent=self._host,
        )
        result = intro.exec()
        if result != intro.DialogCode.Accepted:
            self._mark_seen()
            return

        self._idx = 0
        self._active = True
        # Prof. lives inside the floating speech panel while the walkthrough
        # is on — the workshop section stays in its 'absent' state.
        presence().summon()
        self._overlay.resize(self._host.size())
        self._overlay.show()
        self._overlay.raise_()
        self._panel.show()
        self._panel.raise_()
        self._render_step()

    def cancel(self) -> None:
        self._finish(mark_seen=True)

    def is_active(self) -> bool:
        return self._active

    def reposition(self) -> None:
        """Called when the host window moves or resizes."""
        if not self._active:
            return
        self._overlay.resize(self._host.size())
        self._place_panel()
        self._refresh_target_rect()

    # ── Step navigation ──────────────────────────────────────────────────────

    def _on_next(self) -> None:
        if self._idx >= len(self._steps) - 1:
            self._finish(mark_seen=True, show_outro=True)
            return
        self._idx += 1
        self._render_step()

    def _on_back(self) -> None:
        if self._idx == 0:
            return
        self._idx -= 1
        self._render_step()

    def _render_step(self) -> None:
        step = self._steps[self._idx]
        self._panel.set_content(
            title=step.title,
            message=step.message,
            step_text=f"STEP {self._idx + 1} OF {len(self._steps)}",
            is_first=(self._idx == 0),
            is_last=(self._idx == len(self._steps) - 1),
        )
        self._refresh_target_rect()
        self._place_panel()
        self._panel.raise_()
        # Hand keyboard focus to the Next button so Enter advances and the
        # focused control is unmistakable. Without this, focus stays on the
        # MainWindow (or the previously-focused top-bar button), which means
        # Enter would activate that button instead of the tutorial.
        self._panel.activateWindow()
        self._panel._next_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def _refresh_target_rect(self) -> None:
        step = self._steps[self._idx]
        target = self._resolve_target(step.target_attr)
        if target is None or not target.isVisible():
            self._overlay.set_target_rect(None)
            return
        top_left = target.mapTo(self._host, QPoint(0, 0))
        rect = QRect(top_left, target.size()).adjusted(-4, -4, 4, 4)
        # Clamp to inside the overlay — the highlight border draws 1 px
        # outside the rect on the top/left, so any portion past x=0 or y=0
        # is invisible. This is what made the left-panel highlights look
        # like they were missing their left edge.
        overlay_bounds = self._overlay.rect().adjusted(2, 2, -2, -2)
        rect = rect.intersected(overlay_bounds)
        self._overlay.set_target_rect(rect)

    def _resolve_target(self, attr: Optional[str]) -> Optional[QWidget]:
        if not attr:
            return None
        return getattr(self._host, attr, None)

    def _place_panel(self) -> None:
        """Position the speech panel adjacent to the target — avoiding overlap
        when possible. Falls back to centred-bottom when no target.

        Strategy: rank the four sides of the target by how much room they
        offer, then clamp the panel into the host bounds. Strict
        rejection-based placement (the previous approach) broke for targets
        flush with an edge (e.g. the top bar at y=0) because every candidate
        failed the margin check and the panel slid to the bottom fallback."""
        self._panel.adjustSize()
        host_w = self._host.width()
        host_h = self._host.height()
        pw = self._panel.width()
        ph = self._panel.height()
        margin = 18

        step = self._steps[self._idx]
        target = self._resolve_target(step.target_attr)

        if target is None or not target.isVisible():
            x = (host_w - pw) // 2
            y = host_h - ph - margin
            self._panel.move(max(margin, x), max(margin, y))
            return

        top_left = target.mapTo(self._host, QPoint(0, 0))
        t_rect = QRect(top_left, target.size())

        # Available room on each side of the target rect.
        room_below = host_h - t_rect.bottom() - margin
        room_above = t_rect.top() - margin
        room_right = host_w - t_rect.right() - margin
        room_left = t_rect.left() - margin

        def clamp_x(x: int) -> int:
            return max(margin, min(x, host_w - pw - margin))

        def clamp_y(y: int) -> int:
            return max(margin, min(y, host_h - ph - margin))

        def overlaps(x: int, y: int) -> bool:
            return QRect(x, y, pw, ph).intersects(t_rect)

        # Side proposal — returns (x, y) for a panel placed on that side.
        proposals = {
            "below": (clamp_x(t_rect.left()), t_rect.bottom() + margin),
            "above": (clamp_x(t_rect.left()), t_rect.top() - ph - margin),
            "right": (t_rect.right() + margin, clamp_y(t_rect.top())),
            "left": (t_rect.left() - pw - margin, clamp_y(t_rect.top())),
        }
        rooms = {
            "below": (room_below, ph),
            "above": (room_above, ph),
            "right": (room_right, pw),
            "left": (room_left, pw),
        }

        # Order: prefer the side with the largest absolute room, but only
        # consider a side viable if it has enough room for the relevant axis.
        ranked = sorted(rooms.items(), key=lambda kv: kv[1][0], reverse=True)
        for side, (room, needed) in ranked:
            if room < needed:
                continue
            x, y = proposals[side]
            x_clamped, y_clamped = clamp_x(x), clamp_y(y)
            if not overlaps(x_clamped, y_clamped):
                self._panel.move(x_clamped, y_clamped)
                return

        # Fallback — no side has enough room. Place on the half of the host
        # opposite to the target's center so the user can still see both.
        if t_rect.center().y() < host_h // 2:
            y = host_h - ph - margin
        else:
            y = margin
        x = clamp_x((host_w - pw) // 2)
        self._panel.move(x, max(margin, y))

    # ── Teardown ─────────────────────────────────────────────────────────────

    def _finish(self, mark_seen: bool, show_outro: bool = False) -> None:
        was_active = self._active
        self._active = False
        self._overlay.hide()
        self._panel.hide()
        if was_active:
            # Hand Prof. back to the workshop (or to the outro popup that
            # follows — its QDialog show event will summon him again).
            presence().release()
        if mark_seen:
            self._mark_seen()
        if show_outro:
            from src.ui.dialogs.prof_dialog import ProfDialog

            ProfDialog(
                message=TUTORIAL_OUTRO,
                title="Prof. Annotate",
                primary_label="> So be it",
                parent=self._host,
            ).exec()

    @staticmethod
    def _mark_seen() -> None:
        from src.utils.prefs import set_pref

        set_pref("tutorial_seen", True)


# ── Event filter: resize/move + keyboard while the tutorial is up ────────────


class _HostFilter(QObject):
    """Forwards resize/move events to the controller so the overlay tracks
    the host window. While the tutorial is active, also intercepts keyboard
    input so Enter / Esc / arrow keys drive the walkthrough regardless of
    which child widget happens to hold focus."""

    def __init__(self, controller: TutorialWalkthrough, parent: QObject) -> None:
        super().__init__(parent)
        self._ctrl = controller

    def eventFilter(self, _obj, event) -> bool:  # noqa: D401
        t = event.type()
        if t in (QEvent.Type.Resize, QEvent.Type.Move):
            self._ctrl.reposition()
            return False

        if t == QEvent.Type.KeyPress and self._ctrl.is_active():
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._ctrl._on_next()
                return True
            if key == Qt.Key.Key_Escape:
                self._ctrl.cancel()
                return True
            if key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
                self._ctrl._on_next()
                return True
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Up):
                self._ctrl._on_back()
                return True
        return False
