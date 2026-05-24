"""
bytemark/ui/dialogs/splash_screen.py
Startup splash — Prof. Annotate welcomes the Annotator.

A frameless centered widget shown for ~2 seconds (or until clicked)
before the main window appears.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.config.constants import APP_NAME, APP_VERSION
from src.ui.dialogs._prof_layout import _screen_metrics
from src.ui.prof_annotate import (
    PROF_PORTRAIT,
    PROF_PORTRAIT_SMALL,
    PROF_PORTRAIT_TINY,
    SPLASH_LINES,
    presence,
)


class SplashScreen(QWidget):
    """Frameless centered splash with Prof.'s portrait + branding."""

    def __init__(self, duration_ms: int = 2200) -> None:
        super().__init__(
            None,
            Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._duration_ms = duration_ms
        self._dismiss_cb: Optional[Callable[[], None]] = None
        self._dismissed = False

        self._build_ui()
        self._center_on_screen()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Pick a size that fits comfortably on the active screen. The splash
        # always uses fixed dimensions (no live resize), but those dimensions
        # are derived from the screen rather than hard-coded.
        metrics = _screen_metrics(None)
        # Floor at 320×240 so the splash survives even a tablet-class
        # display, ceiling at 640×420 on a desktop.
        target_w = min(640, max(320, int(metrics.width * 0.55)))
        target_h = min(420, max(240, int(metrics.height * 0.55)))
        # Tier the portrait by available screen + splash size.
        if metrics.width < 700 or target_w < 400:
            portrait_tier = "tiny"
        elif metrics.width < 900 or target_w < 560:
            portrait_tier = "compact"
        else:
            portrait_tier = "full"

        frame = QFrame(self)
        frame.setObjectName("splash_frame")
        frame.setFixedSize(target_w, target_h)

        outer = QHBoxLayout(frame)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(24)

        # ── Portrait column ──────────────────────────────────────────────────
        portrait_col = QVBoxLayout()
        portrait_col.setSpacing(4)

        portrait_text = {
            "full": PROF_PORTRAIT,
            "compact": PROF_PORTRAIT_SMALL,
            "tiny": PROF_PORTRAIT_TINY,
        }[portrait_tier]
        portrait = QLabel(portrait_text)
        portrait.setObjectName("prof_portrait")
        portrait.setTextFormat(Qt.TextFormat.PlainText)
        portrait.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        portrait_col.addWidget(portrait)

        name = QLabel("PROF. ANNOTATE")
        name.setObjectName("prof_name")
        name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        portrait_col.addWidget(name)
        portrait_col.addStretch(1)

        portrait_wrap = QWidget()
        portrait_wrap.setLayout(portrait_col)
        portrait_wrap.setFixedWidth({"full": 240, "compact": 150, "tiny": 96}[portrait_tier])
        outer.addWidget(portrait_wrap)

        # ── Branding column ──────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addStretch(1)

        title = QLabel(APP_NAME.upper())
        title.setObjectName("splash_title")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right.addWidget(title)

        subtitle = QLabel(SPLASH_LINES[0])
        subtitle.setObjectName("splash_subtitle")
        right.addWidget(subtitle)

        spacer = QLabel("")
        spacer.setFixedHeight(6)
        right.addWidget(spacer)

        tagline = QLabel(SPLASH_LINES[1])
        tagline.setObjectName("splash_tagline")
        tagline.setWordWrap(True)
        right.addWidget(tagline)

        prof_line = QLabel(SPLASH_LINES[2])
        prof_line.setObjectName("splash_tagline")
        prof_line.setWordWrap(True)
        right.addWidget(prof_line)

        right.addStretch(2)

        version = QLabel(f"v{APP_VERSION}  ·  ARCANE WORKBENCH")
        version.setObjectName("splash_version")
        right.addWidget(version)

        outer.addLayout(right, stretch=1)

        root.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(frame.size())

    def _center_on_screen(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        screen = app.primaryScreen()
        if screen is None:
            return
        geom = screen.geometry()
        self.move(
            geom.center().x() - self.width() // 2,
            geom.center().y() - self.height() // 2,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self, on_dismiss: Callable[[], None]) -> None:
        """Show the splash and call `on_dismiss()` after the duration elapses
        (or immediately on click). Fired exactly once."""
        self._dismiss_cb = on_dismiss
        # Prof. travels onto the splash — his workshop section (if it exists
        # yet) goes to its 'absent' state.
        presence().summon()
        self.show()
        self.raise_()
        QApplication.processEvents()
        QTimer.singleShot(self._duration_ms, self._dismiss)

    def _dismiss(self) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        cb = self._dismiss_cb
        self._dismiss_cb = None
        self.close()
        presence().release()
        if cb is not None:
            cb()

    def mousePressEvent(self, event) -> None:  # noqa: D401
        self._dismiss()
