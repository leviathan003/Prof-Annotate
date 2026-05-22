"""
bytemark/ui/dialogs/_prof_layout.py
Shared helpers for Prof.-flavored popups.

Two responsibilities:
 - `screen_aware_size(...)`  — clamp dialog dimensions to the active screen.
 - `build_prof_column(...)`  — assemble a portrait column whose size scales
   with the active screen (compact ASCII on small displays, full art on big).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from src.ui.prof_annotate import (
    PROF_PORTRAIT,
    PROF_PORTRAIT_SMALL,
    PROF_PORTRAIT_TINY,
)
from src.utils.ui_scaling import portrait_column_width, prof_portrait_tier


# ── Screen sampling ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ScreenMetrics:
    width: int
    height: int


def _screen_metrics(parent: Optional[QWidget]) -> _ScreenMetrics:
    """Return availableGeometry of the screen `parent` lives on (or the primary
    screen if `parent` is None / not yet placed)."""
    screen = None
    if parent is not None:
        win = parent.window() if parent else None
        if win is not None:
            handle = win.windowHandle()
            if handle is not None:
                screen = handle.screen()
        if screen is None and parent.screen() is not None:
            screen = parent.screen()
    if screen is None:
        app = QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
    if screen is None:
        # Fallback for headless tests.
        return _ScreenMetrics(width=1280, height=720)
    geom = screen.availableGeometry()
    return _ScreenMetrics(width=geom.width(), height=geom.height())


# ── Adaptive sizing ──────────────────────────────────────────────────────────


def screen_aware_size(
    widget: QWidget,
    *,
    preferred_w: int,
    min_w: int = 320,
    max_w_pct: float = 0.88,
    max_h_pct: float = 0.88,
    parent: Optional[QWidget] = None,
) -> int:
    """Clamp `widget`'s size to a fraction of the active screen.

    - `preferred_w` is the design width on a comfortable display.
    - `min_w` is the floor so the widget never collapses unreadably.
    - `max_w_pct` / `max_h_pct` cap the widget to a percentage of the screen.

    The returned int is the width the caller should apply (already clamped).
    Height is left to layout — caller only gets a maximum-height cap applied.
    """
    metrics = _screen_metrics(parent or widget.parent())
    max_w = max(min_w, int(metrics.width * max_w_pct))
    max_h = max(240, int(metrics.height * max_h_pct))

    chosen_w = max(min_w, min(preferred_w, max_w))
    chosen_min_w = min(min_w, max_w)

    widget.setMinimumWidth(chosen_min_w)
    widget.setMaximumWidth(max_w)
    widget.setMaximumHeight(max_h)
    return chosen_w


# ── Prof. portrait column ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProfColumnTuning:
    """How to scale Prof's portrait based on the active screen + size hint."""

    portrait_text: str
    column_width: int
    name_size_px: int
    show_subline: bool


def _tune_prof_column(
    parent: Optional[QWidget],
    requested_size: str,
) -> ProfColumnTuning:
    """Pick the right portrait variant + column width for the active screen."""
    m = _screen_metrics(parent)
    from src.utils.ui_scaling import ScreenInfo

    tier = prof_portrait_tier(
        screen_info=ScreenInfo(width=m.width, height=m.height),
        prefer=requested_size if requested_size in ("full", "compact", "tiny") else "auto",
    )

    if tier == "full":
        return ProfColumnTuning(
            portrait_text=PROF_PORTRAIT,
            column_width=portrait_column_width("full"),
            name_size_px=10,
            show_subline=True,
        )
    if tier == "compact":
        return ProfColumnTuning(
            portrait_text=PROF_PORTRAIT_SMALL,
            column_width=portrait_column_width("compact"),
            name_size_px=9,
            show_subline=False,
        )
    # tiny
    return ProfColumnTuning(
        portrait_text=PROF_PORTRAIT_TINY,
        column_width=portrait_column_width("tiny"),
        name_size_px=8,
        show_subline=False,
    )


def build_prof_column(
    parent: Optional[QWidget],
    *,
    size: str = "compact",
    name_text: str = "PROF. ANNOTATE",
) -> tuple[QWidget, int]:
    """Construct a fixed-width portrait column showing Prof.

    `size` is "full" (large portrait) or "compact" (small portrait). The
    helper may downgrade `full` to `compact` on small screens automatically.

    Returns `(column_widget, column_width_px)`.
    """
    tune = _tune_prof_column(parent, size)

    col_widget = QWidget()
    col_layout = QVBoxLayout(col_widget)
    col_layout.setContentsMargins(0, 0, 0, 0)
    col_layout.setSpacing(4)

    portrait = QLabel(tune.portrait_text)
    portrait.setObjectName("prof_portrait")
    portrait.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
    portrait.setTextFormat(Qt.TextFormat.PlainText)
    col_layout.addWidget(portrait)

    name = QLabel(name_text)
    name.setObjectName("prof_name")
    name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    col_layout.addWidget(name)

    if tune.show_subline:
        sub = QLabel("arcane workbench")
        sub.setObjectName("prof_subline")
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        col_layout.addWidget(sub)

    col_layout.addStretch(1)

    col_widget.setFixedWidth(tune.column_width)
    return col_widget, tune.column_width
