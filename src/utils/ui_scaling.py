"""
bytemark/utils/ui_scaling.py
Helpers that translate "screen size" into concrete pixel budgets for the
main window's splitters and Prof.'s portrait.

The goal is for ByteMark to feel native at every common form factor:

  - 800×540    (tiny tablet)
  - 1024×600   (small netbook)
  - 1280×720   (standard small laptop)
  - 1366×768   (mainstream laptop)
  - 1920×1080  (desktop)
  - 2560×1440+ (HiDPI desktop)

All helpers are pure and side-effect-free — callers apply the returned
sizes via the Qt widgets they own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtWidgets import QApplication, QWidget


# ── Screen sampling ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenInfo:
    width: int
    height: int


def screen_for(parent: Optional[QWidget] = None) -> ScreenInfo:
    """Return the width/height of the screen `parent` lives on (or the
    primary screen if none provided). Used at startup before the main
    window has been placed."""
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
        return ScreenInfo(width=1280, height=720)
    geom = screen.availableGeometry()
    return ScreenInfo(width=geom.width(), height=geom.height())


# ── Form-factor classification ───────────────────────────────────────────────


def form_factor(info: Optional[ScreenInfo] = None) -> str:
    """Bucket the active screen into one of:
        'tiny'     — under 900 px wide  (small tablets, tiny laptops)
        'small'    — 900-1200 px wide   (small laptops, netbooks)
        'medium'   — 1200-1600 px wide  (mainstream laptops)
        'large'    — 1600+ px wide      (desktops, HiDPI)
    """
    s = info or screen_for(None)
    if s.width < 900:
        return "tiny"
    if s.width < 1200:
        return "small"
    if s.width < 1600:
        return "medium"
    return "large"


# ── Splitter sizing ──────────────────────────────────────────────────────────


def horizontal_splitter_sizes(window_width: int) -> tuple[int, int, int]:
    """Return (sidebar_w, canvas_w, json_w) for the main 3-column splitter,
    proportional to the actual window width."""
    # Reserve fractions tuned to keep the canvas dominant at every size.
    if window_width >= 1600:
        sidebar_pct, json_pct = 0.16, 0.18
    elif window_width >= 1200:
        sidebar_pct, json_pct = 0.18, 0.20
    elif window_width >= 1000:
        sidebar_pct, json_pct = 0.20, 0.22
    else:
        # Tight: trim both sidebars further so the canvas still has room.
        sidebar_pct, json_pct = 0.21, 0.21

    sidebar = max(130, int(window_width * sidebar_pct))
    json_w = max(150, int(window_width * json_pct))
    canvas = max(280, window_width - sidebar - json_w - 12)  # 12 ≈ handle widths
    return sidebar, canvas, json_w


def right_splitter_sizes(window_height: int) -> tuple[int, int, int]:
    """Return (yaml_h, prof_h, json_h) for the right vertical splitter.
    Shorter windows give the JSON pane more share so it stays usable."""
    # Take roughly: top_bar 32 + status_bar 22 ≈ 54 px reserved.
    main_h = max(360, window_height - 54)
    if main_h >= 700:
        yaml_pct, prof_pct = 0.22, 0.25
    elif main_h >= 500:
        yaml_pct, prof_pct = 0.20, 0.22
    else:
        # Very short — keep prof minimal, JSON gets the rest.
        yaml_pct, prof_pct = 0.18, 0.20

    yaml_h = max(110, int(main_h * yaml_pct))
    prof_h = max(110, int(main_h * prof_pct))
    json_h = max(180, main_h - yaml_h - prof_h - 6)
    return yaml_h, prof_h, json_h


def left_column_split(panel_height: int) -> tuple[int, int]:
    """Return (file_explorer_h, stats_panel_h) for the left vertical column."""
    h = max(280, panel_height)
    stats_h = max(90, int(h * 0.30))
    explorer_h = max(160, h - stats_h)
    return explorer_h, stats_h


# ── Prof. portrait tier ──────────────────────────────────────────────────────


def prof_portrait_tier(
    *,
    available_height: int = 0,
    available_width: int = 0,
    screen_info: Optional[ScreenInfo] = None,
    prefer: str = "auto",
) -> str:
    """Pick which Prof portrait to render: 'full' | 'compact' | 'tiny'.

    `available_height` / `available_width` are the widget's allotted size
    (use 0 for "no constraint, decide from screen"). `prefer` lets the
    caller bias toward a tier — `"full"` will still downgrade if the
    surface or screen can't fit it.
    """
    # Hard ceilings: a portrait can't be taller than its container.
    if available_height and available_height < 110:
        return "tiny"
    if available_height and available_height < 200:
        return "compact"

    # Width matters too — a narrow column can't hold the full portrait.
    if available_width and available_width < 140:
        return "tiny"
    if available_width and available_width < 200 and prefer != "full":
        return "compact"

    # Otherwise defer to the active screen size.
    info = screen_info or screen_for(None)
    ff = form_factor(info)

    if prefer == "tiny":
        return "tiny"
    if prefer == "compact":
        return "compact"
    if prefer == "full":
        # Caller asked for full but downgrade on small screens.
        if ff in ("tiny",):
            return "compact"
        return "full"

    # "auto"
    if ff in ("tiny", "small"):
        return "compact"
    if ff == "medium":
        return "compact"
    return "full"


def portrait_column_width(tier: str) -> int:
    """Pixel width of the portrait column for a given tier."""
    return {"full": 230, "compact": 150, "tiny": 96}.get(tier, 150)
