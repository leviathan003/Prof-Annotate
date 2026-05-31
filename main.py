"""
main.py — ByteMark entry point.
"""

import os
import sys
from pathlib import Path


def _setup_paths() -> None:
    import sys
    from pathlib import Path

    if getattr(sys, "frozen", False):
        root = Path(sys._MEIPASS)
    else:
        root = Path(__file__).resolve().parent

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    _setup_paths()
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase, QIcon
    from PySide6.QtWidgets import QApplication

    from src.config.constants import (
        APP_NAME,
        APP_VERSION,
        FONTS_DIR,
        ICONS_DIR,
        STYLES_DIR,
        WINDOW_MIN_HEIGHT,
        WINDOW_MIN_WIDTH,
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    base_pt = 11
    screen = app.primaryScreen()
    if screen:
        dpi = screen.logicalDotsPerInch()
        if dpi >= 192:
            base_pt = 15
        elif dpi >= 144:
            base_pt = 13
        elif dpi >= 120:
            base_pt = 12
    scale = base_pt / 11.0

    # Load Ubuntu Sans Mono — only the variable fonts (one upright + one
    # italic). The variable font covers every weight, so loading the 8 static
    # TTFs as well is pure startup overhead. This trims 5–8× the font-parse
    # cost on cold launch.
    ubuntu_mono_dir = FONTS_DIR / "Ubuntu_Sans_Mono"
    font_family = None

    if ubuntu_mono_dir.exists():
        variable_fonts = [
            ubuntu_mono_dir / "UbuntuSansMono-VariableFont_wght.ttf",
            ubuntu_mono_dir / "UbuntuSansMono-Italic-VariableFont_wght.ttf",
        ]
        for ttf in variable_fonts:
            if not ttf.exists():
                continue
            fid = QFontDatabase.addApplicationFont(str(ttf))
            if fid != -1 and font_family is None:
                families = QFontDatabase.applicationFontFamilies(fid)
                if families:
                    font_family = families[0]

    base_font = None
    if font_family:
        base_font = QFont(font_family, base_pt)
    else:
        for family in (
            "Ubuntu Sans Mono",
            "Cascadia Code",
            "JetBrains Mono",
            "Consolas",
            "Courier New",
        ):
            if QFontDatabase.hasFamily(family):
                base_font = QFont(family, base_pt)
                break
    if base_font is not None:
        base_font.setWeight(QFont.Weight.Medium)
        app.setFont(base_font)

    # Apply QSS theme — scale font-size declarations to match the chosen base.
    qss_path = STYLES_DIR / "dark_theme.qss"
    if qss_path.exists():
        import re

        qss_text = qss_path.read_text(encoding="utf-8")

        def _scale_px(match: "re.Match[str]") -> str:
            px = int(match.group(1))
            return f"font-size: {max(8, round(px * scale))}px"

        qss_text = re.sub(r"font-size:\s*(\d+)px", _scale_px, qss_text)
        app.setStyleSheet(qss_text)

    icon_path = ICONS_DIR / "bytemark.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    from src.ui.dialogs.splash_screen import SplashScreen
    from src.ui.main_window import MainWindow
    from src.ui.prof_watcher import install_prof_watcher

    # Holds a hard reference so Qt doesn't garbage-collect the filter.
    _prof_watcher = install_prof_watcher(app)  # noqa: F841

    window = MainWindow()
    window.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

    splash = SplashScreen(duration_ms=1400)

    def _reveal_main_window() -> None:
        window.show()
        window.raise_()
        window.activateWindow()
        # Tutorial fires only on first run (or via the in-app replay button).
        window.maybe_show_first_run_tutorial()

    splash.start(on_dismiss=_reveal_main_window)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
