"""
main.py — ByteMark entry point.
"""

import os
import sys
from pathlib import Path


def _setup_paths() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    _setup_paths()
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase, QIcon
    from PySide6.QtWidgets import QApplication

    from bytemark.config.constants import (
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

    # Load Ubuntu Sans Mono — variable font + static variants
    ubuntu_mono_dir = FONTS_DIR / "Ubuntu_Sans_Mono"
    font_family = None

    if ubuntu_mono_dir.exists():
        # Load variable fonts first, then static fallbacks
        for ttf in sorted(ubuntu_mono_dir.rglob("*.ttf")):
            fid = QFontDatabase.addApplicationFont(str(ttf))
            if fid != -1 and font_family is None:
                families = QFontDatabase.applicationFontFamilies(fid)
                if families:
                    font_family = families[0]

    if font_family:
        app.setFont(QFont(font_family, 11))
    else:
        # System fallback stack — both Linux and Windows have at least one
        for family in (
            "Ubuntu Sans Mono",
            "Cascadia Code",
            "JetBrains Mono",
            "Consolas",
            "Courier New",
        ):
            if QFontDatabase.hasFamily(family):
                app.setFont(QFont(family, 11))
                break

    # Apply QSS theme
    qss_path = STYLES_DIR / "dark_theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    icon_path = ICONS_DIR / "bytemark.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    from bytemark.ui.main_window import MainWindow

    window = MainWindow()
    window.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
