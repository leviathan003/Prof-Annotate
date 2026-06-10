"""
profannotate/ui/dialogs/keybindings_dialog.py
Popup that lists every keyboard shortcut and its function.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from profannotate.config.shortcuts import SHORTCUT_GROUPS
from profannotate.ui.dialogs._prof_layout import _screen_metrics, screen_aware_size


class KeybindingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        chosen_w = screen_aware_size(frame, preferred_w=600, min_w=360, parent=parent)
        frame.setMinimumWidth(chosen_w)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(14)

        title = QLabel("Keybindings")
        title.setObjectName("dialog_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(title)

        subtitle = QLabel("All keyboard shortcuts at a glance, Annotator.")
        subtitle.setObjectName("dialog_body")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(subtitle)

        body_widget = QWidget()
        body_layout = QVBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(14)

        for group_name, items in SHORTCUT_GROUPS:
            body_layout.addWidget(self._build_group(group_name, items))

        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("keybindings_scroll")
        scroll.setWidget(body_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Scroll area height scales with the screen so it doesn't dominate
        # short displays nor waste space on tall ones.
        metrics = _screen_metrics(parent)
        scroll.setMaximumHeight(max(260, int(metrics.height * 0.6)))
        inner.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("> Close")
        close_btn.setObjectName("primary_button")
        close_btn.setDefault(True)
        close_btn.setAutoDefault(True)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        inner.addLayout(btn_row)
        self._focus_target = close_btn

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    @staticmethod
    def _build_group(name: str, items: list[tuple[str, str]]) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QLabel(name.upper())
        header.setObjectName("kb_group_header")
        layout.addWidget(header)

        grid_frame = QFrame()
        grid_frame.setObjectName("kb_group_frame")
        grid = QGridLayout(grid_frame)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)

        for row, (label, keys) in enumerate(items):
            name_lbl = QLabel(label)
            name_lbl.setObjectName("kb_name")
            key_lbl = QLabel(keys)
            key_lbl.setObjectName("kb_key")
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(name_lbl, row, 0)
            grid.addWidget(key_lbl, row, 1)

        layout.addWidget(grid_frame)
        return wrap

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        self._focus_target.setFocus(Qt.FocusReason.OtherFocusReason)
