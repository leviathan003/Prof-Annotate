"""
bytemark/ui/dialogs/error_dialog.py
Error overlay dialog.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ErrorDialog(QDialog):
    def __init__(self, message: str, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        frame.setFixedWidth(460)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(16)

        title = QLabel("Error!")
        title.setObjectName("dialog_title")
        title.setStyleSheet("color: #FF4444;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(title)

        msg = QLabel(message)
        msg.setObjectName("dialog_body")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setWordWrap(True)
        inner.addWidget(msg)

        btn = QPushButton("> Try again")
        btn.setObjectName("danger_button")
        btn.clicked.connect(self.reject)
        inner.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)
