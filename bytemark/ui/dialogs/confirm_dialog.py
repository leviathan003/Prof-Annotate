"""
bytemark/ui/dialogs/confirm_dialog.py
Generic confirmation overlay dialog in terminal style.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ConfirmDialog(QDialog):
    def __init__(
        self,
        title: str,
        body: str,
        confirm_text: str = "> Confirm, proceed",
        cancel_text: str = "No, cancel",
        parent=None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        frame.setFixedWidth(500)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(16)

        t = QLabel(title)
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(t)

        b = QLabel(body)
        b.setObjectName("dialog_body")
        b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b.setWordWrap(True)
        inner.addWidget(b)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(24)

        confirm_btn = QPushButton(confirm_text)
        confirm_btn.setObjectName("primary_button")
        confirm_btn.clicked.connect(self.accept)

        cancel_btn = QPushButton(cancel_text)
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(confirm_btn)
        btn_row.addWidget(cancel_btn)
        inner.addLayout(btn_row)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)
