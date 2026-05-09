"""
bytemark/ui/dialogs/split_prompt.py
Train/val split ratio input dialog.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class SplitPrompt(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        frame.setFixedWidth(480)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(14)

        t = QLabel("Split the dataset!")
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(t)

        body = QLabel(
            "Annotator, please enter your desired\n"
            "train/val split distribution in percentage\n"
            "(e.g. 80/20 or 85/15)"
        )
        body.setObjectName("dialog_body")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(body)

        # Train / Val inputs
        row = QHBoxLayout()
        row.setSpacing(20)

        train_col = QVBoxLayout()
        train_col.addWidget(QLabel("> Train:"))
        self._train_input = QLineEdit("80")
        self._train_input.setValidator(QIntValidator(1, 99))
        self._train_input.setFixedWidth(60)
        self._train_input.textChanged.connect(self._sync_val)
        train_col.addWidget(self._train_input)
        row.addLayout(train_col)

        val_col = QVBoxLayout()
        val_col.addWidget(QLabel("Val:"))
        self._val_input = QLineEdit("20")
        self._val_input.setValidator(QIntValidator(1, 99))
        self._val_input.setReadOnly(True)
        self._val_input.setFixedWidth(60)
        val_col.addWidget(self._val_input)
        row.addLayout(val_col)

        inner.addLayout(row)

        self._error_lbl = QLabel("")
        self._error_lbl.setObjectName("accent_red")
        self._error_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self._error_lbl)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("> Proceed")
        ok_btn.setObjectName("primary_button")
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        inner.addLayout(btn_row)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def _sync_val(self, train_text: str) -> None:
        try:
            train = int(train_text)
            self._val_input.setText(str(100 - train))
        except ValueError:
            self._val_input.setText("")

    def _on_ok(self) -> None:
        try:
            train = int(self._train_input.text())
            if not 1 <= train <= 99:
                raise ValueError()
            self._error_lbl.setText("")
            self.accept()
        except ValueError:
            self._error_lbl.setText("Enter a value between 1 and 99.")

    def train_ratio(self) -> float:
        try:
            return int(self._train_input.text()) / 100.0
        except ValueError:
            return 0.8
