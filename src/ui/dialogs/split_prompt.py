"""
bytemark/ui/dialogs/split_prompt.py
Train/val split ratio input dialog — Prof. asks for the split.
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

from src.ui.dialogs._prof_layout import build_prof_column, screen_aware_size


class SplitPrompt(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        chosen_w = screen_aware_size(frame, preferred_w=580, min_w=340, parent=parent)
        frame.setMinimumWidth(chosen_w)

        outer_h = QHBoxLayout(frame)
        outer_h.setContentsMargins(22, 20, 22, 20)
        outer_h.setSpacing(18)

        prof_col, _ = build_prof_column(parent, size="compact")
        outer_h.addWidget(prof_col)

        inner = QVBoxLayout()
        inner.setSpacing(12)

        t = QLabel("Split the dataset, Annotator.")
        t.setObjectName("dialog_title")
        t.setAlignment(Qt.AlignmentFlag.AlignLeft)
        inner.addWidget(t)

        body = QLabel(
            "Enter your desired train/val split distribution\n"
            "in percentage (e.g. 80/20 or 85/15)."
        )
        body.setObjectName("dialog_body")
        body.setAlignment(Qt.AlignmentFlag.AlignLeft)
        body.setWordWrap(True)
        inner.addWidget(body)

        # Train / Val inputs
        row = QHBoxLayout()
        row.setSpacing(20)

        train_col = QVBoxLayout()
        train_col.addWidget(QLabel("> Train %:"))
        self._train_input = QLineEdit("80")
        self._train_input.setValidator(QIntValidator(1, 99))
        self._train_input.setFixedWidth(70)
        self._train_input.textChanged.connect(self._sync_val)
        train_col.addWidget(self._train_input)
        row.addLayout(train_col)

        val_col = QVBoxLayout()
        val_col.addWidget(QLabel("Val %:"))
        self._val_input = QLineEdit("20")
        self._val_input.setValidator(QIntValidator(1, 99))
        self._val_input.setReadOnly(True)
        self._val_input.setFixedWidth(70)
        val_col.addWidget(self._val_input)
        row.addLayout(val_col)
        row.addStretch(1)

        inner.addLayout(row)

        self._error_lbl = QLabel("")
        self._error_lbl.setObjectName("accent_red")
        self._error_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        inner.addWidget(self._error_lbl)

        inner.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("arcane_button_dim")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("> Proceed")
        ok_btn.setObjectName("arcane_button")
        ok_btn.setDefault(True)
        ok_btn.setAutoDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        inner.addLayout(btn_row)
        self._focus_target = ok_btn

        outer_h.addLayout(inner, stretch=1)

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

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        self._focus_target.setFocus(Qt.FocusReason.OtherFocusReason)
