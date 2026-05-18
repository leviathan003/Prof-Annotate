"""
bytemark/ui/widgets/status_bar.py
Bottom status bar: filename | dimensions | corrupted | annotated | git info.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget

from bytemark.core.git.reader import AnnotationCommit


class StatusBar(QStatusBar):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizeGripEnabled(False)

        # Single centered widget containing all fields
        container = QWidget()
        self._layout = QHBoxLayout(container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._filename = self._add_field("File")
        self._dims = self._add_field("Size")
        self._corrupted = self._add_field("Corrupted")
        self._annotated = self._add_field("Annotated")
        self._git_info = self._add_field("Last commit", last=True)

        # Stretch on both sides forces the container to the center
        self.addWidget(self._make_spacer(), 1)
        self.addWidget(container, 0)
        self.addWidget(self._make_spacer(), 1)

    def _make_spacer(self) -> QWidget:
        w = QWidget()
        w.setSizePolicy(w.sizePolicy().horizontalPolicy(), w.sizePolicy().verticalPolicy())
        from PySide6.QtWidgets import QSizePolicy

        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return w

    def _add_field(self, key: str, last: bool = False) -> QLabel:
        from PySide6.QtGui import QFont

        def _sized(text: str, obj_name: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName(obj_name)
            f = QFont()
            f.setPointSize(14)
            lbl.setFont(f)
            return lbl

        key_lbl = _sized(f" {key}: ", "status_field")
        val_lbl = _sized("—", "status_field_value")

        self._layout.addWidget(key_lbl)
        self._layout.addWidget(val_lbl)

        if not last:
            sep = _sized("  │  ", "status_field")
            self._layout.addWidget(sep)

        return val_lbl

    def update_image_info(
        self,
        filename: str,
        width: int,
        height: int,
        corrupted: bool,
        annotated: bool,
    ) -> None:
        self._filename.setText(filename)
        self._dims.setText(f"{width}×{height}")
        self._corrupted.setText("Yes" if corrupted else "No")
        self._corrupted.setStyleSheet("color: #FF4444;" if corrupted else "color: #888888;")
        self._annotated.setText("Yes" if annotated else "No")
        self._annotated.setStyleSheet("color: #00FF88;" if annotated else "color: #888888;")

    def update_git_info(self, commit: Optional[AnnotationCommit]) -> None:
        if commit is None:
            self._git_info.setText("—")
        else:
            self._git_info.setText(
                f"{commit.author}  {commit.timestamp[:10]}  [{commit.commit_hash}]"
            )

    def clear(self) -> None:
        for lbl in (self._filename, self._dims, self._corrupted, self._annotated, self._git_info):
            lbl.setText("—")
            lbl.setStyleSheet("")
