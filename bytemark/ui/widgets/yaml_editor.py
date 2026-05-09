"""
bytemark/ui/widgets/yaml_editor.py
data.yaml mini editor with YAML syntax highlighting.
Auto-creates data.yaml if missing when a dataset is loaded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)
from PySide6.QtWidgets import QFrame, QLabel, QPlainTextEdit, QVBoxLayout


class _YamlHighlighter(QSyntaxHighlighter):
    def __init__(self, doc: QTextDocument) -> None:
        super().__init__(doc)

        def fmt(color: str, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            return f

        self._rules = [
            (re.compile(r"^[^:]+:"), fmt("#9CDCFE")),  # keys
            (re.compile(r":\s*.+$"), fmt("#CE9178")),  # values
            (re.compile(r"#.*$"), fmt("#6A9955")),  # comments
            (re.compile(r"^\s*-\s"), fmt("#FFD700")),  # list items
            (re.compile(r"\b\d+\.?\d*\b"), fmt("#B5CEA8")),  # numbers
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class YamlEditor(QFrame):
    yaml_saved = Signal(str)  # emits yaml text on Ctrl+S

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("json_panel")
        self._yaml_path: Optional[Path] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("DATA.YAML")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText("# data.yaml will appear here")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._editor)

        self._highlighter = _YamlHighlighter(self._editor.document())

    def load(self, yaml_path: Path) -> None:
        self._yaml_path = yaml_path
        if yaml_path.exists():
            self._editor.setPlainText(yaml_path.read_text(encoding="utf-8"))
        else:
            self._editor.setPlainText("")

    def save(self) -> bool:
        if self._yaml_path is None:
            return False
        try:
            self._yaml_path.write_text(self._editor.toPlainText(), encoding="utf-8")
            self.yaml_saved.emit(self._editor.toPlainText())
            return True
        except OSError:
            return False

    def clear(self) -> None:
        self._yaml_path = None
        self._editor.clear()

    def get_text(self) -> str:
        return self._editor.toPlainText()
