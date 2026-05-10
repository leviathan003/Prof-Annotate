"""
bytemark/ui/widgets/json_editor.py
Shows JSON for the currently selected annotation instance only.
"""

from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument
from PySide6.QtWidgets import QFrame, QLabel, QPlainTextEdit, QVBoxLayout

from bytemark.core.annotation.models import Annotation, ImageAnnotations


class _JsonHighlighter(QSyntaxHighlighter):
    def __init__(self, doc: QTextDocument) -> None:
        super().__init__(doc)
        import re

        def fmt(color: str, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            return f

        self._rules = [
            (re.compile(r'"[^"]*"\s*:'), fmt("#9CDCFE")),
            (re.compile(r':\s*"[^"]*"'), fmt("#CE9178")),
            (re.compile(r":\s*-?\d+\.?\d*"), fmt("#B5CEA8")),
            (re.compile(r"\b(true|false|null)\b"), fmt("#569CD6")),
            (re.compile(r"[{}\[\]]"), fmt("#FFD700", bold=True)),
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class JsonEditor(QFrame):
    annotation_edited = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("json_panel")
        self._annotations: Optional[ImageAnnotations] = None
        self._selected_idx: Optional[int] = None
        self._suppress_update = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("JSON EDITOR")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText("// Click an annotation to inspect it")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.textChanged.connect(self._on_user_edit)
        layout.addWidget(self._editor)

        self._highlighter = _JsonHighlighter(self._editor.document())

    def show_instance(self, ann: ImageAnnotations, idx: int) -> None:
        """Show JSON for a single selected instance."""
        self._annotations = ann
        self._selected_idx = idx
        self._refresh_display()

    def set_annotations(self, ann: ImageAnnotations) -> None:
        """Called on annotation changes — only refreshes if an instance is already selected."""
        self._annotations = ann
        if self._selected_idx is not None:
            self._refresh_display()

    def clear_selection(self) -> None:
        """Called when no instance is selected."""
        self._selected_idx = None
        self._suppress_update = True
        self._editor.clear()
        self._suppress_update = False

    def clear(self) -> None:
        self._annotations = None
        self._selected_idx = None
        self._suppress_update = True
        self._editor.clear()
        self._suppress_update = False

    def _refresh_display(self) -> None:
        if self._annotations is None or self._selected_idx is None:
            return
        instances = self._annotations.instances
        if self._selected_idx >= len(instances):
            return
        if self._editor.hasFocus():
            return
        try:
            data = _instance_to_dict(instances[self._selected_idx])
            text = json.dumps(data, indent=2)
            if text == self._editor.toPlainText():
                return
            self._suppress_update = True
            pos = self._editor.textCursor().position()
            self._editor.setPlainText(text)
            cursor = self._editor.textCursor()
            cursor.setPosition(min(pos, len(text)))
            self._editor.setTextCursor(cursor)
            self._suppress_update = False
        except Exception:
            pass

    def _on_user_edit(self) -> None:
        if self._suppress_update:
            return
        self.annotation_edited.emit(self._editor.toPlainText())

    def get_json_text(self) -> str:
        return self._editor.toPlainText()


def _instance_to_dict(inst: Annotation) -> dict:
    d: dict = {"class": inst.class_id}
    if inst.bbox:
        b = inst.bbox
        d["bbox"] = {
            "cx": round(b.cx, 6),
            "cy": round(b.cy, 6),
            "w": round(b.w, 6),
            "h": round(b.h, 6),
        }
    if inst.keypoints:
        d["keypoints"] = [
            {"x": round(k.x, 6), "y": round(k.y, 6), "v": k.visibility} if k is not None else None
            for k in inst.keypoints
        ]
    if inst.mask:
        d["mask"] = {"points": [[round(x, 6), round(y, 6)] for x, y in inst.mask.points]}
    return d
