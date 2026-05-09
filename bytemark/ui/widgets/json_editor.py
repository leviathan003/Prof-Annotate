"""
bytemark/ui/widgets/json_editor.py
Hot-reloading JSON annotation editor with syntax highlighting.
Collapsible structure, editable for pixel-perfect adjustments.
"""

from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)
from PySide6.QtWidgets import QFrame, QLabel, QPlainTextEdit, QVBoxLayout

from bytemark.config.constants import JSON_RELOAD_INTERVAL_MS
from bytemark.core.annotation.models import ImageAnnotations


class _JsonHighlighter(QSyntaxHighlighter):
    def __init__(self, doc: QTextDocument) -> None:
        super().__init__(doc)
        self._rules: list[tuple] = []

        def fmt(color: str, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            return f

        import re

        # Keys
        self._rules.append((re.compile(r'"[^"]*"\s*:'), fmt("#9CDCFE")))
        # String values
        self._rules.append((re.compile(r':\s*"[^"]*"'), fmt("#CE9178")))
        # Numbers
        self._rules.append((re.compile(r":\s*-?\d+\.?\d*"), fmt("#B5CEA8")))
        # Booleans / null
        self._rules.append((re.compile(r"\b(true|false|null)\b"), fmt("#569CD6")))
        # Braces / brackets
        self._rules.append((re.compile(r"[{}\[\]]"), fmt("#FFD700", bold=True)))

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class JsonEditor(QFrame):
    annotation_edited = Signal(str)  # emits raw JSON string on user edit

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("json_panel")
        self._annotations: Optional[ImageAnnotations] = None
        self._suppress_update = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("JSON EDITOR")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText("// No annotation loaded")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.textChanged.connect(self._on_user_edit)
        layout.addWidget(self._editor)

        self._highlighter = _JsonHighlighter(self._editor.document())

        self._reload_timer = QTimer(self)
        self._reload_timer.setInterval(JSON_RELOAD_INTERVAL_MS)
        self._reload_timer.timeout.connect(self._hot_reload)

    def set_annotations(self, ann: Optional[ImageAnnotations]) -> None:
        self._annotations = ann
        self._refresh_display()
        self._reload_timer.start()

    def clear(self) -> None:
        self._annotations = None
        self._reload_timer.stop()
        self._suppress_update = True
        self._editor.clear()
        self._suppress_update = False

    def _refresh_display(self) -> None:
        if self._annotations is None:
            return
        if self._editor.hasFocus():
            return  # Don't clobber user's active edit
        try:
            data = _annotations_to_dict(self._annotations)
            text = json.dumps(data, indent=2)
            if text != self._editor.toPlainText():
                self._suppress_update = True
                cursor_pos = self._editor.textCursor().position()
                self._editor.setPlainText(text)
                cursor = self._editor.textCursor()
                cursor.setPosition(min(cursor_pos, len(text)))
                self._editor.setTextCursor(cursor)
                self._suppress_update = False
        except Exception:
            pass

    def _hot_reload(self) -> None:
        self._refresh_display()

    def _on_user_edit(self) -> None:
        if self._suppress_update:
            return
        self.annotation_edited.emit(self._editor.toPlainText())

    def get_json_text(self) -> str:
        return self._editor.toPlainText()


def _annotations_to_dict(ann: ImageAnnotations) -> dict:
    instances = []
    for inst in ann.instances:
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
                {"x": round(k.x, 6), "y": round(k.y, 6), "v": k.visibility} for k in inst.keypoints
            ]
        if inst.mask:
            d["mask"] = {"points": [[round(x, 6), round(y, 6)] for x, y in inst.mask.points]}
        instances.append(d)
    return {"instances": instances}
