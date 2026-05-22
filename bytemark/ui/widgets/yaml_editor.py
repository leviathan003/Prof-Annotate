"""
bytemark/ui/widgets/yaml_editor.py
data.yaml mini editor with YAML syntax highlighting.

Now editable — tracks dirty state, shows a `*` in the header when unsaved,
saves on Ctrl+S, and autosaves to disk on a 2-second debounce so a crash or
abrupt close still flushes the most recent edits.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QShortcut,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)
from PySide6.QtWidgets import QFrame, QLabel, QPlainTextEdit, QVBoxLayout

logger = logging.getLogger(__name__)

_AUTOSAVE_DEBOUNCE_MS = 2000


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
    yaml_saved = Signal(str)  # emits the saved text after a successful write
    yaml_dirty_changed = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("json_panel")
        self._yaml_path: Optional[Path] = None
        self._saved_text: str = ""
        self._dirty: bool = False
        self._suppress_change = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = QLabel("DATA.YAML")
        self._header.setObjectName("section_header")
        layout.addWidget(self._header)

        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText("# data.yaml will appear here")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._editor.setUndoRedoEnabled(True)
        self._editor.textChanged.connect(self._on_text_changed)
        layout.addWidget(self._editor)

        self._highlighter = _YamlHighlighter(self._editor.document())

        # Ctrl+S — scoped to this widget so it doesn't conflict with the canvas
        # save shortcut. ApplicationShortcut would steal focus; WidgetShortcut
        # fires when the editor has keyboard focus.
        save_sc = QShortcut(QKeySequence("Ctrl+S"), self._editor)
        save_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        save_sc.activated.connect(self.save)

        # Debounced autosave so a crash or abrupt close doesn't lose edits.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(_AUTOSAVE_DEBOUNCE_MS)
        self._autosave_timer.timeout.connect(self._autosave_to_disk)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, yaml_path: Path) -> None:
        self._yaml_path = yaml_path
        text = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
        self._suppress_change = True
        self._editor.setPlainText(text)
        self._suppress_change = False
        self._saved_text = text
        self._set_dirty(False)
        self._autosave_timer.stop()

    def save(self) -> bool:
        """Explicit (Ctrl+S) save. Writes text to disk and clears the dirty flag."""
        if self._yaml_path is None:
            return False
        text = self._editor.toPlainText()
        try:
            self._yaml_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.error("yaml save failed for %s: %s", self._yaml_path, exc)
            return False
        self._saved_text = text
        self._set_dirty(False)
        self._autosave_timer.stop()
        self.yaml_saved.emit(text)
        return True

    def flush_if_dirty(self) -> bool:
        """Used by closeEvent — write pending edits to disk before exit."""
        if not self._dirty or self._yaml_path is None:
            return False
        return self.save()

    def clear(self) -> None:
        self._yaml_path = None
        self._suppress_change = True
        self._editor.clear()
        self._suppress_change = False
        self._saved_text = ""
        self._set_dirty(False)
        self._autosave_timer.stop()

    def is_dirty(self) -> bool:
        return self._dirty

    def get_text(self) -> str:
        return self._editor.toPlainText()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _on_text_changed(self) -> None:
        if self._suppress_change:
            return
        current = self._editor.toPlainText()
        self._set_dirty(current != self._saved_text)
        if self._dirty:
            self._autosave_timer.start()  # restart debounce on every keystroke

    def _autosave_to_disk(self) -> None:
        """Background autosave — same as save() but doesn't emit yaml_saved
        (callers like main_window often only care about explicit Ctrl+S)."""
        if not self._dirty or self._yaml_path is None:
            return
        text = self._editor.toPlainText()
        try:
            self._yaml_path.write_text(text, encoding="utf-8")
            self._saved_text = text
            self._set_dirty(False)
        except OSError as exc:
            logger.error("yaml autosave failed for %s: %s", self._yaml_path, exc)

    def _set_dirty(self, dirty: bool) -> None:
        if dirty == self._dirty:
            return
        self._dirty = dirty
        self._header.setText("DATA.YAML *" if dirty else "DATA.YAML")
        self.yaml_dirty_changed.emit(dirty)
