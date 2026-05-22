"""
bytemark/ui/widgets/file_explorer.py
Sidebar file tree — lazy-loaded, color-coded by annotation status.
Green = annotated, Yellow = unsaved/partial, Red = no annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    QObject,
    Qt,
    QThread,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QStackedWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from src.config.constants import (
    COLOR_ANNOTATED,
    COLOR_PARTIAL,
    COLOR_UNANNOTATED,
)
from src.core.dataset.loader import DatasetIndex, ImageEntry

_GROUP = 0xFFFFFFFFFFFFFFFF  # sentinel for top-level group nodes


class _FileModel(QAbstractItemModel):
    def __init__(self, index: DatasetIndex, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        self._groups: list[tuple[str, list[ImageEntry]]] = [
            ("train", index.train_entries),
            ("val", index.val_entries),
        ]
        self._unsaved: set[str] = set()
        # path -> ImageEntry, built once. The legacy mark_saved walked the
        # entire entries list per call (O(N) per save). With 10k images that
        # was 10k str comparisons every time the user hit Ctrl+S.
        self._entry_by_path: dict[str, ImageEntry] = {
            str(e.image_path): e for e in index.entries
        }

    def mark_unsaved(self, image_path: str) -> None:
        self._unsaved.add(image_path)
        self._emit_data_changed()

    def mark_saved(self, image_path: str, has_label: bool = True) -> None:
        self._unsaved.discard(image_path)
        # O(1) lookup — keeps the row's colour accurate after save.
        entry = self._entry_by_path.get(image_path)
        if entry is not None:
            entry.has_label = has_label
        self._emit_data_changed()

    def _emit_data_changed(self) -> None:
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, 0))

    def index(self, row: int, col: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, col, parent):
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, col, _GROUP)
        return self.createIndex(row, col, parent.row())

    def parent(self, child: QModelIndex) -> QModelIndex:
        if not child.isValid():
            return QModelIndex()
        if child.internalId() == _GROUP:
            return QModelIndex()
        return self.createIndex(int(child.internalId()), 0, _GROUP)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if not parent.isValid():
            return len(self._groups)
        if parent.internalId() == _GROUP:
            g = parent.row()
            return len(self._groups[g][1]) if 0 <= g < len(self._groups) else 0
        return 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 1

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        is_group = index.internalId() == _GROUP
        if is_group:
            if role == Qt.ItemDataRole.DisplayRole:
                name, entries = self._groups[index.row()]
                return f"|--{name.upper()}  [{len(entries)}]"
            if role == Qt.ItemDataRole.ForegroundRole:
                return QBrush(QColor("#E0E0E0"))
            if role == Qt.ItemDataRole.FontRole:
                f = QFont()
                f.setWeight(QFont.Weight.Bold)
                return f
            return None

        g = int(index.internalId())
        if not (0 <= g < len(self._groups)):
            return None
        entries = self._groups[g][1]
        if not (0 <= index.row() < len(entries)):
            return None
        entry = entries[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return f"|-{entry.image_path.name}"
        if role == Qt.ItemDataRole.ForegroundRole:
            path_str = str(entry.image_path)
            if path_str in self._unsaved:
                return QBrush(QColor(COLOR_PARTIAL))
            if entry.is_corrupted:
                return QBrush(QColor(COLOR_UNANNOTATED))
            if entry.has_label:
                return QBrush(QColor(COLOR_ANNOTATED))
            return QBrush(QColor(COLOR_UNANNOTATED))
        if role == Qt.ItemDataRole.UserRole:
            return entry
        return None

    def get_entry(self, index: QModelIndex) -> Optional[ImageEntry]:
        if not index.isValid() or index.internalId() == _GROUP:
            return None
        g = int(index.internalId())
        entries = self._groups[g][1]
        if 0 <= index.row() < len(entries):
            return entries[index.row()]
        return None


class FileExplorer(QFrame):
    file_selected = Signal(object)  # ImageEntry
    open_folder_requested = Signal()  # kept for external callers

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar_panel")
        self._model: Optional[_FileModel] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QLabel("FILE EXPLORER")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        # ── Root label ────────────────────────────────────────────────────────
        self._root_label = QLabel("")
        self._root_label.setObjectName("explorer_root")
        self._root_label.setContentsMargins(8, 4, 8, 2)
        layout.addWidget(self._root_label)

        # ── Stacked: placeholder hint  ←→  tree ───────────────────────────────
        self._stack = QStackedWidget()

        # Page 0 — empty placeholder shown before any dataset is opened
        hint_page = QWidget()
        hint_layout = QVBoxLayout(hint_page)
        hint_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_layout.setSpacing(6)

        # Page 1 — tree view
        self._tree = QTreeView()
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(False)
        self._tree.setIndentation(12)
        self._tree.setUniformRowHeights(True)
        self._tree.clicked.connect(self._on_clicked)

        self._stack.addWidget(hint_page)  # index 0
        self._stack.addWidget(self._tree)  # index 1
        self._stack.setCurrentIndex(0)

        layout.addWidget(self._stack, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_index(self, index: DatasetIndex) -> None:
        self._root_label.setText(index.root.name)
        self._model = _FileModel(index, self)
        self._tree.setModel(self._model)
        self._tree.expandAll()
        self._stack.setCurrentIndex(1)

    def mark_unsaved(self, image_path: str) -> None:
        if self._model:
            self._model.mark_unsaved(image_path)

    def mark_saved(self, image_path: str, has_label: bool = True) -> None:
        if self._model:
            self._model.mark_saved(image_path, has_label)

    def clear(self) -> None:
        self._model = None
        self._tree.setModel(None)
        self._root_label.setText("")
        self._stack.setCurrentIndex(0)

    def _on_clicked(self, index: QModelIndex) -> None:
        if self._model is None:
            return
        entry = self._model.get_entry(index)
        if entry is not None:
            self.file_selected.emit(entry)
