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
    QPushButton,
    QTreeView,
    QVBoxLayout,
)

from bytemark.config.constants import (
    COLOR_ANNOTATED,
    COLOR_PARTIAL,
    COLOR_UNANNOTATED,
)
from bytemark.core.dataset.loader import DatasetIndex, ImageEntry

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

    def mark_unsaved(self, image_path: str) -> None:
        self._unsaved.add(image_path)
        self._emit_data_changed()

    def mark_saved(self, image_path: str) -> None:
        self._unsaved.discard(image_path)
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
                return f"|--{name}  [{len(entries)}]"
            if role == Qt.ItemDataRole.ForegroundRole:
                return QBrush(QColor("#555555"))
            if role == Qt.ItemDataRole.FontRole:
                f = QFont()
                f.setWeight(QFont.Weight.Medium)
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
    open_folder_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar_panel")
        self._model: Optional[_FileModel] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        hdr = QLabel("FILE EXPLORER")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        # Root label
        self._root_label = QLabel("root")
        self._root_label.setObjectName("dimmed")
        self._root_label.setContentsMargins(8, 4, 8, 2)
        layout.addWidget(self._root_label)

        # Tree view
        self._tree = QTreeView()
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(False)
        self._tree.setIndentation(12)
        self._tree.setUniformRowHeights(True)
        self._tree.clicked.connect(self._on_clicked)
        layout.addWidget(self._tree, stretch=1)

        # Open folder button
        self._open_btn = QPushButton("Open Folder")
        self._open_btn.setObjectName("primary_button")
        self._open_btn.setContentsMargins(8, 4, 8, 4)
        self._open_btn.clicked.connect(self.open_folder_requested)
        layout.addWidget(self._open_btn)

    def load_index(self, index: DatasetIndex) -> None:
        self._root_label.setText(index.root.name)
        self._model = _FileModel(index, self)
        self._tree.setModel(self._model)
        self._tree.expandAll()

    def mark_unsaved(self, image_path: str) -> None:
        if self._model:
            self._model.mark_unsaved(image_path)

    def mark_saved(self, image_path: str) -> None:
        if self._model:
            self._model.mark_saved(image_path)

    def clear(self) -> None:
        self._model = None
        self._tree.setModel(None)
        self._root_label.setText("root")

    def _on_clicked(self, index: QModelIndex) -> None:
        if self._model is None:
            return
        entry = self._model.get_entry(index)
        if entry is not None:
            self.file_selected.emit(entry)
