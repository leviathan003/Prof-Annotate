"""
bytemark/ui/widgets/stats_panel.py
Hot-reloading dataset statistics panel.
Runs stat computation on a background thread, updates UI on result.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from bytemark.config.constants import STATS_RELOAD_INTERVAL_MS
from bytemark.core.dataset.loader import DatasetIndex


class _StatsWorker(QObject):
    result_ready = Signal(dict)

    def __init__(self, index: DatasetIndex) -> None:
        super().__init__()
        self._index = index

    def compute(self) -> None:
        idx = self._index
        total = idx.total
        train = len(idx.train_entries)
        val = len(idx.val_entries)
        ann = idx.annotated_count
        corr = idx.corrupted_count
        unann = total - ann

        train_pct = round(train / total * 100) if total else 0
        val_pct = round(val / total * 100) if total else 0
        ann_pct = round(ann / total * 100) if total else 0

        # Class distribution from label files
        from collections import Counter

        class_counts: Counter = Counter()
        from bytemark.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR

        lbl_dir = idx.root / YOLO_LABELS_SUBDIR
        if lbl_dir.exists():
            for lbl in lbl_dir.rglob(f"*{YOLO_LABEL_EXT}"):
                try:
                    for line in lbl.read_text().splitlines():
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_counts[int(parts[0])] += 1
                            except ValueError:
                                pass
                except OSError:
                    pass

        self.result_ready.emit(
            {
                "total": total,
                "train": train,
                "val": val,
                "train_pct": train_pct,
                "val_pct": val_pct,
                "annotated": ann,
                "annotated_pct": ann_pct,
                "unannotated": unann,
                "corrupted": corr,
                "class_counts": dict(class_counts),
            }
        )


class StatsPanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("stats_panel")
        self._index: Optional[DatasetIndex] = None
        self._timer = QTimer(self)
        self._timer.setInterval(STATS_RELOAD_INTERVAL_MS)
        self._timer.timeout.connect(self._schedule_update)
        self._thread: Optional[QThread] = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        hdr = QLabel("DATASET STATS")
        hdr.setObjectName("section_header")
        root_layout.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 6, 8, 6)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch()

        scroll.setWidget(self._content)
        root_layout.addWidget(scroll)

        self._rows: dict[str, QLabel] = {}

    def set_index(self, index: DatasetIndex) -> None:
        self._index = index
        self._schedule_update()
        self._timer.start()

    def clear(self) -> None:
        self._timer.stop()
        self._index = None
        self._clear_rows()

    def _schedule_update(self) -> None:
        if self._index is None or self._thread is not None:
            return
        self._thread = QThread()
        self._worker = _StatsWorker(self._index)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.compute)
        self._worker.result_ready.connect(self._apply_stats)
        self._worker.result_ready.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.start()

    def _apply_stats(self, data: dict) -> None:
        self._clear_rows()
        rows = [
            (
                "Train/Val Split",
                f"{data['train']} / {data['val']}  ({data['train_pct']}% / {data['val_pct']}%)",
                "neutral",
            ),
            ("Total Images", str(data["total"]), "neutral"),
            ("Annotated", f"{data['annotated']}  ({data['annotated_pct']}%)", "green"),
            ("Unannotated", str(data["unannotated"]), "red" if data["unannotated"] else "neutral"),
            ("Corrupted", str(data["corrupted"]), "red" if data["corrupted"] else "neutral"),
        ]
        for cls_id, count in sorted(data["class_counts"].items()):
            rows.append((f"Class {cls_id}", str(count), "neutral"))

        for key, val, color in rows:
            self._add_row(key, val, color)

    def _add_row(self, key: str, value: str, color: str = "neutral") -> None:
        row = QHBoxLayout()
        row.setSpacing(4)

        k = QLabel(f"• {key}")
        k.setObjectName("stat_key")

        color_map = {
            "green": "stat_value_green",
            "yellow": "stat_value_yellow",
            "red": "stat_value_red",
            "neutral": "stat_value",
        }
        v = QLabel(value)
        v.setObjectName(color_map.get(color, "stat_value"))
        v.setAlignment(Qt.AlignmentFlag.AlignRight)

        row.addWidget(k)
        row.addStretch()
        row.addWidget(v)

        container = QWidget()
        container.setLayout(row)
        # Insert before the stretch
        self._content_layout.insertWidget(self._content_layout.count() - 1, container)
        self._rows[key] = v

    def _clear_rows(self) -> None:
        while self._content_layout.count() > 1:
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()
