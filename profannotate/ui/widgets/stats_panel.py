"""
profannotate/ui/widgets/stats_panel.py
Event-driven dataset statistics panel.

The cheap counters (total/train/val/annotated/corrupted) come straight from the
frozen DatasetIndex caches and render immediately on every `set_index` call
(which main_window fires on dataset load and on every save). Only the class
distribution needs a label-file walk — that runs on a background thread behind
a debounce, never on a wall-clock poll.
"""

from __future__ import annotations

from pathlib import Path
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

from profannotate.core.dataset.loader import DatasetIndex


class _ClassScanWorker(QObject):
    """Walks the labels tree once to build the class distribution."""

    result_ready = Signal(dict)

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root

    def compute(self) -> None:
        from collections import Counter

        from profannotate.config.constants import YOLO_LABEL_EXT, YOLO_LABELS_SUBDIR

        counts: Counter = Counter()
        lbl_dir = self._root / YOLO_LABELS_SUBDIR
        if lbl_dir.exists():
            for lbl in lbl_dir.rglob(f"*{YOLO_LABEL_EXT}"):
                try:
                    for line in lbl.read_text(errors="ignore").splitlines():
                        # Only the first field (class id) matters — don't split
                        # the whole 400-field pose line.
                        parts = line.split(None, 1)
                        if parts:
                            try:
                                counts[int(parts[0])] += 1
                            except ValueError:
                                pass
                except OSError:
                    pass
        self.result_ready.emit(dict(counts))


class StatsPanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("stats_panel")
        self._index: Optional[DatasetIndex] = None
        self._class_counts: dict = {}
        self._scanned_root: Optional[Path] = None
        # Single-shot debounce: bursts of saves collapse into one label scan.
        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._rescan_timer.setInterval(5000)
        self._rescan_timer.timeout.connect(self._start_class_scan)
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

    def set_index(self, index: Optional[DatasetIndex]) -> None:
        self._index = index
        if index is None:
            return
        if index.root != self._scanned_root:
            # New dataset: stale class counts must not bleed across.
            self._scanned_root = index.root
            self._class_counts = {}
            self._render(index)
            self._start_class_scan()
        else:
            # ponytail: full label rescan per save-burst; switch to per-file
            # delta updates only if this ever shows up on a profile.
            self._render(index)
            self._rescan_timer.start()

    def clear(self) -> None:
        self._rescan_timer.stop()
        self._index = None
        self._clear_rows()

    def shutdown(self) -> None:
        """Stop pending rescans and wait out an in-flight scan thread — a
        QThread wrapper GC'd while its thread runs hard-aborts the process."""
        self._rescan_timer.stop()
        thread = self._thread
        if thread is None:
            return
        try:
            thread.quit()
            thread.wait(3000)
        except RuntimeError:
            pass  # already deleted via deleteLater

    def _render(self, idx: DatasetIndex) -> None:
        """Rebuild the rows from the frozen index caches (all O(1)) plus the
        last known class distribution."""
        total = idx.total
        train = len(idx.train_entries)
        val = len(idx.val_entries)
        ann = idx.annotated_count
        self._apply_stats(
            {
                "total": total,
                "train": train,
                "val": val,
                "train_pct": round(train / total * 100) if total else 0,
                "val_pct": round(val / total * 100) if total else 0,
                "annotated": ann,
                "annotated_pct": round(ann / total * 100) if total else 0,
                "unannotated": total - ann,
                "corrupted": idx.corrupted_count,
                "class_counts": self._class_counts,
            }
        )

    def _start_class_scan(self) -> None:
        if self._index is None:
            return
        if self._thread is not None:
            # A scan is already running — re-arm so this request isn't lost.
            self._rescan_timer.start()
            return
        self._thread = QThread()
        self._worker = _ClassScanWorker(self._index.root)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.compute)
        self._worker.result_ready.connect(self._on_class_counts)
        self._worker.result_ready.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.start()

    def _on_class_counts(self, counts: dict) -> None:
        self._class_counts = counts
        if self._index is not None:
            self._render(self._index)

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
