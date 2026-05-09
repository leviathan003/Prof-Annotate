"""
bytemark/ui/drawing/bbox_tool.py
BBox drawing tool. Activated with 'B'. Click + drag to draw.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QPen
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsScene

from bytemark.core.annotation.models import BBox
from bytemark.utils.color import bbox_color


class BBoxTool:
    def __init__(
        self,
        scene: QGraphicsScene,
        img_w: int,
        img_h: int,
        on_complete: Callable[[BBox, int], None],
        class_id: int = 0,
    ) -> None:
        self._scene = scene
        self._img_w = img_w
        self._img_h = img_h
        self._on_complete = on_complete
        self._class_id = class_id
        self._start: Optional[QPointF] = None
        self._preview: Optional[QGraphicsRectItem] = None
        self._active = False

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False
        self._cancel()

    def is_active(self) -> bool:
        return self._active

    def mouse_press(self, scene_pos: QPointF) -> bool:
        if not self._active:
            return False
        self._start = scene_pos
        self._preview = self._scene.addRect(
            scene_pos.x(),
            scene_pos.y(),
            0,
            0,
            QPen(bbox_color(), 1.5, Qt.PenStyle.DashLine),
            QBrush(Qt.BrushStyle.NoBrush),
        )
        return True

    def mouse_move(self, scene_pos: QPointF) -> bool:
        if not self._active or self._start is None or self._preview is None:
            return False
        x = min(self._start.x(), scene_pos.x())
        y = min(self._start.y(), scene_pos.y())
        w = abs(scene_pos.x() - self._start.x())
        h = abs(scene_pos.y() - self._start.y())
        self._preview.setRect(x, y, w, h)
        return True

    def mouse_release(self, scene_pos: QPointF) -> bool:
        if not self._active or self._start is None:
            return False
        x1 = min(self._start.x(), scene_pos.x())
        y1 = min(self._start.y(), scene_pos.y())
        x2 = max(self._start.x(), scene_pos.x())
        y2 = max(self._start.y(), scene_pos.y())

        # Ignore tiny accidental drags
        if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
            self._cancel()
            return True

        self._cancel()
        bbox = BBox.from_xyxy(x1, y1, x2, y2, self._img_w, self._img_h)
        self._on_complete(bbox, self._class_id)
        return True

    def _cancel(self) -> None:
        if self._preview is not None:
            self._scene.removeItem(self._preview)
            self._preview = None
        self._start = None
