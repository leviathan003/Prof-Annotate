"""
bytemark/ui/drawing/segmentation_tool.py
Single click = add point. Double click = close polygon.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsScene

from bytemark.core.annotation.models import SegmentationMask
from bytemark.utils.color import segmentation_color


class SegmentationTool:
    def __init__(
        self,
        scene: QGraphicsScene,
        img_w: int,
        img_h: int,
        on_complete: Callable[[SegmentationMask, int], None],
        class_id: int = 0,
    ) -> None:
        self._scene = scene
        self._img_w = img_w
        self._img_h = img_h
        self._on_complete = on_complete
        self._class_id = class_id
        self._points: list[tuple[float, float]] = []
        self._preview: Optional[QGraphicsLineItem] = None
        self._active = False

    def activate(self) -> None:
        self._active = True
        self._points = []

    def deactivate(self) -> None:
        self._active = False
        self._points = []
        self._remove_preview()

    def is_active(self) -> bool:
        return self._active

    def mouse_move(self, scene_pos: QPointF) -> bool:
        if not self._active or not self._points:
            return False
        self._remove_preview()
        last = self._points[-1]
        pen = QPen(segmentation_color(), 1.2, Qt.PenStyle.DashLine)
        self._preview = self._scene.addLine(last[0], last[1], scene_pos.x(), scene_pos.y(), pen)
        return True

    def mouse_press(self, scene_pos: QPointF) -> bool:
        """Single click — add a point."""
        if not self._active:
            return False
        self._points.append((scene_pos.x(), scene_pos.y()))
        return True

    def double_click(self, scene_pos: QPointF) -> bool:
        """Double click — close the polygon. Pop the duplicate point from the preceding press."""
        if not self._active:
            return False
        # The press event of the double-click already appended a point; remove it
        if self._points:
            self._points.pop()
        if len(self._points) >= 3:
            self._close_polygon()
        return True

    def mouse_release(self, scene_pos: QPointF) -> bool:
        return self._active

    def undo_last_point(self) -> None:
        if self._points:
            self._points.pop()

    def _close_polygon(self) -> None:
        self._remove_preview()
        mask = SegmentationMask(
            points=[(px / self._img_w, py / self._img_h) for px, py in self._points]
        )
        self._points = []
        self._on_complete(mask, self._class_id)

    def _remove_preview(self) -> None:
        if self._preview is not None:
            self._scene.removeItem(self._preview)
            self._preview = None

    @property
    def point_count(self) -> int:
        return len(self._points)

    @property
    def in_progress_points(self) -> list[tuple[float, float]]:
        return list(self._points)
