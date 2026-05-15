"""
bytemark/ui/drawing/keypoint_tool.py
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsScene, QGraphicsSimpleTextItem

from bytemark.config.constants import NUM_KEYPOINTS
from bytemark.config.skeleton import KEYPOINT_NAMES
from bytemark.core.annotation.models import Keypoint
from bytemark.utils.color import keypoint_color


class KeypointTool:
    def __init__(
        self,
        scene: QGraphicsScene,
        img_w: int,
        img_h: int,
        on_keypoint_placed: Callable[[int, Keypoint], None],
    ) -> None:
        self._scene = scene
        self._img_w = img_w
        self._img_h = img_h
        self._on_placed = on_keypoint_placed
        self._current_idx = 0
        self._active = False
        self._cursor_item = None
        self._zoom = 1.0

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.01, zoom)

    def activate(self, start_idx: int = 0) -> None:
        self._active = True
        self._current_idx = start_idx % NUM_KEYPOINTS

    def deactivate(self) -> None:
        self._active = False
        self._remove_cursor()

    def is_active(self) -> bool:
        return self._active

    def current_name(self) -> str:
        return KEYPOINT_NAMES.get(self._current_idx, str(self._current_idx))

    def mouse_move(self, scene_pos: QPointF) -> bool:
        if not self._active:
            return False
        self._remove_cursor()
        r = 3.0 / self._zoom
        color = keypoint_color()
        pen = QPen(Qt.GlobalColor.white, max(0.4, 0.7 / self._zoom))
        ellipse = self._scene.addEllipse(
            scene_pos.x() - r,
            scene_pos.y() - r,
            r * 2,
            r * 2,
            pen,
            QBrush(color),
        )
        # Name label — rendered in scene space but sized for screen
        name = KEYPOINT_NAMES.get(self._current_idx, str(self._current_idx))
        label_text = f"{self._current_idx:02d} · {name}"
        text = self._scene.addSimpleText(label_text)
        font = QFont()
        font.setPointSizeF(max(5.0, 8.0 / self._zoom))
        text.setFont(font)
        text.setBrush(QBrush(QColor("#FFFFFF")))
        text.setPos(scene_pos.x() + r + 2 / self._zoom, scene_pos.y() - 5 / self._zoom)
        # Group both items under a single handle
        self._cursor_item = (ellipse, text)
        return True

    def mouse_press(self, scene_pos: QPointF) -> bool:
        if not self._active:
            return False
        kp = Keypoint.from_pixel(
            scene_pos.x(), scene_pos.y(), self._img_w, self._img_h, visibility=2
        )
        self._on_placed(self._current_idx, kp)
        self._current_idx = (self._current_idx + 1) % NUM_KEYPOINTS
        return True

    def mouse_release(self, scene_pos: QPointF) -> bool:
        return self._active

    def _remove_cursor(self) -> None:
        if self._cursor_item is not None:
            for item in self._cursor_item:
                self._scene.removeItem(item)
            self._cursor_item = None
