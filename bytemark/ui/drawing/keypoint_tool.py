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
        active_kpt_names: list[str] | None = None,
    ) -> None:
        self._scene = scene
        self._img_w = img_w
        self._img_h = img_h
        self._on_placed = on_keypoint_placed

        if active_kpt_names:
            self._active_names = active_kpt_names
        else:
            self._active_names = [KEYPOINT_NAMES[i] for i in range(NUM_KEYPOINTS)]

        self._num_active = len(self._active_names)
        self._current_idx = 0
        self._active = False
        self._cursor_item = None
        self._zoom = 1.0

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.01, zoom)

    def activate(self, start_idx: int = 0) -> None:
        self._active = True
        self._current_idx = start_idx % self._num_active

    def deactivate(self) -> None:
        self._active = False
        self._remove_cursor()

    def is_active(self) -> bool:
        return self._active

    def current_name(self) -> str:
        if 0 <= self._current_idx < len(self._active_names):
            return self._active_names[self._current_idx]
        return str(self._current_idx)

    def skip(self) -> None:
        """Advance to the next keypoint without placing — triggered by right-click."""
        self._current_idx = (self._current_idx + 1) % self._num_active

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
        label_text = f"{self._current_idx:02d} · {self.current_name()}"
        text = self._scene.addSimpleText(label_text)
        font = QFont()
        font.setPointSizeF(max(5.0, 8.0 / self._zoom))
        text.setFont(font)
        text.setBrush(QBrush(QColor("#FFFFFF")))
        text.setPos(scene_pos.x() + r + 2 / self._zoom, scene_pos.y() - 5 / self._zoom)
        self._cursor_item = (ellipse, text)
        return True

    def mouse_press(self, scene_pos: QPointF) -> bool:
        if not self._active:
            return False
        kp = Keypoint.from_pixel(
            scene_pos.x(), scene_pos.y(), self._img_w, self._img_h, visibility=2
        )
        self._on_placed(self._current_idx, kp)
        self._current_idx = (self._current_idx + 1) % self._num_active
        return True

    def mouse_release(self, scene_pos: QPointF) -> bool:
        return self._active

    def _remove_cursor(self) -> None:
        if self._cursor_item is not None:
            for item in self._cursor_item:
                self._scene.removeItem(item)
            self._cursor_item = None
