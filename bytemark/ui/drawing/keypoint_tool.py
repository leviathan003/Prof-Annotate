"""
bytemark/ui/drawing/keypoint_tool.py
Keypoint placement tool. Activated with 'K'.
Cycles through keypoint indices 0..NUM_KEYPOINTS-1 on each click.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsScene

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
        color = keypoint_color()
        self._cursor_item = self._scene.addEllipse(
            scene_pos.x() - 4,
            scene_pos.y() - 4,
            8,
            8,
            QPen(Qt.GlobalColor.white, 1),
            QBrush(color),
        )
        return True

    def mouse_press(self, scene_pos: QPointF) -> bool:
        if not self._active:
            return False
        kp = Keypoint.from_pixel(
            scene_pos.x(),
            scene_pos.y(),
            self._img_w,
            self._img_h,
            visibility=2,
        )
        self._on_placed(self._current_idx, kp)
        self._current_idx = (self._current_idx + 1) % NUM_KEYPOINTS
        return True

    def mouse_release(self, scene_pos: QPointF) -> bool:
        return self._active

    def _remove_cursor(self) -> None:
        if self._cursor_item is not None:
            self._scene.removeItem(self._cursor_item)
            self._cursor_item = None
