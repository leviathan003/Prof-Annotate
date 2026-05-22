"""
bytemark/ui/drawing/keypoint_tool.py
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene

from bytemark.config.constants import NUM_KEYPOINTS
from bytemark.config.skeleton import KEYPOINT_NAMES
from bytemark.core.annotation.models import Keypoint
from bytemark.utils.color import keypoint_color


def _build_dot_cursor(label: str = "") -> QCursor:
    dot_size = 14
    r = 4
    hotspot = dot_size // 2

    font = QFont()
    font.setPointSizeF(9.0)
    font.setBold(True)
    metrics = QFontMetrics(font)
    text_w = metrics.horizontalAdvance(label) if label else 0
    text_h = metrics.height()

    pad_x = 6
    pad_y = 2
    total_w = dot_size + (pad_x + text_w + 4 if label else 0)
    total_h = max(dot_size, text_h + pad_y * 2)

    pix = QPixmap(total_w, total_h)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

    # Dot
    painter.setPen(QPen(Qt.GlobalColor.white, 1.2))
    painter.setBrush(QBrush(keypoint_color()))
    painter.drawEllipse(hotspot - r, total_h // 2 - r, r * 2, r * 2)

    # Label
    if label:
        text_x = dot_size + pad_x
        bg_rect_x = text_x - 3
        bg_rect_y = (total_h - text_h) // 2 - 1
        bg_rect_w = text_w + 6
        bg_rect_h = text_h + 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.drawRoundedRect(bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h, 3, 3)
        painter.setFont(font)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(text_x, total_h // 2 + metrics.ascent() // 2 - 1, label)

    painter.end()
    return QCursor(pix, hotspot, total_h // 2)


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
        self._zoom = 1.0

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.01, zoom)

    def activate(self, start_idx: int = 0) -> None:
        self._active = True
        self._current_idx = start_idx % self._num_active

    def deactivate(self) -> None:
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def current_name(self) -> str:
        if 0 <= self._current_idx < len(self._active_names):
            return self._active_names[self._current_idx]
        return str(self._current_idx)

    def skip(self) -> None:
        """Advance past current keypoint without placing — right-click."""
        self._current_idx = (self._current_idx + 1) % self._num_active

    def mouse_move(self, scene_pos: QPointF) -> bool:
        return self._active

    def mouse_press(self, scene_pos: QPointF) -> bool:
        if not self._active:
            return False
        kp = Keypoint.from_pixel(
            scene_pos.x(), scene_pos.y(), self._img_w, self._img_h, visibility=2
        )
        placed_idx = self._current_idx
        self._current_idx = (self._current_idx + 1) % self._num_active
        self._on_placed(placed_idx, kp)
        return True

    def mouse_release(self, scene_pos: QPointF) -> bool:
        return self._active
