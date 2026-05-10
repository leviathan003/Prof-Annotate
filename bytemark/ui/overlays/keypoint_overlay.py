"""
bytemark/ui/overlays/keypoint_overlay.py
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from bytemark.config.constants import KEYPOINT_SNAP_RADIUS_PX
from bytemark.config.skeleton import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from bytemark.core.annotation.models import Keypoint
from bytemark.utils.color import keypoint_color, skeleton_color

_KPT_RADIUS = 2.0
_SELECTED_RADIUS = 3.5


class KeypointOverlay(QGraphicsItem):
    def __init__(
        self, keypoints: list[Keypoint], img_w: int, img_h: int, instance_idx: int = 0
    ) -> None:
        super().__init__()
        self._keypoints = keypoints
        self._img_w = img_w
        self._img_h = img_h
        self._idx = instance_idx
        self._selected_kpt: int | None = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._compute_bounds()

    def _compute_bounds(self) -> None:
        pts = [kp for kp in self._keypoints if kp is not None]
        if not pts:
            self._bounds = QRectF(0, 0, self._img_w, self._img_h)
            return
        xs = [kp.x * self._img_w for kp in pts]
        ys = [kp.y * self._img_h for kp in pts]
        m = _SELECTED_RADIUS + 4
        self._bounds = QRectF(
            min(xs) - m, min(ys) - m, max(xs) - min(xs) + m * 2, max(ys) - min(ys) + m * 2
        )

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        base_kp = keypoint_color()
        base_sk = skeleton_color(220)

        # Skeleton
        for a, b in SKELETON_CONNECTIONS:
            if a >= len(self._keypoints) or b >= len(self._keypoints):
                continue
            ka, kb = self._keypoints[a], self._keypoints[b]
            if ka is None or kb is None:
                continue
            alpha = 90 if (ka.visibility == 0 or kb.visibility == 0) else 220
            c = QColor(base_sk)
            c.setAlpha(alpha)
            painter.setPen(QPen(c, 1.0))
            painter.drawLine(
                QPointF(ka.x * self._img_w, ka.y * self._img_h),
                QPointF(kb.x * self._img_w, kb.y * self._img_h),
            )

        # Keypoints
        for i, kp in enumerate(self._keypoints):
            if kp is None:
                continue
            px, py = kp.x * self._img_w, kp.y * self._img_h
            r = _SELECTED_RADIUS if i == self._selected_kpt else _KPT_RADIUS
            c = QColor(base_kp)
            if kp.visibility == 0:
                c.setAlpha(110)
            elif kp.visibility == 1:
                c.setAlpha(180)
            # v=2 → full alpha
            painter.setPen(QPen(QColor(0, 0, 0, 160), 0.6))
            painter.setBrush(QBrush(c))
            painter.drawEllipse(QPointF(px, py), r, r)
            if i == self._selected_kpt:
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.drawText(QPointF(px + 5, py - 3), KEYPOINT_NAMES.get(i, str(i)))

    def update_keypoints(self, keypoints: list[Keypoint]) -> None:
        self.prepareGeometryChange()
        self._keypoints = keypoints
        self._compute_bounds()
        self.update()

    def select_keypoint(self, idx: int | None) -> None:
        self._selected_kpt = idx
        self.update()

    def hit_test_keypoint(self, scene_pos: QPointF) -> int | None:
        for i, kp in enumerate(self._keypoints):
            if kp is None:
                continue
            dx = scene_pos.x() - kp.x * self._img_w
            dy = scene_pos.y() - kp.y * self._img_h
            if (dx * dx + dy * dy) ** 0.5 <= KEYPOINT_SNAP_RADIUS_PX:
                return i
        return None
