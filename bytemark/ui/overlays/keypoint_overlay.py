"""
bytemark/ui/overlays/keypoint_overlay.py
Renders keypoints and skeleton connections on the canvas.
Each keypoint is individually selectable and draggable.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from bytemark.config.constants import KEYPOINT_SNAP_RADIUS_PX
from bytemark.config.skeleton import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from bytemark.core.annotation.models import Keypoint
from bytemark.utils.color import keypoint_color, skeleton_color

_KPT_RADIUS = 4.0
_SELECTED_RADIUS = 6.0


class KeypointOverlay(QGraphicsItem):
    def __init__(
        self,
        keypoints: list[Keypoint],
        img_w: int,
        img_h: int,
        instance_idx: int = 0,
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
        if not self._keypoints:
            self._bounds = QRectF(0, 0, self._img_w, self._img_h)
            return
        xs = [kp.x * self._img_w for kp in self._keypoints if kp.visibility > 0]
        ys = [kp.y * self._img_h for kp in self._keypoints if kp.visibility > 0]
        if not xs:
            self._bounds = QRectF(0, 0, self._img_w, self._img_h)
            return
        margin = _SELECTED_RADIUS + 2
        self._bounds = QRectF(
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + margin * 2,
            max(ys) - min(ys) + margin * 2,
        )

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        kp_color = keypoint_color()
        sk_color = skeleton_color(160)

        # Draw skeleton lines
        sk_pen = QPen(sk_color, 1.2, Qt.PenStyle.SolidLine)
        painter.setPen(sk_pen)
        for a, b in SKELETON_CONNECTIONS:
            if a >= len(self._keypoints) or b >= len(self._keypoints):
                continue
            ka, kb = self._keypoints[a], self._keypoints[b]
            if ka.visibility == 0 or kb.visibility == 0:
                continue
            painter.drawLine(
                QPointF(ka.x * self._img_w, ka.y * self._img_h),
                QPointF(kb.x * self._img_w, kb.y * self._img_h),
            )

        # Draw keypoints
        for i, kp in enumerate(self._keypoints):
            if kp.visibility == 0:
                continue
            px, py = kp.x * self._img_w, kp.y * self._img_h
            r = _SELECTED_RADIUS if i == self._selected_kpt else _KPT_RADIUS

            color = kp_color
            if kp.visibility == 1:
                color = QColor(kp_color)
                color.setAlpha(120)

            painter.setPen(QPen(QColor("#000000"), 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(px, py), r, r)

            # Label on hover/select
            if i == self._selected_kpt:
                name = KEYPOINT_NAMES.get(i, str(i))
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.drawText(QPointF(px + 6, py - 4), name)

    def update_keypoints(self, keypoints: list[Keypoint]) -> None:
        self.prepareGeometryChange()
        self._keypoints = keypoints
        self._compute_bounds()
        self.update()

    def select_keypoint(self, idx: int | None) -> None:
        self._selected_kpt = idx
        self.update()

    def hit_test_keypoint(self, scene_pos: QPointF) -> int | None:
        """Return keypoint index if scene_pos is within snap radius, else None."""
        for i, kp in enumerate(self._keypoints):
            if kp.visibility == 0:
                continue
            dx = scene_pos.x() - kp.x * self._img_w
            dy = scene_pos.y() - kp.y * self._img_h
            if (dx * dx + dy * dy) ** 0.5 <= KEYPOINT_SNAP_RADIUS_PX:
                return i
        return None
