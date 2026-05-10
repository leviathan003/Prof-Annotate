"""
bytemark/ui/overlays/segmentation_overlay.py
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from bytemark.config.constants import POLYGON_CLOSE_RADIUS_PX
from bytemark.core.annotation.models import SegmentationMask
from bytemark.utils.color import segmentation_color, segmentation_fill_color

_POINT_RADIUS = 2.0
_SELECTED_RADIUS = 3.5


class SegmentationOverlay(QGraphicsItem):
    def __init__(
        self,
        mask: SegmentationMask,
        img_w: int,
        img_h: int,
        instance_idx: int = 0,
        is_drawing: bool = False,
    ) -> None:
        super().__init__()
        self._mask = mask
        self._img_w = img_w
        self._img_h = img_h
        self._idx = instance_idx
        self._is_drawing = is_drawing
        self._selected_pt: int | None = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def boundingRect(self) -> QRectF:
        if not self._mask.points:
            return QRectF(0, 0, self._img_w, self._img_h)
        xs = [x * self._img_w for x, _ in self._mask.points]
        ys = [y * self._img_h for _, y in self._mask.points]
        m = _SELECTED_RADIUS + 3
        return QRectF(
            min(xs) - m, min(ys) - m, max(xs) - min(xs) + m * 2, max(ys) - min(ys) + m * 2
        )

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        if not self._mask.points:
            return

        pts_px = [(x * self._img_w, y * self._img_h) for x, y in self._mask.points]
        polygon = QPolygonF([QPointF(x, y) for x, y in pts_px])
        color = segmentation_color()

        # Fill
        if self._mask.is_closed() and not self._is_drawing:
            painter.setBrush(QBrush(segmentation_fill_color()))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(polygon)

        # Outline — deeper, fully opaque
        outline_color = QColor(color)
        outline_color.setAlpha(230)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.setPen(QPen(outline_color, 1.5, Qt.PenStyle.SolidLine))
        if self._mask.is_closed() and not self._is_drawing:
            painter.drawPolygon(polygon)
        else:
            for i in range(len(pts_px) - 1):
                painter.drawLine(QPointF(*pts_px[i]), QPointF(*pts_px[i + 1]))

        # Control points
        for i, (px, py) in enumerate(pts_px):
            r = _SELECTED_RADIUS if i == self._selected_pt else _POINT_RADIUS
            pt_color = QColor(color)
            pt_color.setAlpha(230)
            if i == 0 and self._is_drawing:
                painter.setPen(QPen(Qt.GlobalColor.white, 1.2))
                painter.setBrush(QBrush(pt_color))
                painter.drawEllipse(QPointF(px, py), r + 1.5, r + 1.5)
            else:
                painter.setPen(QPen(QColor(0, 0, 0, 160), 0.6))
                painter.setBrush(QBrush(pt_color))
                painter.drawEllipse(QPointF(px, py), r, r)

    def update_mask(self, mask: SegmentationMask) -> None:
        self.prepareGeometryChange()
        self._mask = mask
        self.update()

    def select_point(self, idx: int | None) -> None:
        self._selected_pt = idx
        self.update()

    def set_drawing(self, drawing: bool) -> None:
        self._is_drawing = drawing
        self.update()

    def hit_test_point(self, scene_pos: QPointF) -> int | None:
        for i, (x, y) in enumerate(self._mask.points):
            dx = scene_pos.x() - x * self._img_w
            dy = scene_pos.y() - y * self._img_h
            if (dx * dx + dy * dy) ** 0.5 <= POLYGON_CLOSE_RADIUS_PX:
                return i
        return None
