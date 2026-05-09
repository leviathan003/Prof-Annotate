"""
bytemark/ui/overlays/diff_overlay.py
Visual diff between old and new annotations (Ctrl+Y auto-annotate preview).
Old annotations shown in red, new in green. Press Enter to accept.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from bytemark.core.annotation.models import Annotation
from bytemark.utils.color import diff_new_color, diff_old_color


class DiffOverlay(QGraphicsItem):
    def __init__(
        self,
        old_annotations: list[Annotation],
        new_annotations: list[Annotation],
        img_w: int,
        img_h: int,
    ) -> None:
        super().__init__()
        self._old = old_annotations
        self._new = new_annotations
        self._w = img_w
        self._h = img_h

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._w, self._h)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        self._draw_annotations(painter, self._old, diff_old_color(), "OLD")
        self._draw_annotations(painter, self._new, diff_new_color(), "NEW")

    def _draw_annotations(
        self,
        painter: QPainter,
        annotations: list[Annotation],
        color: QColor,
        label: str,
    ) -> None:
        pen = QPen(color, 2.0, Qt.PenStyle.DashLine if label == "OLD" else Qt.PenStyle.SolidLine)
        fill = QColor(color)
        fill.setAlpha(40)

        for ann in annotations:
            painter.setPen(pen)

            if ann.has_bbox():
                x1, y1, x2, y2 = ann.bbox.to_xyxy(self._w, self._h)
                rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                painter.setBrush(QBrush(fill))
                painter.drawRect(rect)
                painter.setPen(QPen(color))
                painter.drawText(QPointF(x1, y1 - 4), label)

            if ann.has_mask() and ann.mask.is_closed():
                pts = ann.mask.to_pixel_list(self._w, self._h)
                polygon = QPolygonF([QPointF(x, y) for x, y in pts])
                painter.setBrush(QBrush(fill))
                painter.drawPolygon(polygon)

            if ann.has_keypoints():
                painter.setBrush(QBrush(color))
                for kp in ann.keypoints:
                    if kp.visibility > 0:
                        painter.drawEllipse(QPointF(kp.x * self._w, kp.y * self._h), 4, 4)
