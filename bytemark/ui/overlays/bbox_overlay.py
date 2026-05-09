"""
bytemark/ui/overlays/bbox_overlay.py
QGraphicsItem for rendering a bounding box annotation on the canvas.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QStyleOptionGraphicsItem, QWidget

from bytemark.core.annotation.models import BBox
from bytemark.utils.color import bbox_color, class_color


class BBoxOverlay(QGraphicsItem):
    def __init__(
        self,
        bbox: BBox,
        img_w: int,
        img_h: int,
        class_id: int = 0,
        instance_idx: int = 0,
    ) -> None:
        super().__init__()
        self._bbox = bbox
        self._img_w = img_w
        self._img_h = img_h
        self._class_id = class_id
        self._idx = instance_idx
        self._selected = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        x1, y1, x2, y2 = bbox.to_xyxy(img_w, img_h)
        self._rect = QRectF(x1, y1, x2 - x1, y2 - y1)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-2, -2, 2, 2)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        color = class_color(self._class_id)
        pen = QPen(color, 1.5, Qt.PenStyle.SolidLine)
        if self._selected:
            pen.setWidth(2.5)
            pen.setColor(QColor("#FFFFFF"))
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(self._rect)

        # Class label
        painter.setPen(QPen(color))
        painter.drawText(
            QRectF(self._rect.x(), self._rect.y() - 14, self._rect.width(), 14),
            Qt.AlignmentFlag.AlignLeft,
            f"cls:{self._class_id}",
        )

    def update_bbox(self, bbox: BBox) -> None:
        self.prepareGeometryChange()
        self._bbox = bbox
        x1, y1, x2, y2 = bbox.to_xyxy(self._img_w, self._img_h)
        self._rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        self.update()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()
