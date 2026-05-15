"""
bytemark/ui/overlays/bbox_overlay.py
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from bytemark.core.annotation.models import BBox
from bytemark.utils.color import class_color

# Handle constants — imported by canvas.py
HANDLE_NONE = -1
HANDLE_MOVE = 0
HANDLE_TL = 1
HANDLE_TC = 2
HANDLE_TR = 3
HANDLE_ML = 4
HANDLE_MR = 5
HANDLE_BL = 6
HANDLE_BC = 7
HANDLE_BR = 8

_HANDLE_R = 4.5  # visual radius (scene px)
_HIT_R = 8.0  # hit-test radius


class BBoxOverlay(QGraphicsItem):
    def __init__(
        self, bbox: BBox, img_w: int, img_h: int, class_id: int = 0, instance_idx: int = 0
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
        self._update_rect()

    def _update_rect(self) -> None:
        x1, y1, x2, y2 = self._bbox.to_xyxy(self._img_w, self._img_h)
        self._rect = QRectF(x1, y1, x2 - x1, y2 - y1)

    def _handle_positions(self) -> list[tuple[float, float]]:
        r = self._rect
        cx = r.x() + r.width() / 2
        cy = r.y() + r.height() / 2
        return [
            (r.x(), r.y()),  # TL 1
            (cx, r.y()),  # TC 2
            (r.x() + r.width(), r.y()),  # TR 3
            (r.x(), cy),  # ML 4
            (r.x() + r.width(), cy),  # MR 5
            (r.x(), r.y() + r.height()),  # BL 6
            (cx, r.y() + r.height()),  # BC 7
            (r.x() + r.width(), r.y() + r.height()),  # BR 8
        ]

    def boundingRect(self) -> QRectF:
        m = _HANDLE_R + 3
        return self._rect.adjusted(-m, -m, m, m)

    def paint(self, painter, option, widget=None):
        transform = painter.worldTransform()
        color = class_color(self._class_id)

        pw = 2.0 if self._selected else 1.5
        pen = QPen(color, pw)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(self._rect)

        # Class label — fixed screen size
        tl_screen = transform.map(QPointF(self._rect.x(), self._rect.y()))
        painter.save()
        painter.resetTransform()
        font = QFont()
        font.setPointSizeF(8.5)
        painter.setFont(font)
        lp = QPen(color)
        lp.setCosmetic(True)
        painter.setPen(lp)
        painter.drawText(tl_screen + QPointF(2, -4), f"cls:{self._class_id}")
        painter.restore()

        if self._selected:
            h_ids = [
                HANDLE_TL,
                HANDLE_TC,
                HANDLE_TR,
                HANDLE_ML,
                HANDLE_MR,
                HANDLE_BL,
                HANDLE_BC,
                HANDLE_BR,
            ]
            for _, (hx, hy) in zip(h_ids, self._handle_positions()):
                screen = transform.map(QPointF(hx, hy))
                painter.save()
                painter.resetTransform()
                hp = QPen(QColor("#000000"), 0.8)
                hp.setCosmetic(True)
                painter.setPen(hp)
                painter.setBrush(QBrush(QColor("#FFFFFF")))
                painter.drawEllipse(screen, _HANDLE_R, _HANDLE_R)
                painter.restore()

    def hit_test_handle(self, scene_pos: QPointF) -> int:
        """Returns a HANDLE_* constant. Only checks resize handles when selected."""
        if self._selected:
            h_ids = [
                HANDLE_TL,
                HANDLE_TC,
                HANDLE_TR,
                HANDLE_ML,
                HANDLE_MR,
                HANDLE_BL,
                HANDLE_BC,
                HANDLE_BR,
            ]
            for hid, (hx, hy) in zip(h_ids, self._handle_positions()):
                dx = scene_pos.x() - hx
                dy = scene_pos.y() - hy
                if (dx * dx + dy * dy) ** 0.5 <= _HIT_R:
                    return hid
        if self._rect.contains(scene_pos):
            return HANDLE_MOVE
        return HANDLE_NONE

    def update_bbox(self, bbox: BBox) -> None:
        self.prepareGeometryChange()
        self._bbox = bbox
        self._update_rect()
        self.update()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()
