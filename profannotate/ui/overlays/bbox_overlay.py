"""
profannotate/ui/overlays/bbox_overlay.py
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from profannotate.config.constants import (
    HANDLE_BC,
    HANDLE_BL,
    HANDLE_BR,
    HANDLE_ML,
    HANDLE_MOVE,
    HANDLE_MR,
    HANDLE_NONE,
    HANDLE_TC,
    HANDLE_TL,
    HANDLE_TR,
)
from profannotate.core.annotation.models import BBox
from profannotate.utils.color import class_color

# Re-export for legacy `from profannotate.ui.overlays.bbox_overlay import HANDLE_*` callers.
__all__ = [
    "HANDLE_NONE",
    "HANDLE_MOVE",
    "HANDLE_TL",
    "HANDLE_TC",
    "HANDLE_TR",
    "HANDLE_ML",
    "HANDLE_MR",
    "HANDLE_BL",
    "HANDLE_BC",
    "HANDLE_BR",
    "BBoxOverlay",
]

_HANDLE_R = 4.5
_HIT_R = 8.0


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
        self._violated = False
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
            (r.x(), r.y()),
            (cx, r.y()),
            (r.x() + r.width(), r.y()),
            (r.x(), cy),
            (r.x() + r.width(), cy),
            (r.x(), r.y() + r.height()),
            (cx, r.y() + r.height()),
            (r.x() + r.width(), r.y() + r.height()),
        ]

    def boundingRect(self) -> QRectF:
        m = _HANDLE_R + 3
        return self._rect.adjusted(-m, -m, m, m)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        transform = painter.worldTransform()

        if self._violated:
            color = QColor("#FF4444")
            pw = 2.5
            # Flashing fill to draw attention
            fill = QColor("#FF4444")
            fill.setAlpha(30)
            painter.setBrush(QBrush(fill))
        else:
            color = class_color(self._class_id)
            pw = 2.0 if self._selected else 1.5
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        pen = QPen(color, pw)
        if self._violated:
            pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(self._rect)

        # Class label
        tl_screen = transform.map(QPointF(self._rect.x(), self._rect.y()))
        painter.save()
        painter.resetTransform()
        font = QFont()
        font.setPointSizeF(8.5)
        painter.setFont(font)
        label_color = QColor("#FF4444") if self._violated else color
        lp = QPen(label_color)
        lp.setCosmetic(True)
        painter.setPen(lp)
        label = f"⚠ cls:{self._class_id}" if self._violated else f"cls:{self._class_id}"
        painter.drawText(tl_screen + QPointF(2, -4), label)
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

    def set_violated(self, violated: bool) -> None:
        self._violated = violated
        self.update()
