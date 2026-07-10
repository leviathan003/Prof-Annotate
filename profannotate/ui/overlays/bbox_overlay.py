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

_H_IDS = (
    HANDLE_TL,
    HANDLE_TC,
    HANDLE_TR,
    HANDLE_ML,
    HANDLE_MR,
    HANDLE_BL,
    HANDLE_BC,
    HANDLE_BR,
)

# Immutable paint objects shared by every BBoxOverlay — built lazily on first
# paint (after QApplication exists) instead of per paint call.
_SHARED: dict = {}


def _shared():
    if not _SHARED:
        font = QFont()
        font.setPointSizeF(8.5)
        handle_pen = QPen(QColor("#000000"), 0.8)
        handle_pen.setCosmetic(True)
        violated = QColor("#FF4444")
        violated_fill = QColor("#FF4444")
        violated_fill.setAlpha(30)
        _SHARED.update(
            font=font,
            handle_pen=handle_pen,
            handle_brush=QBrush(QColor("#FFFFFF")),
            violated=violated,
            violated_fill_brush=QBrush(violated_fill),
            no_brush=QBrush(Qt.BrushStyle.NoBrush),
        )
    return _SHARED


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
        # Handle positions only change with geometry — compute once here, not
        # in every paint AND every hit test.
        r = self._rect
        cx = r.x() + r.width() / 2
        cy = r.y() + r.height() / 2
        self._handles: list[tuple[float, float]] = [
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
        shared = _shared()

        if self._violated:
            color = shared["violated"]
            pw = 2.5
            painter.setBrush(shared["violated_fill_brush"])
        else:
            color = class_color(self._class_id)
            pw = 2.0 if self._selected else 1.5
            painter.setBrush(shared["no_brush"])

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
        painter.setFont(shared["font"])
        lp = QPen(color)
        lp.setCosmetic(True)
        painter.setPen(lp)
        label = f"⚠ cls:{self._class_id}" if self._violated else f"cls:{self._class_id}"
        painter.drawText(tl_screen + QPointF(2, -4), label)
        painter.restore()

        if self._selected:
            painter.save()
            painter.resetTransform()
            painter.setPen(shared["handle_pen"])
            painter.setBrush(shared["handle_brush"])
            for hx, hy in self._handles:
                screen = transform.map(QPointF(hx, hy))
                painter.drawEllipse(screen, _HANDLE_R, _HANDLE_R)
            painter.restore()

    def hit_test_handle(self, scene_pos: QPointF) -> int:
        if self._selected:
            for hid, (hx, hy) in zip(_H_IDS, self._handles):
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
