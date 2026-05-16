"""
bytemark/ui/overlays/keypoint_overlay.py
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from bytemark.config.constants import KEYPOINT_SNAP_RADIUS_PX
from bytemark.config.skeleton import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from bytemark.core.annotation.models import Keypoint
from bytemark.utils.color import keypoint_color, skeleton_color

_KPT_RADIUS = 3.0
_SELECTED_RADIUS = 4.5


def _kpt_visible(kp) -> bool:
    """A keypoint is drawable only if it exists and is not zeroed out."""
    return kp is not None and not (kp.x == 0 and kp.y == 0)


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
        pts = [kp for kp in self._keypoints if _kpt_visible(kp)]
        if not pts:
            self._bounds = QRectF(0, 0, self._img_w, self._img_h)
            return
        xs = [kp.x * self._img_w for kp in pts]
        ys = [kp.y * self._img_h for kp in pts]
        m = _SELECTED_RADIUS + 4
        self._bounds = QRectF(
            min(xs) - m,
            min(ys) - m,
            max(xs) - min(xs) + m * 2,
            max(ys) - min(ys) + m * 2,
        )

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        base_sk = skeleton_color(220)
        base_kp = keypoint_color()
        transform = painter.worldTransform()

        # ── Skeleton lines ────────────────────────────────────────────────────
        for a, b in SKELETON_CONNECTIONS:
            if a >= len(self._keypoints) or b >= len(self._keypoints):
                continue
            ka, kb = self._keypoints[a], self._keypoints[b]
            if not (_kpt_visible(ka) and _kpt_visible(kb)):
                continue
            alpha = 220 if (ka.visibility == 2 and kb.visibility == 2) else 130
            c = QColor(base_sk)
            c.setAlpha(alpha)
            pen = QPen(c, 1.2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(ka.x * self._img_w, ka.y * self._img_h),
                QPointF(kb.x * self._img_w, kb.y * self._img_h),
            )

        # ── Keypoint dots ─────────────────────────────────────────────────────
        for i, kp in enumerate(self._keypoints):
            if not _kpt_visible(kp):
                continue
            px, py = kp.x * self._img_w, kp.y * self._img_h
            screen = transform.map(QPointF(px, py))
            r = _SELECTED_RADIUS if i == self._selected_kpt else _KPT_RADIUS
            c = QColor(base_kp)
            if kp.visibility == 1:
                c.setAlpha(180)

            painter.save()
            painter.resetTransform()
            op = QPen(QColor(0, 0, 0, 180), 0.8)
            op.setCosmetic(True)
            painter.setPen(op)
            painter.setBrush(QBrush(c))
            painter.drawEllipse(screen, r, r)

            if i == self._selected_kpt:
                name = KEYPOINT_NAMES.get(i, str(i))
                font = QFont()
                font.setPointSizeF(8.5)
                painter.setFont(font)
                tp = QPen(QColor("#FFFFFF"))
                tp.setCosmetic(True)
                painter.setPen(tp)
                painter.drawText(screen + QPointF(r + 3, 4), f"{i:02d} {name}")
            painter.restore()

    def update_keypoints(self, keypoints: list[Keypoint]) -> None:
        self.prepareGeometryChange()
        self._keypoints = keypoints
        self._compute_bounds()
        self.update()

    def select_keypoint(self, idx: int | None) -> None:
        self._selected_kpt = idx
        self.update()

    def hit_test_keypoint(self, scene_pos: QPointF, zoom: float = 1.0) -> int | None:
        radius = KEYPOINT_SNAP_RADIUS_PX / max(0.01, zoom)
        for i, kp in enumerate(self._keypoints):
            if not _kpt_visible(kp):
                continue
            dx = scene_pos.x() - kp.x * self._img_w
            dy = scene_pos.y() - kp.y * self._img_h
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                return i
        return None
