"""
bytemark/ui/overlays/keypoint_overlay.py
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from src.config.constants import KEYPOINT_SNAP_RADIUS_PX
from src.config.skeleton import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from src.core.annotation.models import Keypoint
from src.utils.color import keypoint_color, skeleton_color

_KPT_RADIUS = 3.0
_SELECTED_RADIUS = 4.5


def _kpt_visible(kp) -> bool:
    return kp is not None and not (kp.x == 0 and kp.y == 0)


class KeypointOverlay(QGraphicsItem):
    def __init__(
        self,
        keypoints: list[Keypoint],
        img_w: int,
        img_h: int,
        instance_idx: int = 0,
        active_kpt_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._keypoints = keypoints
        self._img_w = img_w
        self._img_h = img_h
        self._idx = instance_idx
        self._selected_kpt: int | None = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        if active_kpt_names:
            self._kpt_names = active_kpt_names
            # Remap skeleton connections to indices within the active subset
            name_to_new: dict[str, int] = {n: i for i, n in enumerate(active_kpt_names)}
            self._connections: list[tuple[int, int]] = [
                (name_to_new[KEYPOINT_NAMES[a]], name_to_new[KEYPOINT_NAMES[b]])
                for a, b in SKELETON_CONNECTIONS
                if KEYPOINT_NAMES.get(a) in name_to_new and KEYPOINT_NAMES.get(b) in name_to_new
            ]
        else:
            self._kpt_names = [KEYPOINT_NAMES.get(i, str(i)) for i in range(len(keypoints))]
            self._connections = list(SKELETON_CONNECTIONS)

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
        iw = self._img_w
        ih = self._img_h
        kpts = self._keypoints
        n_kp = len(kpts)
        selected = self._selected_kpt

        # Skeleton lines — reuse a single QPen, only mutate its colour. This
        # avoids allocating a new QColor + QPen per drawn segment.
        sk_pen = QPen(QColor(base_sk), 1.2)
        sk_pen.setCosmetic(True)
        for a, b in self._connections:
            if a >= n_kp or b >= n_kp:
                continue
            ka, kb = kpts[a], kpts[b]
            if not (_kpt_visible(ka) and _kpt_visible(kb)):
                continue
            alpha = 220 if (ka.visibility == 2 and kb.visibility == 2) else 130
            c = QColor(base_sk)
            c.setAlpha(alpha)
            sk_pen.setColor(c)
            painter.setPen(sk_pen)
            painter.drawLine(
                QPointF(ka.x * iw, ka.y * ih),
                QPointF(kb.x * iw, kb.y * ih),
            )

        # Keypoint dots — share the outline pen + label pen + label font.
        outline_pen = QPen(QColor(0, 0, 0, 180), 0.8)
        outline_pen.setCosmetic(True)
        label_pen = QPen(QColor("#FFFFFF"))
        label_pen.setCosmetic(True)
        label_font: QFont | None = None  # built lazily only if a dot is selected

        for i, kp in enumerate(kpts):
            if not _kpt_visible(kp):
                continue
            px, py = kp.x * iw, kp.y * ih
            screen = transform.map(QPointF(px, py))
            r = _SELECTED_RADIUS if i == selected else _KPT_RADIUS
            c = QColor(base_kp)
            if kp.visibility == 1:
                c.setAlpha(180)

            painter.save()
            painter.resetTransform()
            painter.setPen(outline_pen)
            painter.setBrush(QBrush(c))
            painter.drawEllipse(screen, r, r)

            if i == selected:
                name = self._kpt_names[i] if i < len(self._kpt_names) else str(i)
                if label_font is None:
                    label_font = QFont()
                    label_font.setPointSizeF(8.5)
                painter.setFont(label_font)
                painter.setPen(label_pen)
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
