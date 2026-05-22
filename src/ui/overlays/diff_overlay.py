"""
bytemark/ui/overlays/diff_overlay.py
Visual diff between old and new annotations (Ctrl+Y auto-annotate preview).

OLD = dashed thin outlines, faint fills, small dim kpts.
NEW = solid bolder outlines, brighter fills, skeleton + indexed kpts.
Pen widths are cosmetic so the diff stays readable at any zoom.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from src.config.skeleton import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from src.core.annotation.models import Annotation
from src.utils.color import diff_new_color, diff_old_color


def _kpt_visible(kp) -> bool:
    return kp is not None and not (kp.x == 0 and kp.y == 0)


class DiffOverlay(QGraphicsItem):
    def __init__(
        self,
        old_annotations: list[Annotation],
        new_annotations: list[Annotation],
        img_w: int,
        img_h: int,
        active_kpt_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._old = old_annotations
        self._new = new_annotations
        self._w = img_w
        self._h = img_h

        if active_kpt_names:
            name_to_new: dict[str, int] = {n: i for i, n in enumerate(active_kpt_names)}
            self._connections: list[tuple[int, int]] = [
                (name_to_new[KEYPOINT_NAMES[a]], name_to_new[KEYPOINT_NAMES[b]])
                for a, b in SKELETON_CONNECTIONS
                if KEYPOINT_NAMES.get(a) in name_to_new and KEYPOINT_NAMES.get(b) in name_to_new
            ]
            self._kpt_names = active_kpt_names
        else:
            self._connections = list(SKELETON_CONNECTIONS)
            self._kpt_names = [KEYPOINT_NAMES.get(i, str(i)) for i in range(64)]

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._w, self._h)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # OLD first → NEW drawn over it
        self._draw_set(painter, self._old, diff_old_color(), is_new=False)
        self._draw_set(painter, self._new, diff_new_color(), is_new=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _draw_set(
        self,
        painter: QPainter,
        annotations: list[Annotation],
        base_color: QColor,
        is_new: bool,
    ) -> None:
        outline_color = QColor(base_color)
        outline_color.setAlpha(255)
        fill_color = QColor(base_color)
        fill_color.setAlpha(70 if is_new else 35)

        outline = QPen(outline_color, 2.4 if is_new else 1.6)
        outline.setCosmetic(True)
        outline.setStyle(Qt.PenStyle.SolidLine if is_new else Qt.PenStyle.DashLine)
        outline.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        outline.setCapStyle(Qt.PenCapStyle.RoundCap)

        for inst_idx, ann in enumerate(annotations):
            # ── Bbox ──────────────────────────────────────────────────────────
            if ann.has_bbox():
                x1, y1, x2, y2 = ann.bbox.to_xyxy(self._w, self._h)
                rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                painter.setPen(outline)
                painter.setBrush(QBrush(fill_color))
                painter.drawRect(rect)
                self._draw_pill_tag(
                    painter, x1, y1, "NEW" if is_new else "OLD", outline_color, inst_idx
                )

            # ── Mask ──────────────────────────────────────────────────────────
            if ann.has_mask() and ann.mask.is_closed():
                pts = ann.mask.to_pixel_list(self._w, self._h)
                polygon = QPolygonF([QPointF(x, y) for x, y in pts])

                if is_new:
                    painter.setPen(outline)
                    painter.setBrush(QBrush(fill_color))
                    painter.drawPolygon(polygon)
                else:
                    # Diagonal-stripe brush for OLD so it visually separates from NEW fill.
                    path = QPainterPath()
                    path.addPolygon(polygon)
                    stripe = QBrush(fill_color, Qt.BrushStyle.BDiagPattern)
                    painter.setPen(outline)
                    painter.setBrush(stripe)
                    painter.drawPath(path)

            # ── Keypoints + skeleton ──────────────────────────────────────────
            if ann.has_keypoints():
                self._draw_skeleton_and_kpts(painter, ann, outline_color, is_new)

    def _draw_skeleton_and_kpts(
        self,
        painter: QPainter,
        ann: Annotation,
        color: QColor,
        is_new: bool,
    ) -> None:
        kpts = ann.keypoints

        # Skeleton
        sk_pen = QPen(color, 1.6 if is_new else 1.0)
        sk_pen.setCosmetic(True)
        sk_pen.setStyle(Qt.PenStyle.SolidLine if is_new else Qt.PenStyle.DashLine)
        sk_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(sk_pen)
        for a, b in self._connections:
            if a >= len(kpts) or b >= len(kpts):
                continue
            ka, kb = kpts[a], kpts[b]
            if not (_kpt_visible(ka) and _kpt_visible(kb)):
                continue
            painter.drawLine(
                QPointF(ka.x * self._w, ka.y * self._h),
                QPointF(kb.x * self._w, kb.y * self._h),
            )

        # Keypoint dots — drawn in scene coordinates with cosmetic outlines so
        # they're crisp at any zoom. NEW dots are larger with a white halo and
        # show their index; OLD dots are small and hollow-bordered.
        transform = painter.worldTransform()
        radius = 5.0 if is_new else 3.5
        white_pen = QPen(QColor(255, 255, 255, 230), 1.4)
        white_pen.setCosmetic(True)
        dim_pen = QPen(QColor(0, 0, 0, 200), 1.0)
        dim_pen.setCosmetic(True)

        font = QFont()
        font.setPointSizeF(8.5)
        font.setBold(True)
        metrics = QFontMetrics(font)

        for i, kp in enumerate(kpts):
            if not _kpt_visible(kp):
                continue
            px, py = kp.x * self._w, kp.y * self._h
            screen = transform.map(QPointF(px, py))

            painter.save()
            painter.resetTransform()
            painter.setBrush(QBrush(color))
            painter.setPen(white_pen if is_new else dim_pen)
            painter.drawEllipse(screen, radius, radius)

            if is_new:
                label = f"{i:02d}"
                text_w = metrics.horizontalAdvance(label)
                text_h = metrics.height()
                pad = 2
                bg_rect = QRectF(
                    screen.x() + radius + 3,
                    screen.y() - text_h / 2 - pad,
                    text_w + pad * 2,
                    text_h + pad * 2,
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                painter.drawRoundedRect(bg_rect, 3, 3)
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.setFont(font)
                painter.drawText(
                    QPointF(bg_rect.x() + pad, bg_rect.y() + pad + metrics.ascent() - 1),
                    label,
                )
            painter.restore()

    def _draw_pill_tag(
        self,
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        color: QColor,
        inst_idx: int,
    ) -> None:
        transform = painter.worldTransform()
        anchor = transform.map(QPointF(x, y))

        font = QFont()
        font.setPointSizeF(9.0)
        font.setBold(True)
        metrics = QFontMetrics(font)
        label = f"{text} · {inst_idx}"
        text_w = metrics.horizontalAdvance(label)
        text_h = metrics.height()
        pad_x = 6
        pad_y = 2

        pill = QRectF(
            anchor.x(),
            anchor.y() - text_h - pad_y * 2 - 2,
            text_w + pad_x * 2,
            text_h + pad_y * 2,
        )

        painter.save()
        painter.resetTransform()
        bg = QColor(color)
        bg.setAlpha(230)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(pill, 3, 3)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0, 230)))
        painter.drawText(
            QPointF(pill.x() + pad_x, pill.y() + pad_y + metrics.ascent() - 1),
            label,
        )
        painter.restore()
