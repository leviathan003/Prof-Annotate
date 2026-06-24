"""
profannotate/ui/widgets/skeleton_preview.py

Live diagram of the active keypoint schema on a reference body layout. As the
annotator selects keypoints, the chosen points light up and the skeleton bones
between chosen points are drawn — so they can see *which* keypoints they picked
and *where they sit* on the body / face / hands.

Coordinates come from `pose_template.reference_pose`; bones come from the active
schema's `connections` (drawn only when BOTH endpoints are selected).
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from profannotate.config.pose_template import reference_pose
from profannotate.config.skeleton import get_active_schema
from profannotate.utils.color import keypoint_color, skeleton_color

# Region label anchors (normalized) — purely cosmetic captions for WB133.
_REGION_LABELS = {
    "Face": (0.16, 0.015),
    "Body": (0.50, 0.015),
    "L-Hand": (0.15, 0.55),
    "R-Hand": (0.84, 0.55),
}

_DIM_DOT = QColor(120, 120, 120, 150)
_DIM_BONE = QColor(120, 120, 120, 70)


class SkeletonPreview(QWidget):
    """Paints the reference skeleton; `set_selected(names)` highlights a subset."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 280)
        self._schema = get_active_schema()
        self._pose = reference_pose(self._schema)
        self._selected: set[str] = set()
        self._highlight: str | None = None

    # ----- public API -----

    def set_selected(self, names: list[str]) -> None:
        self._selected = set(names)
        self.update()

    def highlight(self, name: str | None) -> None:
        self._highlight = name
        self.update()

    def refresh_schema(self) -> None:
        """Re-read the active schema (call if it was swapped)."""
        self._schema = get_active_schema()
        self._pose = reference_pose(self._schema)
        self.update()

    # ----- painting -----

    def _to_px(self, xy: tuple[float, float], w: int, h: int, pad: int) -> QPointF:
        return QPointF(pad + xy[0] * (w - 2 * pad), pad + xy[1] * (h - 2 * pad))

    def paintEvent(self, event) -> None:  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h, pad = self.width(), self.height(), 10

        names = self._schema.keypoint_names
        sel = self._selected

        # Bones first (under the dots).
        for a, b in self._schema.connections:
            na, nb = names.get(a), names.get(b)
            if na not in self._pose or nb not in self._pose:
                continue
            both = na in sel and nb in sel
            pen = QPen(skeleton_color(220) if both else _DIM_BONE, 2.0 if both else 1.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(
                self._to_px(self._pose[na], w, h, pad),
                self._to_px(self._pose[nb], w, h, pad),
            )

        # Dots.
        outline = QPen(QColor(0, 0, 0, 160), 0.8)
        outline.setCosmetic(True)
        bright = keypoint_color()
        for name, xy in self._pose.items():
            p = self._to_px(xy, w, h, pad)
            chosen = name in sel
            r = 4.5 if name == self._highlight else (3.2 if chosen else 2.0)
            painter.setPen(outline)
            if name == self._highlight:
                painter.setBrush(QBrush(QColor("#FFFFFF")))
            else:
                painter.setBrush(QBrush(bright if chosen else _DIM_DOT))
            painter.drawEllipse(p, r, r)

        # Region captions (only meaningful for the composite WB133 layout).
        if self._schema.name == "wholebody133":
            font = QFont()
            font.setPointSizeF(7.5)
            painter.setFont(font)
            painter.setPen(QPen(QColor(170, 170, 170, 200)))
            for label, anchor in _REGION_LABELS.items():
                painter.drawText(
                    QRectF(self._to_px(anchor, w, h, pad) - QPointF(30, 8), self._to_px(anchor, w, h, pad) + QPointF(30, 8)),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )
        painter.end()
