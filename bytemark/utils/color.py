"""
bytemark/utils/color.py
Class-to-color mapping and annotation color helpers.
Generates distinct colors per class id deterministically.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

from bytemark.config.constants import (
    BBOX_COLOR,
    DIFF_NEW_ALPHA,
    DIFF_NEW_COLOR,
    DIFF_OLD_ALPHA,
    DIFF_OLD_COLOR,
    KEYPOINT_COLOR,
    SEGMENTATION_COLOR,
    SEGMENTATION_FILL_ALPHA,
    SKELETON_COLOR,
)

# Deterministic palette — cycles for class_id > len
_CLASS_PALETTE = [
    "#00CFFF",
    "#FF6B6B",
    "#FFD700",
    "#00FF88",
    "#CC44FF",
    "#FF8800",
    "#00FFFF",
    "#FF44AA",
    "#88FF00",
    "#4488FF",
    "#FF4444",
    "#44FFAA",
    "#FFAA44",
    "#AA44FF",
    "#44AAFF",
]


def class_color(class_id: int, alpha: int = 255) -> QColor:
    hex_color = _CLASS_PALETTE[class_id % len(_CLASS_PALETTE)]
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


def bbox_color(alpha: int = 255) -> QColor:
    c = QColor(BBOX_COLOR)
    c.setAlpha(alpha)
    return c


def keypoint_color(alpha: int = 255) -> QColor:
    c = QColor(KEYPOINT_COLOR)
    c.setAlpha(alpha)
    return c


def skeleton_color(alpha: int = 255) -> QColor:
    c = QColor(SKELETON_COLOR)
    c.setAlpha(alpha)
    return c


def segmentation_color(alpha: int = 255) -> QColor:
    c = QColor(SEGMENTATION_COLOR)
    c.setAlpha(alpha)
    return c


def segmentation_fill_color() -> QColor:
    c = QColor(SEGMENTATION_COLOR)
    c.setAlpha(SEGMENTATION_FILL_ALPHA)
    return c


def diff_old_color() -> QColor:
    c = QColor(DIFF_OLD_COLOR)
    c.setAlpha(DIFF_OLD_ALPHA)
    return c


def diff_new_color() -> QColor:
    c = QColor(DIFF_NEW_COLOR)
    c.setAlpha(DIFF_NEW_ALPHA)
    return c
