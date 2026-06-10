"""
profannotate/core/annotation/models.py
Pure dataclasses for annotation primitives.
Zero UI or I/O dependencies.
All coordinates are YOLO-normalized (0.0–1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Modality(Enum):
    BBOX = auto()
    KEYPOINTS = auto()
    SEGMENTATION = auto()


@dataclass
class BBox:
    cx: float
    cy: float
    w: float
    h: float

    def clamp(self) -> "BBox":
        return BBox(
            cx=max(0.0, min(1.0, self.cx)),
            cy=max(0.0, min(1.0, self.cy)),
            w=max(0.0, min(1.0, self.w)),
            h=max(0.0, min(1.0, self.h)),
        )

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        x1 = (self.cx - self.w / 2) * img_w
        y1 = (self.cy - self.h / 2) * img_h
        x2 = (self.cx + self.w / 2) * img_w
        y2 = (self.cy + self.h / 2) * img_h
        return x1, y1, x2, y2

    @classmethod
    def from_xyxy(
        cls, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
    ) -> "BBox":
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return cls(cx, cy, w, h).clamp()


@dataclass
class Keypoint:
    x: float
    y: float
    visibility: int = 2  # 0=not labeled, 1=labeled hidden, 2=labeled visible

    def clamp(self) -> "Keypoint":
        return Keypoint(
            x=max(0.0, min(1.0, self.x)),
            y=max(0.0, min(1.0, self.y)),
            visibility=self.visibility,
        )

    def to_pixel(self, img_w: int, img_h: int) -> tuple[float, float]:
        return self.x * img_w, self.y * img_h

    @classmethod
    def from_pixel(
        cls, px: float, py: float, img_w: int, img_h: int, visibility: int = 2
    ) -> "Keypoint":
        return cls(px / img_w, py / img_h, visibility).clamp()


@dataclass
class SegmentationMask:
    points: list[tuple[float, float]] = field(default_factory=list)

    def is_closed(self) -> bool:
        return len(self.points) >= 3

    def add_point(self, x: float, y: float) -> None:
        self.points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))

    def remove_point(self, index: int) -> None:
        if 0 <= index < len(self.points):
            self.points.pop(index)

    def update_point(self, index: int, x: float, y: float) -> None:
        if 0 <= index < len(self.points):
            self.points[index] = (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def to_pixel_list(self, img_w: int, img_h: int) -> list[tuple[float, float]]:
        return [(x * img_w, y * img_h) for x, y in self.points]


@dataclass
class Annotation:
    class_id: int
    bbox: Optional[BBox] = None
    keypoints: Optional[list[Keypoint]] = None
    mask: Optional[SegmentationMask] = None

    def has_bbox(self) -> bool:
        return self.bbox is not None

    def has_keypoints(self) -> bool:
        if not self.keypoints:
            return False
        return any(kp is not None for kp in self.keypoints)

    def has_mask(self) -> bool:
        return self.mask is not None and self.mask.is_closed()

    def modalities(self) -> list[Modality]:
        out = []
        if self.has_bbox():
            out.append(Modality.BBOX)
        if self.has_keypoints():
            out.append(Modality.KEYPOINTS)
        if self.has_mask():
            out.append(Modality.SEGMENTATION)
        return out


@dataclass
class ImageAnnotations:
    image_path: str
    label_path: str
    instances: list[Annotation] = field(default_factory=list)
    is_corrupted: bool = False

    def is_annotated(self) -> bool:
        return len(self.instances) > 0

    def is_partially_annotated(self) -> bool:
        if not self.instances:
            return False
        return any(not inst.modalities() for inst in self.instances)

    def add_instance(self, annotation: Annotation) -> None:
        self.instances.append(annotation)

    def remove_instance(self, index: int) -> None:
        if 0 <= index < len(self.instances):
            self.instances.pop(index)
