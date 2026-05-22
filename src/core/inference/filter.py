"""
bytemark/core/inference/filter.py
Filters Annotation objects by selected modalities.
"""

from __future__ import annotations

from src.core.annotation.models import Annotation, Modality


def filter_by_modality(
    annotations: list[Annotation],
    modalities: set[Modality],
) -> list[Annotation]:
    """
    Return new Annotation objects containing only the requested modalities.
    Original list is not mutated.
    """
    result = []
    for ann in annotations:
        new = Annotation(class_id=ann.class_id)
        if Modality.BBOX in modalities and ann.has_bbox():
            new.bbox = ann.bbox
        if Modality.KEYPOINTS in modalities and ann.has_keypoints():
            new.keypoints = ann.keypoints
        if Modality.SEGMENTATION in modalities and ann.has_mask():
            new.mask = ann.mask
        # Only include if at least one modality survived
        if new.bbox or new.keypoints or new.mask:
            result.append(new)
    return result
