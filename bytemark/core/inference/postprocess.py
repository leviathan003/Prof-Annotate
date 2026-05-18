"""
bytemark/core/inference/postprocess.py
Converts raw ONNX output → list[Annotation].
Handles letterbox inverse transform, NMS, threshold filtering.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bytemark.config.constants import (
    MODEL_CONF_THRESHOLD,
    MODEL_INPUT_SIZE,
    MODEL_IOU_THRESHOLD,
    NUM_KEYPOINTS,
)
from bytemark.core.annotation.models import Annotation, BBox, Keypoint, SegmentationMask

logger = logging.getLogger(__name__)

NM = 32  # mask coefficients per detection
NC = 1  # num classes


def postprocess(
    engine_output,
    orig_w,
    orig_h,
    conf_threshold=MODEL_CONF_THRESHOLD,
    iou_threshold=MODEL_IOU_THRESHOLD,
):
    raw = engine_output["raw"]
    input_shape = engine_output["input_shape"]
    input_h, input_w = input_shape[2], input_shape[3]

    scale = min(input_w / orig_w, input_h / orig_h)
    pad_x = (input_w - orig_w * scale) / 2
    pad_y = (input_h - orig_h * scale) / 2

    # Find the rank-3 combined-head output; shape is (1, C, A) — channel-first
    combined_raw = next((o for o in raw if o.ndim == 3), raw[0])
    predictions = combined_raw[0]  # (C, A)
    # Transpose to (A, C) if channel-first
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    annotations: list[Annotation] = []

    scores = predictions[:, 4]
    mask = scores >= conf_threshold
    predictions = predictions[mask]
    if len(predictions) == 0:
        return annotations

    boxes_xyxy = _cxcywh_to_xyxy(predictions[:, :4])
    keep = _nms(boxes_xyxy, predictions[:, 4], iou_threshold)
    predictions = predictions[keep]

    kpt_offset = 4 + NC + NM  # 37 for segpose (bbox + class_scores + mask_coeffs)
    seg_offset = kpt_offset + NUM_KEYPOINTS * 3

    for pred in predictions:
        cx, cy, bw, bh = pred[:4]

        def inv_x(px):
            return (px - pad_x) / (scale * orig_w)

        def inv_y(py):
            return (py - pad_y) / (scale * orig_h)

        bbox = BBox(
            cx=max(0.0, min(1.0, inv_x(cx))),
            cy=max(0.0, min(1.0, inv_y(cy))),
            w=max(0.0, min(1.0, bw / (scale * orig_w))),
            h=max(0.0, min(1.0, bh / (scale * orig_h))),
        )

        keypoints: list[Keypoint] = []
        for i in range(NUM_KEYPOINTS):
            kx = pred[kpt_offset + i * 3]
            ky = pred[kpt_offset + i * 3 + 1]
            kv = int(pred[kpt_offset + i * 3 + 2])
            keypoints.append(
                Keypoint(
                    x=max(0.0, min(1.0, inv_x(kx))),
                    y=max(0.0, min(1.0, inv_y(ky))),
                    visibility=kv,
                )
            )

        mask_out = None
        if pred.shape[0] > seg_offset + 1:
            seg_vals = pred[seg_offset:]
            if len(seg_vals) >= 6 and len(seg_vals) % 2 == 0:
                pts = [
                    (
                        max(0.0, min(1.0, inv_x(seg_vals[j]))),
                        max(0.0, min(1.0, inv_y(seg_vals[j + 1]))),
                    )
                    for j in range(0, len(seg_vals), 2)
                ]
                mask_out = SegmentationMask(points=pts)

        annotations.append(Annotation(class_id=0, bbox=bbox, keypoints=keypoints, mask=mask_out))

    return annotations


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = _iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)
