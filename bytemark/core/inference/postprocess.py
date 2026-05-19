"""
bytemark/core/inference/postprocess.py
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from bytemark.config.constants import (
    MODEL_CONF_THRESHOLD,
    MODEL_INPUT_SIZE,
    MODEL_IOU_THRESHOLD,
    NUM_KEYPOINTS,
)
from bytemark.core.annotation.models import Annotation, BBox, Keypoint, SegmentationMask

logger = logging.getLogger(__name__)

_NC = 1
_NM = 32


def postprocess(
    engine_output: dict[str, Any],
    orig_w: int,
    orig_h: int,
    conf_threshold: float = MODEL_CONF_THRESHOLD,
    iou_threshold: float = MODEL_IOU_THRESHOLD,
) -> list[Annotation]:
    raw = engine_output["raw"]
    input_shape = engine_output["input_shape"]
    input_h, input_w = input_shape[2], input_shape[3]

    scale = min(input_w / orig_w, input_h / orig_h)
    pad_x = (input_w - orig_w * scale) / 2
    pad_y = (input_h - orig_h * scale) / 2

    # Locate combined head (rank-3) and proto tensor (rank-4, dim-1 == NM)
    combined_raw = next((o for o in raw if o.ndim == 3), None)
    proto_raw = next((o for o in raw if o.ndim == 4 and o.shape[1] == _NM), None)
    if combined_raw is None:
        logger.warning("postprocess: no rank-3 combined output found")
        return []

    predictions = combined_raw[0]  # (C, A) or (A, C)
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T  # ensure (A, C)

    scores = predictions[:, 4]
    mask = scores >= conf_threshold
    predictions = predictions[mask]
    if len(predictions) == 0:
        return []

    boxes_xyxy = _cxcywh_to_xyxy(predictions[:, :4])
    keep = _nms(boxes_xyxy, predictions[:, 4], iou_threshold)
    predictions = predictions[keep]
    boxes_xyxy = boxes_xyxy[keep]

    kpt_offset = 4 + _NC + _NM  # 37

    annotations: list[Annotation] = []
    for det_idx, pred in enumerate(predictions):
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

        mask_out = _decode_mask(
            pred,
            det_idx,
            boxes_xyxy,
            proto_raw,
            input_w,
            input_h,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        )

        annotations.append(Annotation(class_id=0, bbox=bbox, keypoints=keypoints, mask=mask_out))

    return annotations


def _decode_mask(
    pred, det_idx, boxes_xyxy, proto_raw, input_w, input_h, pad_x, pad_y, orig_w, orig_h
):
    if proto_raw is None:
        return None
    proto = proto_raw[0]  # (NM, mH, mW)
    nm, mH, mW = proto.shape
    coeffs = pred[4 + _NC : 4 + _NC + nm].astype(np.float32)

    # Decode: sigmoid(coeffs @ proto.reshape(nm, -1))
    mask_flat = coeffs @ proto.reshape(nm, -1)
    mask_map = (1.0 / (1.0 + np.exp(-mask_flat))).reshape(mH, mW).astype(np.float32)

    # Upsample to input size
    mask_input = cv2.resize(mask_map, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    # Crop to bbox in input-image space
    bx1, by1, bx2, by2 = boxes_xyxy[det_idx]
    x1c, y1c = int(max(0, bx1)), int(max(0, by1))
    x2c, y2c = int(min(input_w, bx2)), int(min(input_h, by2))
    crop = np.zeros_like(mask_input)
    crop[y1c:y2c, x1c:x2c] = mask_input[y1c:y2c, x1c:x2c]

    # Remove letterbox padding
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    h_unpad = input_h - 2 * top
    w_unpad = input_w - 2 * left
    crop = crop[top : top + h_unpad, left : left + w_unpad]
    mask_orig = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    binary = (mask_orig > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) < 3:
        return None

    pts = [(float(p[0][0]) / orig_w, float(p[0][1]) / orig_h) for p in approx]
    return SegmentationMask(points=pts)


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
