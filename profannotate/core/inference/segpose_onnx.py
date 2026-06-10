"""ONNX-runtime wrapper for the exported YOLO11 SegPose model.

Mirrors `SegmentPosePredictor.postprocess`: NMS on the combined head output,
mask decoding against the proto tensor, and letterbox-aware scaling of boxes,
masks, and keypoints back to original image coordinates.

The wrapper is framework-light: onnxruntime for inference, numpy/torch for
postprocess (torch only for NMS + the mask matmul / interpolation, which are
cheap and let us reuse `torchvision.ops.nms` and `F.interpolate`).

Usage:
    from segpose_onnx import SegPoseONNX

    model = SegPoseONNX("best.onnx", imgsz=640, conf=0.30, iou=0.5)
    res = model(frame_bgr)              # frame: HxWx3 BGR uint8
    res.boxes   # (N, 6) float32 — x1, y1, x2, y2, conf, cls in orig pixels
    res.kpts    # (N, K, 3) float32 — x, y, v
    res.masks   # (N, H_orig, W_orig) bool, or None if model has no masks
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision


@dataclass
class SegPoseResult:
    boxes: np.ndarray              # (N, 6) — xyxy, conf, cls (orig image coords)
    kpts: Optional[np.ndarray]     # (N, K, 3) — x, y, v (orig image coords)
    masks: Optional[np.ndarray]    # (N, H_orig, W_orig) bool
    orig_shape: Tuple[int, int]    # (H, W)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize-with-padding so longest side equals new_shape; returns (img, ratio, (dw, dh))."""
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


# ---------------------------------------------------------------------------
# Postprocessing helpers (numpy/torch — mirror ultralytics.utils.ops)
# ---------------------------------------------------------------------------


def _scale_boxes_inplace(boxes_xyxy: np.ndarray, ratio: float, pad: Tuple[float, float]) -> None:
    boxes_xyxy[:, [0, 2]] -= pad[0]
    boxes_xyxy[:, [1, 3]] -= pad[1]
    boxes_xyxy[:, :4] /= ratio


def _scale_kpts_inplace(kpts_xy: np.ndarray, ratio: float, pad: Tuple[float, float]) -> None:
    kpts_xy[..., 0] -= pad[0]
    kpts_xy[..., 1] -= pad[1]
    kpts_xy /= ratio


def _clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> None:
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def _process_masks(
    proto: np.ndarray,           # (nm, mH, mW)
    coeffs: np.ndarray,          # (N, nm)
    boxes_input: np.ndarray,     # (N, 4) xyxy in input-image coords
    input_shape: Tuple[int, int],
    pad: Tuple[float, float],
    ratio: float,
    orig_shape: Tuple[int, int],
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Decode masks → upsample to input shape → crop to bbox → un-letterbox → resize to orig."""
    nm, mH, mW = proto.shape
    if coeffs.shape[0] == 0:
        return np.zeros((0, orig_shape[0], orig_shape[1]), dtype=bool)

    proto_t = torch.from_numpy(proto).to(device, dtype=torch.float32, non_blocking=True)
    coeffs_t = torch.from_numpy(coeffs).to(device, dtype=torch.float32, non_blocking=True)
    masks = (coeffs_t @ proto_t.view(nm, -1)).sigmoid().view(-1, mH, mW)

    masks = F.interpolate(masks.unsqueeze(0), size=input_shape, mode="bilinear", align_corners=False)[0]

    # Crop to bbox in input-image coords
    yy = torch.arange(input_shape[0], dtype=torch.float32, device=device).view(1, -1, 1)
    xx = torch.arange(input_shape[1], dtype=torch.float32, device=device).view(1, 1, -1)
    b = torch.from_numpy(boxes_input).to(device, dtype=torch.float32, non_blocking=True)
    x1, y1, x2, y2 = b[:, 0:1, None], b[:, 1:2, None], b[:, 2:3, None], b[:, 3:4, None]
    inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    masks = masks * inside.float()

    # Un-letterbox: crop the padded border, then resize to orig
    top = int(round(pad[1] - 0.1))
    left = int(round(pad[0] - 0.1))
    h_unpad = input_shape[0] - 2 * top
    w_unpad = input_shape[1] - 2 * left
    masks = masks[:, top : top + h_unpad, left : left + w_unpad]
    masks = F.interpolate(masks.unsqueeze(0), size=orig_shape, mode="bilinear", align_corners=False)[0]

    return (masks > 0.5).cpu().numpy()


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------


class SegPoseONNX:
    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf: float = 0.30,
        iou: float = 0.5,
        max_det: int = 50,
        nc: int = 1,
        nm: int = 32,
        kpt_shape: Tuple[int, int] = (19, 3),
        device: str = "cuda",
    ):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.nc = nc
        self.nm = nm
        self.kK, self.kD = kpt_shape
        self.nk = self.kK * self.kD
        self.torch_device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        # Identify the combined-head output (rank-3, C = 4+nc+nm+nk) and mask proto
        # (rank-4, C = nm, largest spatial extent). The exporter may emit per-scale
        # intermediates with the same channel count as the proto (mask-coeff heads),
        # so we run a tiny dry inference at self.imgsz and pick by realized shape.
        combined_ch = 4 + self.nc + self.nm + self.nk
        dry = self.session.run(
            None,
            {self.input_name: np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)},
        )
        out_meta = self.session.get_outputs()
        self.combined_name = None
        self.proto_name = None
        best_spatial = -1
        for arr, meta in zip(dry, out_meta):
            shp = arr.shape
            if self.combined_name is None and len(shp) == 3 and shp[1] == combined_ch:
                self.combined_name = meta.name
            elif len(shp) == 4 and shp[1] == self.nm:
                spatial = shp[2] * shp[3]
                if spatial > best_spatial:
                    best_spatial = spatial
                    self.proto_name = meta.name
        if self.combined_name is None or self.proto_name is None:
            raise RuntimeError(
                f"could not locate combined/proto outputs (combined_ch={combined_ch}, nm={self.nm}); "
                f"available: {[(o.name, o.shape) for o in self.session.get_outputs()]}"
            )
        print(
            f"SegPoseONNX ready | providers={self.session.get_providers()} | "
            f"combined='{self.combined_name}' proto='{self.proto_name}'"
        )

    # ----- inference -----

    def _preprocess(self, img_bgr: np.ndarray):
        lb, ratio, pad = letterbox(img_bgr, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None]  # (1, 3, H, W)
        return np.ascontiguousarray(chw), ratio, pad, lb.shape[:2]

    def __call__(self, img_bgr: np.ndarray) -> SegPoseResult:
        orig_shape = img_bgr.shape[:2]
        tensor, ratio, pad, input_shape = self._preprocess(img_bgr)

        combined, proto = self.session.run(
            [self.combined_name, self.proto_name], {self.input_name: tensor}
        )
        # combined: (1, 4 + nc + nm + nk, A)   proto: (1, nm, mH, mW)
        pred = combined[0].T  # (A, C)

        # Filter by max class score
        cls_scores = pred[:, 4 : 4 + self.nc]
        score_max = cls_scores.max(axis=1)
        cls_idx = cls_scores.argmax(axis=1)
        keep_mask = score_max > self.conf
        pred = pred[keep_mask]
        score_max = score_max[keep_mask]
        cls_idx = cls_idx[keep_mask]
        if pred.shape[0] == 0:
            return SegPoseResult(
                boxes=np.zeros((0, 6), dtype=np.float32),
                kpts=np.zeros((0, self.kK, 3), dtype=np.float32),
                masks=np.zeros((0, orig_shape[0], orig_shape[1]), dtype=bool),
                orig_shape=orig_shape,
            )

        # Head emits xywh (center x, center y, w, h) — convert to xyxy.
        xywh = pred[:, :4].astype(np.float32)
        boxes_xyxy = np.empty_like(xywh)
        boxes_xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes_xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes_xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes_xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        mask_coeffs = pred[:, 4 + self.nc : 4 + self.nc + self.nm]
        kpts_flat = pred[:, 4 + self.nc + self.nm :]

        # NMS (per-class)
        b_t = torch.from_numpy(boxes_xyxy)
        s_t = torch.from_numpy(score_max.astype(np.float32))
        c_t = torch.from_numpy(cls_idx.astype(np.int64))
        keep = torchvision.ops.batched_nms(b_t, s_t, c_t, self.iou)[: self.max_det].numpy()

        boxes_xyxy = boxes_xyxy[keep]
        score_max = score_max[keep]
        cls_idx = cls_idx[keep]
        mask_coeffs = mask_coeffs[keep]
        kpts_flat = kpts_flat[keep]

        # Decode masks in input-image space, then scale to orig
        masks = _process_masks(
            proto=proto[0],
            coeffs=mask_coeffs.astype(np.float32),
            boxes_input=boxes_xyxy.copy(),
            input_shape=input_shape,
            pad=pad,
            ratio=ratio,
            orig_shape=orig_shape,
            device=self.torch_device,
        )

        # Scale boxes to orig
        boxes_orig = boxes_xyxy.copy()
        _scale_boxes_inplace(boxes_orig, ratio, pad)
        _clip_boxes(boxes_orig, orig_shape)

        # Scale kpts to orig
        kpts = kpts_flat.reshape(-1, self.kK, self.kD).astype(np.float32)
        _scale_kpts_inplace(kpts[..., :2], ratio, pad)

        boxes_out = np.concatenate(
            [boxes_orig, score_max[:, None].astype(np.float32), cls_idx[:, None].astype(np.float32)],
            axis=1,
        )
        return SegPoseResult(boxes=boxes_out, kpts=kpts, masks=masks, orig_shape=orig_shape)


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    m = SegPoseONNX(args.onnx, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    # warmup
    for _ in range(3):
        m(img)
    t0 = time.perf_counter()
    res = m(img)
    dt = (time.perf_counter() - t0) * 1000
    print(f"{dt:.1f} ms | boxes={len(res.boxes)} kpts={None if res.kpts is None else res.kpts.shape} "
          f"masks={None if res.masks is None else res.masks.shape}")
