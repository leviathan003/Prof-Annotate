"""
bytemark/core/inference/engine.py
ONNX Runtime wrapper. GPU auto-detect. Load/unload lifecycle.
Never called from UI thread.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from bytemark.config.constants import (
    MODEL_INPUT_SIZE,
    NUM_KEYPOINTS,
    ONNX_MODEL_PATH,
    ONNX_PROVIDERS_PRIORITY,
)

logger = logging.getLogger(__name__)


def resolve_providers() -> list[str]:
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        chosen = [p for p in ONNX_PROVIDERS_PRIORITY if p in available]
        if not chosen:
            chosen = ["CPUExecutionProvider"]
        logger.info("ONNX providers: %s", chosen)
        return chosen
    except ImportError:
        return ["CPUExecutionProvider"]


class InferenceEngine:
    def __init__(self, model_path: str | Path = ONNX_MODEL_PATH) -> None:
        self._model_path = Path(model_path)
        self._session = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        if self._session is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0
        opts.inter_op_num_threads = 0

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=resolve_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("Model loaded: %s", self._model_path)

    def unload(self) -> None:
        self._session = None
        self._input_name = None
        logger.info("Model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def run(self, image_rgb: np.ndarray) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Call load() before run().")
        blob = _preprocess(image_rgb)
        outputs = self._session.run(None, {self._input_name: blob})
        return {"raw": outputs, "input_shape": blob.shape}


def _preprocess(image_rgb: np.ndarray) -> np.ndarray:
    import cv2

    tw, th = MODEL_INPUT_SIZE
    h, w = image_rgb.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    px, py = (tw - nw) // 2, (th - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    return np.expand_dims(blob, 0)
