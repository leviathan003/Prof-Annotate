"""
profannotate/core/inference/engine.py
ONNX Runtime wrapper. GPU auto-detect with clean CPU fallback.
Load/unload lifecycle. Never called from UI thread.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from profannotate.config.constants import (
    MODEL_INPUT_SIZE,
    ONNX_MODEL_PATH,
    ONNX_PROVIDERS_PRIORITY,
)

logger = logging.getLogger(__name__)

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
os.environ.setdefault("ORT_DISABLE_ALL_LOGS", "1")


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect C-level stderr to /dev/null for noisy native library output."""
    import sys

    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    old_stderr_fd = os.dup(2)
    try:
        sys.stderr = devnull
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        sys.stderr = old_stderr
        devnull.close()


def _make_session_opts():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0
    opts.inter_op_num_threads = 0
    opts.log_severity_level = 3
    return opts


class InferenceEngine:
    def __init__(self, model_path: str | Path = ONNX_MODEL_PATH) -> None:
        self._model_path = Path(model_path)
        self._session = None
        self._input_name: Optional[str] = None
        self._active_provider: Optional[str] = None

    def load(self) -> None:
        if self._session is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        import onnxruntime as ort

        # Build candidate provider lists in priority order, always ending with CPU
        with _suppress_stderr():
            available = set(ort.get_available_providers())

        candidates = [
            p for p in ONNX_PROVIDERS_PRIORITY if p in available and p != "CPUExecutionProvider"
        ]

        # Try hardware providers first, then pure CPU — first success wins
        attempts = [[p, "CPUExecutionProvider"] for p in candidates] + [["CPUExecutionProvider"]]

        last_exc = None
        for provider_list in attempts:
            try:
                with _suppress_stderr():
                    session = ort.InferenceSession(
                        str(self._model_path),
                        sess_options=_make_session_opts(),
                        providers=provider_list,
                    )
                self._session = session
                self._input_name = session.get_inputs()[0].name
                self._active_provider = session.get_providers()[0]
                logger.info(
                    "Model loaded: %s | provider: %s",
                    self._model_path.name,
                    self._active_provider,
                )
                return
            except Exception as exc:
                last_exc = exc
                logger.debug("Provider list %s failed: %s — trying next.", provider_list, exc)

        raise RuntimeError(f"Failed to load model on any provider: {last_exc}") from last_exc

    def unload(self) -> None:
        self._session = None
        self._input_name = None
        self._active_provider = None
        logger.info("Model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def active_provider(self) -> Optional[str]:
        return self._active_provider

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
