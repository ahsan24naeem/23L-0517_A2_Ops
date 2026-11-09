"""Inference module: load the champion model ONCE, then predict many times.

Loading a Keras model reads tens of MB from disk and builds the TensorFlow
graph (seconds of work and hundreds of MB of RAM). Doing that per request would
make every request slow and could exhaust memory under concurrent load, so the
model is loaded at start-up and kept in memory.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, DEFAULT_IMG_SIZE
from .preprocess import Source, load_image


@dataclass
class Prediction:
    label: str
    is_ai: bool
    confidence: float      # confidence in the returned label, 0.5 - 1.0
    p_ai: float            # raw sigmoid output = P(AI-generated)
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    def __init__(
        self,
        model_path: str | os.PathLike,
        threshold: float | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        import keras

        from . import models as _custom_layers  # noqa: F401  (registers custom layers)

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        meta = {}
        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.img_size = int(self.model.input_shape[1] or DEFAULT_IMG_SIZE)
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.model_name = model_name or meta.get("model_name", self.model.name)
        self.model_version = str(model_version or meta.get("model_version", "local"))
        self._lock = threading.Lock()
        # warm-up so the first real request is not slow
        self.model.predict(np.zeros((1, self.img_size, self.img_size, 3), np.float32), verbose=0)

    def predict(self, source: Source) -> Prediction:
        t0 = time.perf_counter()
        x = load_image(source, self.img_size).astype(np.float32)[None, ...]
        with self._lock:
            p_ai = float(self.model.predict(x, verbose=0)[0, 0])
        is_ai = p_ai >= self.threshold
        confidence = p_ai if is_ai else 1.0 - p_ai
        return Prediction(
            label=CLASS_NAMES[int(is_ai)],
            is_ai=bool(is_ai),
            confidence=round(confidence, 4),
            p_ai=round(p_ai, 4),
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
