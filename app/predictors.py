"""Two interchangeable back-ends behind one interface.

LocalPredictor  - loads the Keras champion in this process (Docker Compose stack)
RemotePredictor - forwards the image to the model API hosted in the cloud (Task 7/8)
Both return the same dictionary, so the views and the database do not care which is used.
"""
from __future__ import annotations

import requests


class PredictorError(RuntimeError):
    pass


class LocalPredictor:
    mode = "local"

    def __init__(self, model_path: str, threshold: float | None = None):
        from visionguard.inference import Predictor

        self._p = Predictor(model_path, threshold=threshold)
        self.model_name = self._p.model_name
        self.model_version = self._p.model_version
        self.threshold = self._p.threshold

    def predict(self, data: bytes, filename: str) -> dict:
        return self._p.predict(data).to_dict()

    def health(self) -> dict:
        return {"ok": True, "model_name": self.model_name, "model_version": self.model_version,
                "threshold": self.threshold}


class RemotePredictor:
    mode = "remote"

    def __init__(self, url: str, api_key: str, timeout: float = 30):
        if not url:
            raise PredictorError("MODEL_API_URL is not set")
        self.url, self.api_key, self.timeout = url, api_key, timeout
        self.model_name = self.model_version = "remote"
        self.threshold = None
        try:
            info = self.health()
            self.model_name = info.get("model_name", "remote")
            self.model_version = str(info.get("model_version", "remote"))
            self.threshold = info.get("threshold")
        except Exception:
            pass  # API may still be starting; /health will show it

    def predict(self, data: bytes, filename: str) -> dict:
        try:
            r = requests.post(f"{self.url}/predict", files={"image": (filename, data)},
                              headers={"X-API-Key": self.api_key}, timeout=self.timeout)
        except requests.RequestException as e:
            raise PredictorError(f"Model API unreachable: {e.__class__.__name__}") from e
        if r.status_code != 200:
            try:
                msg = r.json().get("error", r.text)
            except ValueError:
                msg = r.text[:200]
            raise PredictorError(f"Model API returned {r.status_code}: {msg}")
        return r.json()

    def health(self) -> dict:
        r = requests.get(f"{self.url}/health", timeout=10)
        r.raise_for_status()
        d = r.json()
        d["ok"] = d.get("status") == "ok"
        return d
