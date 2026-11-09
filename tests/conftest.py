import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakePredictor:
    """Stands in for the Keras model so tests run without TensorFlow."""
    mode = "local"
    model_name = "fake-model"
    model_version = "0"
    threshold = 0.5

    def __init__(self, p_ai=0.8):
        self.p_ai = p_ai

    def predict(self, data, filename=None):
        is_ai = self.p_ai >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real", "is_ai": is_ai,
                "confidence": self.p_ai if is_ai else 1 - self.p_ai, "p_ai": self.p_ai,
                "threshold": self.threshold, "model_name": self.model_name,
                "model_version": self.model_version, "latency_ms": 1.0}

    def to_dict(self):
        return self

    def health(self):
        return {"ok": True}


def make_image_bytes(fmt="PNG", size=(64, 48), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return make_image_bytes()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from app import create_app
    from app.config import Config

    application = create_app(Config(), predictor=FakePredictor())
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()
