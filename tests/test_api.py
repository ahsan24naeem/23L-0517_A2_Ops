import io

import pytest

from tests.conftest import FakePredictor


class _P(FakePredictor):
    img_size = 224

    def predict(self, data, filename=None):
        res = super().predict(data)
        return type("R", (), {"to_dict": lambda self_: res})()


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "secret-key")
    from api import create_api
    return create_api(predictor=_P()).test_client()


def test_health_open(api_client):
    assert api_client.get("/health").get_json()["status"] == "ok"


def test_predict_requires_key(api_client, png_bytes):
    r = api_client.post("/predict", data={"image": (io.BytesIO(png_bytes), "a.png")})
    assert r.status_code == 401


def test_predict_ok(api_client, png_bytes):
    r = api_client.post("/predict", data={"image": (io.BytesIO(png_bytes), "a.png")},
                        headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] == "AI-generated" and body["filename"] == "a.png"


def test_predict_bad_file(api_client):
    r = api_client.post("/predict", data={"image": (io.BytesIO(b"x"), "a.txt")},
                        headers={"X-API-Key": "secret-key"})
    assert r.status_code == 415
