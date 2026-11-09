import io

from tests.conftest import make_image_bytes


def upload(client, data, name="cat.png"):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_pages_render(client):
    assert client.get("/").status_code == 200
    assert b"Prediction history" in client.get("/history").data


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["database"]["ok"] and body["debug"] is False


def test_predict_creates_history_row(client, png_bytes):
    r = upload(client, png_bytes)
    assert r.status_code == 303
    page = client.get(r.headers["Location"])
    assert page.status_code == 200 and b"AI-generated" in page.data and b"Caveat" in page.data
    rows = client.get("/api/history").get_json()
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "AI-generated" and row["model_name"] == "fake-model"
    assert row["filename"] == "cat.png" and abs(row["confidence"] - 0.8) < 1e-6


def test_wrong_type_rejected(client):
    r = upload(client, b"hello", "notes.txt")
    assert r.status_code == 415 and b"Unsupported file type" in r.data


def test_undecodable_rejected(client):
    r = upload(client, b"GIF89a-but-not-really", "broken.gif")
    assert r.status_code == 422 and b"Not a valid image" in r.data


def test_too_large_rejected(client):
    big = make_image_bytes("BMP", size=(800, 800))   # ~1.9 MB > 1 MB limit
    r = upload(client, big, "big.bmp")
    assert r.status_code == 413 and b"too large" in r.data.lower()


def test_rejections_are_not_logged(client):
    upload(client, b"hello", "notes.txt")
    assert client.get("/api/history").get_json() == []
