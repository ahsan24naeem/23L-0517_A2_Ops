"""VisionGuard model API - the service deployed to Oracle Cloud (Task 7).

Endpoints
  GET  /         service info (no auth)
  GET  /health   liveness + model status (no auth)
  POST /predict  multipart field "image"; header "X-API-Key: <MODEL_API_KEY>" required
Stateless: images are processed in memory and never stored; one log line per request.
"""
from __future__ import annotations

import hmac
import io
import logging
import os
import time

from flask import Flask, Request, jsonify, request
from werkzeug.exceptions import HTTPException

from visionguard.validation import UploadError, validate_upload


class InMemoryRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def create_api(predictor=None) -> Flask:
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    max_mb = float(os.environ.get("MAX_UPLOAD_MB", "5"))
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)
    api_key = os.environ.get("MODEL_API_KEY", "")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("visionguard.api")
    if not api_key:
        log.warning("MODEL_API_KEY is empty - /predict will reject every request")

    state = {"p": predictor, "error": None, "started": time.time(), "count": 0}
    if predictor is None:
        try:
            from visionguard.inference import Predictor

            thr = os.environ.get("THRESHOLD", "").strip()
            state["p"] = Predictor(os.environ.get("MODEL_PATH", "models/champion.keras"),
                                   threshold=float(thr) if thr else None)
            log.info("model loaded: %s v%s", state["p"].model_name, state["p"].model_version)
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"
            log.error("MODEL LOAD FAILED: %s", state["error"])

    def info():
        p = state["p"]
        return {"service": "visionguard-model-api", "project": "23L-0517_A2_Ops",
                "model_name": getattr(p, "model_name", None), "model_version": getattr(p, "model_version", None),
                "threshold": getattr(p, "threshold", None), "img_size": getattr(p, "img_size", None)}

    @app.get("/")
    def root():
        endpoints = {"GET /health": "status", "POST /predict": "multipart 'image' + X-API-Key header"}
        return jsonify({**info(), "endpoints": endpoints})

    @app.get("/health")
    def health():
        ok = state["p"] is not None
        return jsonify({"status": "ok" if ok else "error", **info(), "error": state["error"],
                        "uptime_s": round(time.time() - state["started"]), "requests_served": state["count"]}), \
            200 if ok else 503

    @app.post("/predict")
    def predict():
        sent = request.headers.get("X-API-Key", "")
        if not api_key or not hmac.compare_digest(sent, api_key):  # constant-time comparison
            return jsonify({"error": "missing or invalid API key"}), 401
        if state["p"] is None:
            return jsonify({"error": f"model not loaded: {state['error']}"}), 503
        f = request.files.get("image")
        data = f.read() if f else b""
        v = validate_upload(f.filename if f else None, data, app.config["MAX_CONTENT_LENGTH"])
        res = state["p"].predict(data).to_dict()
        state["count"] += 1
        log.info("predict file=%s sha=%s label=%s conf=%.4f latency=%sms", v.safe_name, v.sha256[:12],
                 res["label"], res["confidence"], res["latency_ms"])
        return jsonify({**res, "filename": v.safe_name, "sha256": v.sha256, "width": v.width, "height": v.height})

    @app.errorhandler(UploadError)
    def bad_upload(e: UploadError):
        return jsonify({"error": e.message, "title": e.title}), e.status

    @app.errorhandler(HTTPException)
    def http_err(e: HTTPException):
        return jsonify({"error": e.description, "title": e.name}), e.code

    return app
