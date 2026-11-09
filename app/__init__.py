"""VisionGuard Flask web application (application factory)."""
from __future__ import annotations

import io
import logging
import time

from flask import Flask, Request, abort, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from visionguard.validation import UploadError, validate_upload

from . import db
from .config import Config
from .predictors import LocalPredictor, PredictorError, RemotePredictor

CAVEAT = (
    "VisionGuard is a statistical model trained on about 1,000 web images. It can be wrong - "
    "especially on paintings, illustrations, heavily edited or compressed photos, screenshots and images "
    "from generators it has never seen. Treat the verdict as a hint, never as proof."
)


class InMemoryRequest(Request):
    """Keep uploaded files in RAM. Werkzeug normally spools uploads larger than
    500 KB to a temporary file on disk; the assignment requires that uploaded
    images are never written to disk."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def create_app(config: Config | None = None, predictor=None) -> Flask:
    cfg = config or Config()
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config.update(cfg.as_flask())
    app.config["VG"] = cfg
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("visionguard.web")

    engine = db.make_engine(cfg.DATABASE_URL)
    app.extensions["vg_engine"] = engine

    # Load the model ONCE at start-up (not per request)
    state = {"predictor": predictor, "error": None}
    if predictor is None and cfg.LOAD_MODEL_ON_START:
        try:
            if cfg.INFERENCE_MODE == "remote":
                state["predictor"] = RemotePredictor(cfg.MODEL_API_URL, cfg.MODEL_API_KEY, cfg.MODEL_API_TIMEOUT)
            else:
                state["predictor"] = LocalPredictor(cfg.MODEL_PATH, cfg.THRESHOLD)
            p = state["predictor"]
            log.info("predictor ready: mode=%s model=%s version=%s", p.mode, p.model_name, p.model_version)
        except Exception as e:  # app still starts so /health can report the problem
            state["error"] = f"{type(e).__name__}: {e}"
            log.error("MODEL LOAD FAILED: %s", state["error"])
    app.extensions["vg_state"] = state

    @app.context_processor
    def inject():
        p = state["predictor"]
        return {"mode": getattr(p, "mode", cfg.INFERENCE_MODE), "model_name": getattr(p, "model_name", "n/a"),
                "model_version": getattr(p, "model_version", "n/a"), "max_mb": cfg.MAX_UPLOAD_MB,
                "allowed": ", ".join(sorted(cfg.ALLOWED_EXTENSIONS))}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        data = f.read() if f else b""
        info = validate_upload(f.filename if f else None, data, app.config["MAX_CONTENT_LENGTH"],
                               cfg.ALLOWED_EXTENSIONS)
        p = state["predictor"]
        if p is None:
            raise UploadError(503, "Model unavailable", f"The model is not loaded: {state['error']}")
        t0 = time.perf_counter()
        try:
            res = p.predict(data, info.safe_name)
        except PredictorError as e:
            raise UploadError(502, "Model service error", str(e)) from e
        latency = (time.perf_counter() - t0) * 1000
        pid = db.insert_prediction(
            engine, filename=info.safe_name, file_sha256=info.sha256, file_size_bytes=info.size_bytes,
            image_width=info.width, image_height=info.height, label=res["label"], is_ai=bool(res["is_ai"]),
            confidence=float(res["confidence"]), p_ai=float(res["p_ai"]), threshold=float(res["threshold"]),
            model_name=str(res["model_name"]), model_version=str(res["model_version"]),
            inference_mode=p.mode, latency_ms=round(latency, 1))
        del data  # the image bytes are discarded; only metadata is stored
        log.info("prediction id=%s file=%s label=%s conf=%.3f mode=%s", pid, info.safe_name, res["label"],
                 res["confidence"], p.mode)
        return redirect(url_for("result", pid=pid), code=303)

    @app.get("/result/<int:pid>")
    def result(pid: int):
        row = db.get_prediction(engine, pid)
        if not row:
            abort(404)
        return render_template("result.html", r=row, caveat=CAVEAT)

    @app.get("/history")
    def history():
        page = max(int(request.args.get("page", 1) or 1), 1)
        rows = db.list_predictions(engine, limit=25, offset=(page - 1) * 25)
        return render_template("history.html", rows=rows, stats=db.stats(engine), page=page)

    @app.get("/api/history")
    def history_json():
        rows = db.list_predictions(engine, limit=min(int(request.args.get("limit", 50)), 500))
        for r in rows:
            r["created_at"] = r["created_at"].isoformat()
        return jsonify(rows)

    @app.get("/health")
    def health():
        p = state["predictor"]
        model_ok, details = False, {"error": state["error"]}
        if p is not None:
            try:
                details = p.health()
                model_ok = bool(details.get("ok"))
            except Exception as e:
                details = {"error": f"{type(e).__name__}: {e}"}
        db_ok = db.ping(engine)
        body = {
            "status": "ok" if (model_ok and db_ok) else "degraded",
            "inference_mode": cfg.INFERENCE_MODE,
            "model": {"loaded": model_ok, "name": getattr(p, "model_name", None),
                      "version": getattr(p, "model_version", None), "threshold": getattr(p, "threshold", None),
                      "path": cfg.MODEL_PATH if cfg.INFERENCE_MODE == "local" else None,
                      "api_url": cfg.MODEL_API_URL or None, **{k: v for k, v in details.items() if k == "error"}},
            "database": {"ok": db_ok, "backend": engine.url.get_backend_name()},
            "debug": app.debug,
        }
        return jsonify(body), 200 if body["status"] == "ok" else 503

    # ---------- friendly errors ----------
    @app.errorhandler(UploadError)
    def upload_error(e: UploadError):
        log.warning("rejected upload: %s - %s", e.title, e.message)
        return render_template("error.html", status=e.status, title=e.title, message=e.message), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        return render_template("error.html", status=413, title="File too large",
                               message=f"Uploads are limited to {cfg.MAX_UPLOAD_MB:g} MB."), 413

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        return render_template("error.html", status=e.code, title=e.name, message=e.description), e.code

    @app.errorhandler(Exception)
    def server_error(e: Exception):
        log.exception("unhandled error")
        return render_template("error.html", status=500, title="Something went wrong",
                               message="An unexpected error occurred. Please try again."), 500

    return app
