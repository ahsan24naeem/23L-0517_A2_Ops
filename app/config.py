"""All settings come from environment variables (12-factor style).

Nothing environment-specific or secret is written in code, so the same image
runs on a laptop, in Compose and in the cloud with different .env files.
"""
from __future__ import annotations

import os


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    def __init__(self) -> None:
        env = os.environ
        self.MODEL_PATH = env.get("MODEL_PATH", "models/champion.keras")
        thr = env.get("THRESHOLD", "").strip()
        self.THRESHOLD = float(thr) if thr else None          # None -> use value stored with the model
        self.DATABASE_URL = env.get("DATABASE_URL", "sqlite:///instance/visionguard.db")
        self.INFERENCE_MODE = env.get("INFERENCE_MODE", "local")   # local | remote
        self.MODEL_API_URL = env.get("MODEL_API_URL", "").rstrip("/")
        self.MODEL_API_KEY = env.get("MODEL_API_KEY", "")
        self.MODEL_API_TIMEOUT = float(env.get("MODEL_API_TIMEOUT", "30"))
        self.MAX_UPLOAD_MB = float(env.get("MAX_UPLOAD_MB", "5"))
        self.SECRET_KEY = env.get("SECRET_KEY", os.urandom(16).hex())
        self.DEBUG = _bool("FLASK_DEBUG", False)
        self.LOAD_MODEL_ON_START = _bool("LOAD_MODEL_ON_START", True)
        self.ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}

    def as_flask(self) -> dict:
        return {
            "SECRET_KEY": self.SECRET_KEY,
            "MAX_CONTENT_LENGTH": int(self.MAX_UPLOAD_MB * 1024 * 1024),
            "DEBUG": self.DEBUG,
        }
