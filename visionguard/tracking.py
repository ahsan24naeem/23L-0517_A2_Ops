"""MLflow helpers: tracking URI from environment, reproducibility tags, manual Keras callback."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import keras
import mlflow

from .config import ROOT


def setup_mlflow(experiment: str) -> str:
    """Configure MLflow ONLY from environment variables (never hard-coded secrets).

    MLFLOW_TRACKING_URI       e.g. https://dagshub.com/<user>/23L-0517_A2_Ops.mlflow
    MLFLOW_TRACKING_USERNAME  DagsHub user name
    MLFLOW_TRACKING_PASSWORD  DagsHub access token
    If MLFLOW_TRACKING_URI is unset, a local SQLite store (mlflow.db) is used.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def git_commit() -> str:
    return _cmd(["git", "rev-parse", "HEAD"])


def git_dirty() -> bool:
    if git_commit() == "unknown":
        return False
    return bool(_cmd(["git", "status", "--porcelain", "--untracked-files=no"]).strip())


def dvc_data_version() -> str:
    """md5 of the dataset directory as recorded in data/raw.dvc (the DVC pointer file)."""
    dvc_file = ROOT / "data" / "raw.dvc"
    if dvc_file.exists():
        for line in dvc_file.read_text().splitlines():
            line = line.strip().lstrip("- ")
            if line.startswith("md5:"):
                return line.split(":", 1)[1].strip()
    return "untracked"


def splits_hash() -> str:
    h = hashlib.md5()
    for name in ["train", "val", "test"]:
        p = ROOT / "splits" / f"{name}.csv"
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def standard_tags(seed: int, extra: dict | None = None) -> dict:
    tags = {
        "project": "23L-0517_A2_Ops",
        "git_commit": git_commit(),
        "git_dirty": str(git_dirty()),
        "dvc_data_version": dvc_data_version(),
        "splits_md5": splits_hash(),
        "seed": str(seed),
        "python": sys.version.split()[0],
        "tensorflow": __import__("tensorflow").__version__,
        "keras": keras.__version__,
        "author": "Ahsan Naeem (23L-0517)",
    }
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def pip_freeze(path: Path) -> Path:
    out = _cmd([sys.executable, "-m", "pip", "freeze"])
    if out == "unknown":
        out = _cmd(["uv", "pip", "freeze", "--python", sys.executable])
    path.write_text(out + "\n")
    return path


class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual logging: every epoch's train/val metrics and the learning rate.

    `offset` lets the fine-tuning stage continue the epoch axis of the first
    stage, so both stages appear as one continuous curve in MLflow.
    """

    def __init__(self, offset: int = 0, stage: str = ""):
        super().__init__()
        self.offset = offset
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        try:
            logs["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        except Exception:
            pass
        step = self.offset + epoch + 1
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=step)
