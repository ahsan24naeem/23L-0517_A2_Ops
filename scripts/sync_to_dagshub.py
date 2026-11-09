"""Copy every MLflow run from the local store (mlflow.db + mlruns) to DagsHub.

Experiments were executed on a training machine without access to DagsHub, so
they were tracked in a local MLflow store first. This script replays each run
on the DagsHub tracking server WITHOUT retraining: same run name, start/end
time, parameters, tags, full per-epoch metric histories and all artifacts
(plots, split lists, package list, Keras model, MLflow model). It is
idempotent: runs already copied (tag synced_from_run_id) are skipped.

Usage (credentials from .env, never from code):
  python scripts/sync_to_dagshub.py --experiments VisionGuard VisionGuard-tuning toy-fashion-mnist
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402
from mlflow.entities import Metric, Param, RunTag  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from visionguard.config import ROOT  # noqa: E402

SKIP_TAGS = {"mlflow.user", "mlflow.source.name", "mlflow.log-model.history"}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def copy_run(src: MlflowClient, dst: MlflowClient, run, dst_exp_id: str) -> str:
    tags = {k: v for k, v in run.data.tags.items() if k not in SKIP_TAGS}
    tags["synced_from_run_id"] = run.info.run_id
    tags["synced_from"] = "local MLflow store (training workspace)"
    new = dst.create_run(dst_exp_id, start_time=run.info.start_time, tags=tags, run_name=run.info.run_name)
    rid = new.info.run_id
    params = [Param(k, str(v)[:6000]) for k, v in run.data.params.items()]
    for c in chunks(params, 100):
        dst.log_batch(rid, params=c)
    metrics = []
    for key in run.data.metrics:
        metrics += [Metric(m.key, m.value, m.timestamp, m.step) for m in src.get_metric_history(run.info.run_id, key)]
    for c in chunks(metrics, 1000):
        dst.log_batch(rid, metrics=c)
    with tempfile.TemporaryDirectory() as d:
        local = src.download_artifacts(run.info.run_id, "", d)
        if any(Path(local).iterdir()):
            dst.log_artifacts(rid, local)
    dst.log_batch(rid, tags=[RunTag("mlflow.source.name", "scripts/train.py")])
    dst.set_terminated(rid, status=run.info.status, end_time=run.info.end_time)
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["VisionGuard", "VisionGuard-tuning"])
    ap.add_argument("--source", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    target = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not target.startswith("http"):
        sys.exit("Set MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD in .env first")
    src, dst = MlflowClient(tracking_uri=a.source), MlflowClient(tracking_uri=target)
    print(f"source: {a.source}\ntarget: {target}")
    for name in a.experiments:
        exp = src.get_experiment_by_name(name)
        if exp is None:
            print(f"- experiment {name!r} not found locally, skipped")
            continue
        dexp = dst.get_experiment_by_name(name)
        dexp_id = dexp.experiment_id if dexp else dst.create_experiment(name)
        done = {r.data.tags.get("synced_from_run_id") for r in dst.search_runs([dexp_id], max_results=1000)}
        runs = src.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"], max_results=1000)
        for run in runs:
            if run.info.run_id in done:
                print(f"  = {name}/{run.info.run_name} already synced")
                continue
            rid = copy_run(src, dst, run, dexp_id)
            print(f"  + {name}/{run.info.run_name}  {run.info.run_id[:8]} -> {rid[:8]}")
    print("done. Now run: python scripts/register_model.py   (registers versions on DagsHub)")


if __name__ == "__main__":
    main()
