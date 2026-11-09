"""Task 3.6 - model registry.

Registers two versions of the "visionguard" model and gives the best one the
alias "champion" (the other gets "challenger"). Best = highest validation
ROC-AUC (validation, not test, so the test set stays an unbiased final check).
Also exports the champion to models/champion.keras (+ .json metadata) - the file
the web app and the cloud API load, so serving never calls DagsHub per request.

Usage: python scripts/register_model.py [--challenger cnn_baseline]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.tracking import MlflowClient  # noqa: E402

from visionguard.config import MODELS_DIR  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402

MODEL_NAME = "visionguard"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="VisionGuard")
    ap.add_argument("--challenger", default="cnn_baseline", help="run name registered as the other version")
    ap.add_argument("--metric", default="val_roc_auc")
    a = ap.parse_args()
    uri = setup_mlflow(a.experiment)
    c = MlflowClient()
    exp = c.get_experiment_by_name(a.experiment)
    runs = c.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'",
                         order_by=[f"metrics.{a.metric} DESC", "metrics.val_f1 DESC"])
    runs = [r for r in runs if a.metric in r.data.metrics]
    best = runs[0]
    challenger = next((r for r in runs if r.info.run_name == a.challenger and r.info.run_id != best.info.run_id),
                      runs[1])
    print(f"tracking: {uri.split('@')[-1]}")
    print(f"champion  : {best.info.run_name} ({a.metric}={best.data.metrics[a.metric]:.4f})")
    print(f"challenger: {challenger.info.run_name} ({a.metric}={challenger.data.metrics[a.metric]:.4f})")

    try:
        c.create_registered_model(MODEL_NAME, description="VisionGuard AI-vs-real image classifier (23L-0517_A2_Ops)")
    except Exception:
        pass  # already exists

    versions = {}
    for role, run in [("challenger", challenger), ("champion", best)]:  # champion registered last = newest
        mv = c.create_model_version(MODEL_NAME, f"runs:/{run.info.run_id}/model", run_id=run.info.run_id,
                                    tags={"run_name": run.info.run_name,
                                          "decision_threshold": str(run.data.metrics.get("decision_threshold")),
                                          "val_roc_auc": f"{run.data.metrics.get('val_roc_auc', 0):.4f}",
                                          "test_roc_auc": f"{run.data.metrics.get('test_roc_auc', 0):.4f}"},
                                    description=f"{run.info.run_name}: {run.data.tags.get('mlflow.note.content', '')}")
        c.set_registered_model_alias(MODEL_NAME, role, mv.version)
        versions[role] = mv.version
        print(f"registered {MODEL_NAME} v{mv.version} <- {run.info.run_name}  alias @{role}")

    # export champion for serving
    with tempfile.TemporaryDirectory() as d:
        local = c.download_artifacts(best.info.run_id, "keras_model", d)
        src = next(Path(local).glob("*.keras"))
        MODELS_DIR.mkdir(exist_ok=True)
        shutil.copy(src, MODELS_DIR / "champion.keras")
    meta = {"model_name": MODEL_NAME, "model_version": versions["champion"], "alias": "champion",
            "run_id": best.info.run_id, "run_name": best.info.run_name,
            "threshold": best.data.metrics["decision_threshold"],
            "img_size": int(best.data.params.get("img_size", 224))}
    (MODELS_DIR / "champion.json").write_text(json.dumps(meta, indent=2))
    print("exported models/champion.keras + champion.json:", json.dumps(meta))


if __name__ == "__main__":
    main()
