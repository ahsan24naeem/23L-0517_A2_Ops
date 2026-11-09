"""Task 3.5 - compare all VisionGuard runs in one table and chart (from MLflow)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402

from visionguard.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402

COLS = {
    "tags.mlflow.runName": "run", "params.backbone": "backbone", "params.img_size": "img",
    "params.lr": "lr", "params.augment": "aug", "params.unfreeze": "unfreeze",
    "tags.logging_mode": "logging", "metrics.best_epoch": "best_ep", "metrics.val_roc_auc": "val_auc",
    "metrics.val_f1": "val_f1", "metrics.decision_threshold": "thr", "metrics.test_accuracy": "test_acc",
    "metrics.test_precision": "test_prec", "metrics.test_recall": "test_rec", "metrics.test_f1": "test_f1",
    "metrics.test_roc_auc": "test_auc", "metrics.test_fpr_real_called_ai": "test_fpr",
    "metrics.train_seconds": "train_s", "metrics.trainable_params": "trainable",
}


def main():
    setup_mlflow("VisionGuard")
    df = mlflow.search_runs(experiment_names=["VisionGuard"], filter_string="attributes.status = 'FINISHED'")
    df = df[[c for c in COLS if c in df.columns]].rename(columns=COLS)
    df = df.sort_values("val_auc", ascending=False)
    for c in ["val_auc", "val_f1", "test_acc", "test_prec", "test_rec", "test_f1", "test_auc", "test_fpr"]:
        df[c] = df[c].astype(float).round(3)
    df["train_min"] = (df.pop("train_s").astype(float) / 60).round(1)
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORTS_DIR / "run_comparison.csv", index=False)
    (REPORTS_DIR / "run_comparison.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))

    d = df.sort_values("val_auc")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 1.5))
    ax.barh(y - 0.2, d.val_auc, 0.4, label="validation ROC-AUC (selection)", color="#2b4c7e")
    ax.barh(y + 0.2, d.test_f1, 0.4, label="test F1 (reported)", color="#c8553d")
    for yi, v in zip(y, d.val_auc):
        ax.text(v + 0.005, yi - 0.2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, d.run)
    ax.set_xlim(0.4, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("VisionGuard experiment comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "run_comparison.png", dpi=130)


if __name__ == "__main__":
    main()
