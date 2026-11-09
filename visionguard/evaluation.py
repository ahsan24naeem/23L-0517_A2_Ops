"""Metrics, threshold selection and evaluation figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Product decision: calling a REAL photo "AI-generated" (false positive) is the
# worse error - it accuses an honest photographer / journalist of faking. So the
# threshold is chosen on VALIDATION data to maximise F1 subject to a cap on the
# false-positive rate (real -> AI).
MAX_FPR = 0.15


def choose_threshold(y_true: np.ndarray, p: np.ndarray, max_fpr: float = MAX_FPR) -> dict:
    y_true = np.asarray(y_true).astype(int)
    best = None
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (p >= t).astype(int)
        neg = y_true == 0
        fpr = float((pred[neg] == 1).mean()) if neg.any() else 0.0
        f1 = f1_score(y_true, pred, zero_division=0)
        cand = {"threshold": float(t), "val_f1": float(f1), "val_fpr": fpr}
        ok = fpr <= max_fpr
        key = (ok, f1, -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, cand)
    out = best[1]
    out["constraint_met"] = bool(best[0][0])
    return out


def compute_metrics(y_true, p, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "fpr_real_called_ai": float(fp / max(fp + tn, 1)),
        "fnr_ai_called_real": float(fn / max(fn + tp, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion(m: dict, title: str, path) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Real", "AI"])
    ax.set_yticks([0, 1], ["Real", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(curves: dict, path, title="ROC curve (test set)") -> None:
    """curves: {name: (y_true, p)}"""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False-positive rate (real called AI)")
    ax.set_ylabel("True-positive rate (AI caught)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history: dict, stage_boundaries: list[int], best_epoch: int, title: str, path) -> None:
    """history: dict of lists over the global epoch index (1-based)."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(axes[0], "loss", "Binary cross-entropy"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, history[key], "o-", ms=3, label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], "o-", ms=3, label=f"val {key}")
        ax.axvline(best_epoch, color="green", ls="--", lw=1.2, label=f"best epoch {best_epoch}")
        for b in stage_boundaries:
            ax.axvline(b + 0.5, color="grey", ls=":", lw=1, label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
