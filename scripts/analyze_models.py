"""Tasks 2.5, 2.6 and bonus: final comparison figures, error analysis grid and Grad-CAM.

Uses the per-run outputs written by train.py (reports/runs/<run>/...), so the
test set is NOT re-evaluated - each model was evaluated on it exactly once.
Usage: python scripts/analyze_models.py --cnn cnn_baseline --best mnv2_finetune
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from visionguard import models as _custom  # noqa: E402,F401
from visionguard.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ROOT  # noqa: E402
from visionguard.evaluation import plot_roc  # noqa: E402
from visionguard.preprocess import load_image  # noqa: E402

RUNS = REPORTS_DIR / "runs"


def load_result(run):
    return json.loads((RUNS / run / "result.json").read_text())


def metric_table(runs: dict) -> pd.DataFrame:
    rows = []
    for label, run in runs.items():
        r = load_result(run)
        t = r["test"]
        rows.append({"model": label, "run": run, "threshold": r["threshold_selection"]["threshold"],
                     "accuracy": t["accuracy"], "precision": t["precision"], "recall": t["recall"],
                     "f1": t["f1"], "roc_auc": t["roc_auc"], "FPR (real->AI)": t["fpr_real_called_ai"],
                     "FNR (AI->real)": t["fnr_ai_called_real"], "TN": t["tn"], "FP": t["fp"], "FN": t["fn"],
                     "TP": t["tp"], "best_epoch": r["best_epoch"]})
    return pd.DataFrame(rows)


def confusion_pair(runs: dict, path):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.3 * len(runs), 3.9))
    for ax, (label, run) in zip(np.atleast_1d(axes), runs.items()):
        r = load_result(run)
        t = r["test"]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Real", "AI"])
        ax.set_yticks([0, 1], ["Real", "AI"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{label}\n(test, threshold {r['threshold_selection']['threshold']:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def threshold_curve(run, path):
    v = pd.read_csv(RUNS / run / "val_predictions.csv")
    r = load_result(run)
    ts = np.arange(0.05, 0.96, 0.01)
    neg, pos = v.label == 0, v.label == 1
    fpr = [((v.p_ai >= t) & neg).sum() / neg.sum() for t in ts]
    tpr = [((v.p_ai >= t) & pos).sum() / pos.sum() for t in ts]
    from sklearn.metrics import f1_score
    f1 = [f1_score(v.label, v.p_ai >= t, zero_division=0) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ts, f1, label="F1")
    ax.plot(ts, tpr, label="recall (AI caught)")
    ax.plot(ts, fpr, label="FPR (real called AI)")
    ax.axhline(0.15, color="grey", ls=":", lw=1, label="FPR cap 0.15")
    thr = r["threshold_selection"]["threshold"]
    ax.axvline(thr, color="green", ls="--", label=f"chosen threshold {thr:.2f}")
    ax.set_xlabel("decision threshold on P(AI)")
    ax.set_title(f"Threshold selection on VALIDATION data - {run}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def misclassified_grid(run, path, n=10):
    r = load_result(run)
    thr = r["threshold_selection"]["threshold"]
    t = pd.read_csv(RUNS / run / "test_predictions.csv")
    t["pred"] = (t.p_ai >= thr).astype(int)
    wrong = t[t.pred != t.label].copy()
    wrong["margin"] = (wrong.p_ai - thr).abs()
    wrong = wrong.sort_values("margin", ascending=False).head(n)
    wrong.to_csv(REPORTS_DIR / "misclassified.csv", index=False)
    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    from PIL import Image
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i >= len(wrong):
            continue
        w = wrong.iloc[i]
        ax.imshow(load_image(str(ROOT / w.path), 256))
        with Image.open(ROOT / w.path) as im:
            info = f"{im.width}x{im.height} {im.format}"
        truth = "AI" if w.label else "Real"
        ax.set_title(f"#{i + 1} true {truth}, P(AI)={w.p_ai:.2f}\n{info}", fontsize=9,
                     color="#b5452f" if w.label == 0 else "#2b4c7e")
    fig.suptitle(f"Misclassified test images ({run}, threshold {thr:.2f}) - most confident mistakes first")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return wrong


def gradcam_model(model: keras.Model):
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    x = model.input
    for layer in model.layers[1:]:
        if layer is base:
            break
        x = layer(x)
    conv = base(x, training=False)
    y = conv
    after = model.layers[model.layers.index(base) + 1:]
    for layer in after:
        y = layer(y, training=False) if isinstance(layer, keras.layers.Dropout) else layer(y)
    return keras.Model(model.input, [conv, y])


def gradcam(gm, img):
    x = tf.convert_to_tensor(img[None].astype(np.float32))
    with tf.GradientTape() as tape:
        conv, pred = gm(x)
        score = pred[:, 0]
    grads = tape.gradient(score, conv)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(pred[0, 0])


def gradcam_figure(run, path):
    r = load_result(run)
    thr = r["threshold_selection"]["threshold"]
    model = keras.models.load_model(MODELS_DIR / "runs" / f"{run}.keras")
    size = int(model.input_shape[1])
    gm = gradcam_model(model)
    t = pd.read_csv(RUNS / run / "test_predictions.csv")
    t["pred"] = (t.p_ai >= thr).astype(int)
    rng = np.random.default_rng(0)
    correct = pd.concat([t[(t.pred == t.label) & (t.label == c)].sample(2, random_state=1) for c in [0, 1]])
    wrong = t[t.pred != t.label]
    wrong = pd.concat([wrong[wrong.label == c].head(2) for c in [0, 1]])
    if len(wrong) < 4:
        wrong = t[t.pred != t.label].head(4)
    sel = [("correct", row) for _, row in correct.iterrows()] + [("wrong", row) for _, row in wrong.iterrows()]
    fig, axes = plt.subplots(2, 8, figsize=(22, 6.2))
    for i, (kind, row) in enumerate(sel[:8]):
        img = load_image(str(ROOT / row.path), size)
        cam, p = gradcam(gm, img)
        cam = tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0]
        top, bot = axes[0, i], axes[1, i]
        top.imshow(img)
        bot.imshow(img)
        bot.imshow(cam, cmap="jet", alpha=0.45)
        truth = "AI" if row.label else "Real"
        top.set_title(f"{kind.upper()}: true {truth}\nP(AI)={p:.2f}", fontsize=9,
                      color="#2f7a55" if kind == "correct" else "#b5452f")
        top.axis("off")
        bot.axis("off")
    _ = rng
    fig.suptitle(f"Grad-CAM ({run}): where the last conv block looks when scoring 'AI-generated'")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="cnn_baseline")
    ap.add_argument("--best", required=True)
    a = ap.parse_args()
    runs = {"Baseline CNN": a.cnn, "Transfer (best)": a.best}
    tbl = metric_table(runs)
    tbl.to_csv(REPORTS_DIR / "final_metrics.csv", index=False)
    print(tbl.round(3).to_string(index=False))
    plot_roc({lbl: (pd.read_csv(RUNS / r / "test_predictions.csv").label,
                    pd.read_csv(RUNS / r / "test_predictions.csv").p_ai) for lbl, r in runs.items()},
             FIGURES_DIR / "roc_both_models.png", "ROC curves on the test set")
    confusion_pair(runs, FIGURES_DIR / "confusion_both_models.png")
    threshold_curve(a.best, FIGURES_DIR / "threshold_selection.png")
    wrong = misclassified_grid(a.best, FIGURES_DIR / "misclassified_grid.png")
    print(wrong[["path", "label", "p_ai"]].to_string(index=False))
    gradcam_figure(a.best, FIGURES_DIR / "gradcam.png")
    print("figures written")


if __name__ == "__main__":
    main()
