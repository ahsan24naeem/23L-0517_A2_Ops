"""Task 2.7 apply part: Keras Tuner random search for VisionGuard (MobileNetV2).

Searches learning rate, dropout and number of unfrozen backbone layers with
>= 8 trials. Each trial trains the head + the unfrozen layers for a few epochs and
is scored on validation AUC. The trial table is also logged to MLflow.
Usage: python scripts/tune.py --trials 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import keras_tuner as kt  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import REPORTS_DIR, ROOT  # noqa: E402
from visionguard.data import make_dataset, read_split  # noqa: E402
from visionguard.models import build_transfer_model, unfreeze_last_layers  # noqa: E402
from visionguard.tracking import setup_mlflow, standard_tags  # noqa: E402


def make_builder(img_size: int):
    def build(hp: kt.HyperParameters) -> keras.Model:
        lr = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
        model, base = build_transfer_model("mobilenetv2", img_size, dropout)
        if n_unfrozen > 0:
            unfreeze_last_layers(base, "mobilenetv2", n_unfrozen)
        model.compile(keras.optimizers.Adam(lr), "binary_crossentropy",
                      metrics=["accuracy", keras.metrics.AUC(name="auc")])
        return model
    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    keras.utils.set_random_seed(42)
    tr = make_dataset(read_split("train"), a.img_size, 32, True, True, cache_name="train")
    va = make_dataset(read_split("val"), a.img_size, 32, False, cache_name="val")

    tuner = kt.RandomSearch(make_builder(a.img_size), objective=kt.Objective("val_auc", "max"),
                            max_trials=a.trials, seed=42, directory=str(ROOT / "tuner"),
                            project_name="visionguard_mnv2", overwrite=True)
    tuner.search(tr, validation_data=va, epochs=a.epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2, restore_best_weights=True)])

    rows = []
    for t in tuner.oracle.trials.values():
        hp = t.hyperparameters.values
        rows.append({"trial": t.trial_id, "learning_rate": hp["learning_rate"], "dropout": hp["dropout"],
                     "unfrozen_layers": hp["unfrozen_layers"], "val_auc": t.score})
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    out = REPORTS_DIR / "tuner"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "visionguard_trials.csv", index=False)

    # Which hyper-parameter mattered most? Spearman correlation with the score
    imp = {c: float(df[c].astype(float).corr(df.val_auc, method="spearman")) for c in
           ["learning_rate", "dropout", "unfrozen_layers"]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, c in zip(axes, ["learning_rate", "dropout", "unfrozen_layers"]):
        ax.scatter(df[c], df.val_auc, s=40)
        if c == "learning_rate":
            ax.set_xscale("log")
        ax.set_xlabel(f"{c} (Spearman rho={imp[c]:+.2f})")
        ax.set_ylabel("val AUC")
    fig.suptitle("Keras Tuner random search - VisionGuard MobileNetV2")
    fig.tight_layout()
    fig.savefig(out / "visionguard_tuner.png", dpi=130)
    best = df.iloc[0].to_dict()
    res = {"best": best, "spearman_with_val_auc": imp, "n_trials": len(df)}
    (out / "visionguard_best.json").write_text(json.dumps(res, indent=2, default=float))
    print(df.to_string(index=False))
    print(json.dumps(res, indent=2, default=float))

    setup_mlflow("VisionGuard-tuning")
    with mlflow.start_run(run_name="keras_tuner_random_search"):
        mlflow.set_tags(standard_tags(42, {"logging_mode": "manual", "kind": "hyperparameter_search"}))
        mlflow.log_params({"tuner": "RandomSearch", "max_trials": a.trials, "epochs_per_trial": a.epochs,
                           "objective": "val_auc", **{f"best_{k}": v for k, v in best.items() if k != "val_auc"}})
        mlflow.log_metric("best_val_auc", best["val_auc"])
        for i, r in enumerate(df.sort_values("trial").itertuples()):
            mlflow.log_metric("trial_val_auc", r.val_auc, step=i)
        mlflow.log_artifacts(str(out), artifact_path="tuner")


if __name__ == "__main__":
    main()
