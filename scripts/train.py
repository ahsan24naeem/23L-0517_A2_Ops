"""Train, evaluate and log one VisionGuard experiment to MLflow.

Examples
  # baseline CNN, manual MLflow logging
  python scripts/train.py --run-name cnn_baseline --model cnn --lr 1e-3
  # MobileNetV2: frozen head training, then last block fine-tuned at 1e-5
  python scripts/train.py --run-name mnv2_ft --model transfer --backbone mobilenetv2
  # same configuration with MLflow's TensorFlow autologging
  python scripts/train.py --run-name mnv2_ft_autolog --model transfer --logging autolog

Tracking destination comes from MLFLOW_TRACKING_URI (+ USERNAME/PASSWORD) in the
environment or .env; local SQLite store otherwise.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from visionguard.config import MODELS_DIR, REPORTS_DIR, ROOT  # noqa: E402
from visionguard.data import make_dataset, read_split  # noqa: E402
from visionguard.evaluation import (  # noqa: E402
    choose_threshold, compute_metrics, plot_confusion, plot_history, plot_roc,
)
from visionguard.models import (  # noqa: E402
    build_baseline_cnn, build_transfer_model, trainable_layer_names, unfreeze_last_layers,
)
from visionguard.tracking import MLflowEpochLogger, pip_freeze, setup_mlflow, standard_tags  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--experiment", default="VisionGuard")
    ap.add_argument("--model", choices=["cnn", "transfer"], default="transfer")
    ap.add_argument("--backbone", choices=["mobilenetv2", "resnet50"], default="mobilenetv2")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (CNN / frozen-head stage)")
    ap.add_argument("--ft-lr", type=float, default=1e-5, help="learning rate for the fine-tuning stage")
    ap.add_argument("--epochs", type=int, default=15, help="max epochs (CNN / frozen-head stage)")
    ap.add_argument("--ft-epochs", type=int, default=10, help="max epochs for fine-tuning (0 = skip)")
    ap.add_argument("--unfreeze", default="last_block", help="'last_block' or number of backbone layers")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging", choices=["manual", "autolog"], default="manual")
    ap.add_argument("--description", default="")
    ap.add_argument("--save-as", default="", help="also copy the trained model to models/<name>.keras")
    return ap.parse_args()


def callbacks(ckpt_path: Path, patience: int, min_lr: float):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=min_lr, verbose=1),
    ]


def compile_model(model, lr):
    # Binary cross-entropy matches a single sigmoid output; Adam adapts per-parameter
    # step sizes and works well with little tuning; AUC is threshold-free.
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )


def predict(model, ds):
    return model.predict(ds, verbose=0).ravel()


def main():
    a = parse_args()
    keras.utils.set_random_seed(a.seed)  # python, numpy and TF seeds
    out_dir = REPORTS_DIR / "runs" / a.run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    ckpt_dir = MODELS_DIR / "runs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{a.run_name}.keras"

    train_df, val_df, test_df = read_split("train"), read_split("val"), read_split("test")
    augment = not a.no_augment
    train_ds = make_dataset(train_df, a.img_size, a.batch_size, True, augment, a.seed, cache_name="train")
    val_ds = make_dataset(val_df, a.img_size, a.batch_size, False, cache_name="val")
    test_ds = make_dataset(test_df, a.img_size, a.batch_size, False, cache_name="test")

    uri = setup_mlflow(a.experiment)
    print("MLflow tracking URI:", uri.split("@")[-1])
    if a.logging == "autolog":
        # Autologging captures fit() params, per-epoch metrics, model summary and model.
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False, checkpoint=False)

    params = {
        "model_type": a.model, "backbone": a.backbone if a.model == "transfer" else "none",
        "img_size": a.img_size, "batch_size": a.batch_size, "lr": a.lr,
        "ft_lr": a.ft_lr if a.model == "transfer" else None, "max_epochs": a.epochs,
        "max_ft_epochs": a.ft_epochs if a.model == "transfer" else 0,
        "unfreeze": a.unfreeze if a.model == "transfer" else None, "dropout": a.dropout,
        "augment": augment, "seed": a.seed, "optimizer": "adam", "loss": "binary_crossentropy",
        "early_stopping_patience": a.patience, "n_train": len(train_df), "n_val": len(val_df),
        "n_test": len(test_df),
    }
    tags = standard_tags(a.seed, {"logging_mode": a.logging, "model_family": params["backbone"]})

    t0 = time.time()
    with mlflow.start_run(run_name=a.run_name, description=a.description or None) as run:
        mlflow.set_tags(tags)
        if a.logging == "manual":
            mlflow.log_params(params)
        else:
            # autolog logs fit() arguments itself; keep only the data/model choices it cannot see
            mlflow.log_params({k: params[k] for k in ["model_type", "backbone", "img_size", "augment",
                                                      "unfreeze", "dropout", "seed", "ft_lr"]})

        history: dict[str, list] = {}
        boundaries = []
        trainable_report = {}

        def run_fit(model, lr, epochs, offset, stage):
            compile_model(model, lr)
            cbs = callbacks(ckpt_path, a.patience, min_lr=lr / 100)
            if a.logging == "manual":
                cbs.append(MLflowEpochLogger(offset=offset, stage=stage))
            h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
            for k, v in h.history.items():
                history.setdefault(k, []).extend([float(x) for x in v])
            return len(h.history["loss"])

        if a.model == "cnn":
            model = build_baseline_cnn(a.img_size, a.dropout)
            trainable_report["stage1"] = trainable_layer_names(model)
            run_fit(model, a.lr, a.epochs, 0, "cnn")
        else:
            model, base = build_transfer_model(a.backbone, a.img_size, a.dropout)
            trainable_report["stage1_frozen_backbone"] = trainable_layer_names(model)
            n1 = run_fit(model, a.lr, a.epochs, 0, "head")
            if a.ft_epochs > 0:
                boundaries.append(n1)
                n_unfreeze = None if a.unfreeze == "last_block" else int(a.unfreeze)
                unfreeze_last_layers(base, a.backbone, n_unfreeze)
                trainable_report["stage2_fine_tune"] = trainable_layer_names(model)
                run_fit(model, a.ft_lr, a.ft_epochs, n1, "fine_tune")

        # restore_best_weights already gives the best weights of the last stage;
        # the checkpoint holds the overall best val_loss model across both stages.
        model = keras.models.load_model(ckpt_path)
        best_epoch = int(np.argmin(history["val_loss"]) + 1)
        train_seconds = time.time() - t0

        # ---- threshold on VALIDATION, then evaluate on TEST exactly once ----
        p_val = predict(model, val_ds)
        thr = choose_threshold(val_df.label.values, p_val)
        val_m = compute_metrics(val_df.label.values, p_val, thr["threshold"])
        p_test = predict(model, test_ds)
        test_m = compute_metrics(test_df.label.values, p_test, thr["threshold"])
        test_m_05 = compute_metrics(test_df.label.values, p_test, 0.5)

        metrics = {f"val_{k}": v for k, v in val_m.items()}
        metrics.update({f"test_{k}": v for k, v in test_m.items()})
        metrics.update({"test_accuracy_at_0.5": test_m_05["accuracy"], "test_f1_at_0.5": test_m_05["f1"],
                        "decision_threshold": thr["threshold"], "best_epoch": best_epoch,
                        "best_val_loss": float(np.min(history["val_loss"])),
                        "epochs_trained": len(history["loss"]), "train_seconds": round(train_seconds, 1),
                        "total_params": model.count_params(),
                        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights))})
        mlflow.log_metrics(metrics)

        # ---- artifacts ----
        title = f"{a.run_name}: best epoch {best_epoch} (val_loss {metrics['best_val_loss']:.3f})"
        plot_history(history, boundaries, best_epoch, title, out_dir / "training_curves.png")
        plot_confusion(test_m, f"{a.run_name} test (thr={thr['threshold']:.2f})", out_dir / "confusion_matrix.png")
        plot_roc({a.run_name: (test_df.label.values, p_test)}, out_dir / "roc_curve.png")
        pd.DataFrame(history).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(out_dir / "history.csv", index=False)
        pd.DataFrame({"path": test_df.path, "label": test_df.label, "p_ai": p_test}).to_csv(
            out_dir / "test_predictions.csv", index=False)
        pd.DataFrame({"path": val_df.path, "label": val_df.label, "p_ai": p_val}).to_csv(
            out_dir / "val_predictions.csv", index=False)
        (out_dir / "trainable_layers.json").write_text(json.dumps(trainable_report, indent=2))
        summary_lines = []
        model.summary(print_fn=lambda s, **k: summary_lines.append(s), expand_nested=False)
        (out_dir / "model_summary.txt").write_text("\n".join(summary_lines))
        result = {"run_id": run.info.run_id, "run_name": a.run_name, "params": params, "tags": tags,
                  "threshold_selection": thr, "val": val_m, "test": test_m, "test_at_0.5": test_m_05,
                  "best_epoch": best_epoch, "boundaries": boundaries, "train_seconds": train_seconds}
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        meta = {"model_name": f"visionguard-{a.run_name}", "threshold": thr["threshold"],
                "img_size": a.img_size, "run_id": run.info.run_id, "model_version": "local"}
        ckpt_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        pip_freeze(out_dir / "requirements-freeze.txt")

        mlflow.log_artifacts(str(out_dir), artifact_path="reports")
        for name in ["train", "val", "test"]:
            mlflow.log_artifact(str(ROOT / "splits" / f"{name}.csv"), artifact_path="splits")
        mlflow.log_artifact(str(ckpt_path), artifact_path="keras_model")
        mlflow.log_artifact(str(ckpt_path.with_suffix(".json")), artifact_path="keras_model")
        if a.logging == "manual":
            mlflow.tensorflow.log_model(model, artifact_path="model")

        if a.save_as:
            dst = MODELS_DIR / f"{a.save_as}.keras"
            shutil.copy(ckpt_path, dst)
            shutil.copy(ckpt_path.with_suffix(".json"), dst.with_suffix(".json"))

        print(json.dumps({"run_id": run.info.run_id, "best_epoch": best_epoch, "threshold": thr,
                          "val": {k: round(v, 4) for k, v in val_m.items() if isinstance(v, float)},
                          "test": {k: round(v, 4) for k, v in test_m.items() if isinstance(v, float)},
                          "minutes": round(train_seconds / 60, 1)}, indent=2))


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
