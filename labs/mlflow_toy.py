"""Task 3.1 / 3.2 tool lab: tiny Keras model on Fashion-MNIST, two learning rates.

Local:    python labs/mlflow_toy.py                 then  mlflow ui --backend-store-uri sqlite:///mlflow_toy.db
DagsHub:  put MLFLOW_TRACKING_URI / USERNAME / PASSWORD in .env, then
          python labs/mlflow_toy.py --remote
Logs parameters, per-epoch metrics, one plot, tags and the model for each run.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_data import load_fashion_mnist  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)


def one_run(lr: float, data, epochs: int):
    (x, y), (xv, yv) = data
    keras.utils.set_random_seed(42)
    model = keras.Sequential([keras.Input((28, 28)), keras.layers.Rescaling(1 / 255.0), keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="relu"), keras.layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    with mlflow.start_run(run_name=f"toy_lr_{lr:g}"):
        mlflow.set_tags({"lab": "mlflow-toy", "dataset": "fashion-mnist", "author": "Ahsan Naeem (23L-0517)"})
        mlflow.log_params({"learning_rate": lr, "epochs": epochs, "hidden_units": 64, "batch_size": 128,
                           "optimizer": "adam"})
        h = model.fit(x, y, validation_data=(xv, yv), epochs=epochs, batch_size=128, verbose=2,
                      callbacks=[EpochLogger()])
        mlflow.log_metric("final_val_accuracy", h.history["val_accuracy"][-1])
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(h.history["accuracy"], label="train")
        ax.plot(h.history["val_accuracy"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Fashion-MNIST, lr={lr:g}")
        ax.legend()
        fig.tight_layout()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "accuracy_curve.png")
            fig.savefig(p, dpi=120)
            mlflow.log_artifact(p, artifact_path="plots")
        plt.close(fig)
        mlflow.tensorflow.log_model(model, artifact_path="model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="use MLFLOW_TRACKING_URI from the environment/.env")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    if a.remote:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            sys.exit("MLFLOW_TRACKING_URI is not set (see .env.example)")
    else:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow_toy.db'}")
    mlflow.set_experiment("toy-fashion-mnist")
    (x, y), _ = load_fashion_mnist()
    data = ((x[:20000], y[:20000]), (x[-5000:], y[-5000:]))
    for lr in [1e-3, 1e-2]:
        one_run(lr, data, a.epochs)
    print("done - tracking URI:", mlflow.get_tracking_uri().split("@")[-1])


if __name__ == "__main__":
    main()
