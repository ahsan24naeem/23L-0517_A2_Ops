"""Task 2.7 toy part: Keras Tuner Hyperband on Fashion-MNIST.

Searches the learning rate and number of hidden units of a tiny dense network.
Usage: python labs/keras_tuner_toy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import keras
import keras_tuner as kt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_data import load_fashion_mnist  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tuner"


def build(hp: kt.HyperParameters) -> keras.Model:
    units = hp.Int("units", 32, 256, step=32)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Rescaling(1 / 255.0),
        keras.layers.Flatten(),
        keras.layers.Dense(units, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    keras.utils.set_random_seed(42)
    (x, y), (x_test, y_test) = load_fashion_mnist()
    x_val, y_val = x[-10000:], y[-10000:]
    x, y = x[:-10000], y[:-10000]
    tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=6, factor=3, seed=42,
                         directory=str(ROOT / "tuner"), project_name="toy_fashion_mnist", overwrite=True)
    tuner.search(x, y, validation_data=(x_val, y_val), batch_size=256, verbose=0)
    rows = []
    for t in tuner.oracle.trials.values():
        if t.score is None:
            continue
        rows.append({"trial": t.trial_id, "units": t.hyperparameters.get("units"),
                     "learning_rate": round(t.hyperparameters.get("learning_rate"), 5),
                     "epochs": t.hyperparameters.get("tuner/epochs"), "val_accuracy": round(t.score, 4)})
    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best = tuner.get_best_models(1)[0]
    test_acc = float(best.evaluate(x_test, y_test, verbose=0)[1])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "toy_trials.csv", index=False)
    res = {"best_units": best_hp.get("units"), "best_learning_rate": best_hp.get("learning_rate"),
           "best_val_accuracy": float(df.val_accuracy.max()), "test_accuracy": test_acc, "n_trials": len(df)}
    (OUT / "toy_best.json").write_text(json.dumps(res, indent=2))
    print(df.head(15).to_string(index=False))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
