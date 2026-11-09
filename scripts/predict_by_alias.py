"""Task 3.6 - load the registered model BY ALIAS and classify one image.

Usage: python scripts/predict_by_alias.py path/to/image.jpg [--alias champion]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from visionguard.preprocess import load_image  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--name", default="visionguard")
    a = ap.parse_args()
    uri = setup_mlflow("VisionGuard")
    mv = MlflowClient().get_model_version_by_alias(a.name, a.alias)
    print(f"tracking URI : {uri.split('@')[-1]}")
    print(f"model        : models:/{a.name}@{a.alias} -> version {mv.version} (run {mv.run_id[:8]}, "
          f"{mv.tags.get('run_name')})")
    t0 = time.time()
    model = mlflow.tensorflow.load_model(f"models:/{a.name}@{a.alias}")
    print(f"loaded in    : {time.time() - t0:.1f}s")
    thr = float(mv.tags.get("decision_threshold", 0.5))
    size = int(model.input_shape[1])
    x = load_image(a.image, size).astype(np.float32)[None]
    p_ai = float(model.predict(x, verbose=0)[0, 0])
    label = "AI-generated" if p_ai >= thr else "Real"
    conf = p_ai if p_ai >= thr else 1 - p_ai
    print(f"image        : {a.image}")
    print(f"P(AI)={p_ai:.4f}  threshold={thr:.2f}  ->  {label}  (confidence {conf:.1%})")


if __name__ == "__main__":
    main()
