# Conclusion and lessons

VisionGuard works end to end: a de-duplicated, DVC-versioned dataset; a reproducible `tf.data` pipeline whose preprocessing is shared with serving; ten tracked experiments; a registered champion (`visionguard@champion`, MobileNetV2 fine-tuned with Keras-Tuner settings) with **test ROC-AUC 0.871, precision 0.885 and a 10.6 % false-positive rate** at a threshold chosen on validation data; a Flask app with validation, history and health checks; a two-service Compose stack with a private PostgreSQL; CI; and the model API on Oracle Cloud called from the local app.

The most important lessons were not about architecture. The **data** limits the model: the "Real" class contains paintings, graphics and at least one generated image, AI images are mostly fantasy art, and the error analysis and Grad-CAM show the model learned *style and texture* rather than generator artefacts — it misses photorealistic fakes and flat graphics. With 146 validation and 147 test images, differences of ±0.02 AUC between runs are noise (the identical manual and autolog runs differed by 0.013), so I selected on validation only and report test numbers once. On the engineering side, pinning versions, keeping preprocessing in one function, loading the model once, and reading every setting from the environment prevented most of the problems that otherwise appear between "works on my machine" and a deployed service.

**What I would improve with more time.**

- **More and cleaner data.** 975 images with noisy "Real" labels (paintings, cartoons, a generated image) limit everything downstream. Collecting real photos and outputs of current generators (and labelling the generator) would matter more than any architecture change.
- **Remove shortcuts.** Re-encode every image to the same JPEG quality before training so the format imbalance (PNG mostly AI) cannot be learned, then measure how much accuracy that costs.
- **Robust evaluation.** k-fold cross-validation instead of one 146-image validation set, confidence intervals on test metrics, and a separate test set from generators not seen in training.
- **Serving.** TLS in front of the cloud API, a model-version column already exists so A/B comparisons of champion vs challenger could be added, and the Docker image could be shrunk further with TFLite / LiteRT for inference only.

# Appendix — Repository map and commands

| Path | Purpose |
|---|---|
| `scripts/audit_dataset.py`, `make_splits.py`, `eda.py` | Task 1 |
| `visionguard/data.py`, `models.py`, `evaluation.py`, `scripts/train.py`, `run_experiments.sh` | Task 2/3 training |
| `scripts/tune.py`, `labs/keras_tuner_toy.py` | Keras Tuner |
| `visionguard/tracking.py`, `scripts/compare_runs.py`, `register_model.py`, `predict_by_alias.py`, `sync_to_dagshub.py` | MLflow / registry |
| `visionguard/inference.py`, `preprocess.py`, `validation.py` | shared serving code |
| `app/` | Flask web app, DB layer, templates, CSS |
| `api/` | cloud model API |
| `docker/Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example` | Task 6 |
| `deploy/oracle/`, `deploy/postman/` | Task 7 |
| `.github/workflows/ci.yml`, `tests/`, `labs/ci_toy/` | CI |
| `labs/dvc_toy/`, `labs/docker_warmup/`, `labs/mlflow_toy.py` | tool labs |
| `docs/RUNBOOK.md` | exact commands for every step on my machine |
