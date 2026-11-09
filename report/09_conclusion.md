# Conclusion and lessons

CONCLUSION_SUMMARY

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
