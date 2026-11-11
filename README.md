# VisionGuard — AI-generated vs real image classifier (23L-0517_A2_Ops)

[![CI](https://github.com/ahsan24naeem/23L-0517_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/ahsan24naeem/23L-0517_A2_Ops/actions/workflows/ci.yml)

Ahsan Naeem · 23L-0517 · MLOps Assignment 2

VisionGuard tells whether an uploaded image is AI-generated or real. A TensorFlow/Keras model
(MobileNetV2 fine-tuned, compared against a CNN trained from scratch) is tracked with MLflow on
DagsHub, served by a Flask web app that logs every request to PostgreSQL, packaged with Docker
Compose, and its inference API is deployed on Oracle Cloud.

| | |
|---|---|
| GitHub | https://github.com/ahsan24naeem/23L-0517_A2_Ops |
| DagsHub (code, data, models, MLflow) | https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops |
| Model API (Oracle Cloud) | http://141.148.203.241:8000 |

## Repository layout

```
visionguard/      shared library: preprocessing, tf.data pipeline, models, inference, metrics, MLflow helpers
scripts/          audit, EDA, splits, train, tune, compare, analyse, register, predict-by-alias, DagsHub sync
app/              Flask web app (upload, result, history, health) + templates/CSS
api/              Flask model API deployed to the cloud (X-API-Key protected)
docker/           Dockerfile (slim, non-root, health check) + naive Dockerfile for the size comparison
deploy/oracle/    compose file + VM setup script for Oracle Cloud;  deploy/postman/ Postman collection
labs/             tool labs: DVC toy, MLflow toy, Keras Tuner toy, Docker warm-up, CI toy
tests/            pytest suite (runs in CI without TensorFlow)
data/raw.dvc      DVC pointer to the dataset (data itself lives in DagsHub storage)
models/           champion.keras (DVC-tracked) + champion.json (threshold, version)
splits/           train/val/test lists (70/15/15, stratified, de-duplicated)
reports/          audit tables, figures, run outputs, logs
```

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in the values
dvc pull                          # dataset + champion model from DagsHub
```

Reproduce the pipeline:

```bash
python scripts/audit_dataset.py && python scripts/make_splits.py && python scripts/eda.py
bash scripts/run_experiments.sh all && bash scripts/run_experiments.sh tuned   # 10 MLflow runs
python scripts/tune.py --trials 10            # Keras Tuner
python scripts/compare_runs.py
python scripts/register_model.py              # versions + @champion alias, exports models/champion.keras
python scripts/predict_by_alias.py some.jpg
```

Run the web app:

```bash
python -m app                                  # Flask server, debug off, http://127.0.0.1:5000
docker compose up -d --build                   # web + PostgreSQL, http://localhost:8000
INFERENCE_MODE=remote python -m app            # same UI, inference by the Oracle Cloud API
```

## Configuration

Everything is read from environment variables (see `.env.example`): `MODEL_PATH`, `THRESHOLD`,
`DATABASE_URL`, `INFERENCE_MODE`, `MODEL_API_URL`, `MODEL_API_KEY`, `MAX_UPLOAD_MB`, `SECRET_KEY`,
`MLFLOW_TRACKING_URI/USERNAME/PASSWORD`. `.env` is git-ignored and docker-ignored.
