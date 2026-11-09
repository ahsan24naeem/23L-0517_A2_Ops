# Task 4 — Flask Web Application

## 4.1 Inference module

`visionguard/inference.py` contains a `Predictor` class. Its constructor runs **once**, when the app starts: it loads `champion.keras`, reads the companion `champion.json` (decision threshold, model name, registry version), records the input size from the model itself, and runs one warm-up prediction so the first user does not pay TensorFlow's graph-building cost. `predict(image)` then (1) decodes and resizes the upload with `visionguard.preprocess.load_image` — **the same function the training `tf.data` pipeline calls** — (2) runs the model, (3) compares P(AI) with the threshold and returns a label, the confidence in that label (P(AI) or 1 − P(AI)), the raw probability, the threshold, model name/version and latency.

**Why loading per request is unacceptable.** Loading the model means reading ~10 MB from disk, deserialising the Keras graph and initialising TensorFlow — about 2–4 seconds and a few hundred MB of memory each time. Per request, every user would wait seconds, memory would spike with concurrent users (each request holding its own copy) and the server would fall over under modest load. Loading once and sharing the in-memory model (protected by a lock for thread safety) makes each prediction ~100 ms.

**Why training and serving preprocessing must match.** The model only learned the mapping from pixels prepared one particular way (Pillow decode → RGB → bilinear squash to 224×224 → raw 0–255, with MobileNetV2's `[-1, 1]` scaling inside the model). If serving used, say, TensorFlow's decoder with a different resize filter, cropping instead of squashing, BGR order or 0–1 scaling, the input distribution would shift and accuracy would silently drop — the classic *training/serving skew*. Putting the preprocessing in one shared function, and the model-specific scaling inside the saved model, removes that risk by construction.

```text
$ python -c "from visionguard.inference import Predictor; p = Predictor('models/champion.keras'); ..."
loaded visionguard v2 (threshold 0.71, input 224px) in 6.0s
RealArt   1032249.jpg                            -> Real         conf=0.975 p_ai=0.025 119.1 ms
AiArtData 1EOlo-3uyfHVlEE649tq62g.png            -> AI-generated conf=0.996 p_ai=0.996 125.1 ms
```

> **Screenshot 4.1** — Console output of the inference module for one known real image and one known AI image.

## 4.2 Routes and validation

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Upload page |
| `/predict` | POST | Validates, predicts, writes a DB row, then redirects (303) to `/result/<id>` |
| `/result/<id>` | GET | Verdict, confidence bar, probability, threshold, model + version, latency, caveat |
| `/history` | GET | Paginated table of all requests from the database + summary counts |
| `/api/history` | GET | Same history as JSON |
| `/health` | GET | JSON: model loaded, model name/version/threshold, DB reachable, debug flag; 200 or 503 |

Validation (`visionguard/validation.py`, shared with the cloud API):

- **Wrong file type → 415.** The extension must be one of jpg/jpeg/png/webp/gif/bmp.
- **Too large → 413.** `MAX_CONTENT_LENGTH` (from `MAX_UPLOAD_MB`, default 5 MB) makes Flask reject the body before reading it all, and the validator double-checks the size.
- **Looks like an image but cannot be decoded → 422.** Pillow must both `verify()` and fully `load()` the bytes — a text file renamed `photo.jpg` or a truncated JPEG is rejected.
- **File-name sanitising.** `werkzeug.utils.secure_filename` turns `../../etc/My Photo.png` into `etc_My_Photo.png`; only this cleaned name is stored or displayed.
- **The upload never touches disk.** Werkzeug normally spools uploads larger than 500 KB to a temporary file; a custom `Request` class overrides `_get_file_stream` to return an in-memory `BytesIO`. The bytes are hashed (SHA-256), predicted on and discarded — only metadata is stored.
- **Friendly error pages.** Every rejection renders `error.html` with a plain-language title and message and the right status code; rejected uploads are not written to the history.

The pytest suite (`tests/test_app.py`, `tests/test_validation.py`) checks each of these cases automatically.

> **Screenshot 4.2** — Error page "Unsupported file type" (uploading a `.txt`).
> **Screenshot 4.3** — Error page "File too large" (uploading a > 5 MB image).
> **Screenshot 4.4** — Error page "Not a valid image" (a text file renamed `.jpg`).

## 4.3 HTML and CSS

Jinja templates (`app/templates/`) share a `base.html` layout with a top menu — **Upload · History · Health** — and a footer that always shows which model version and inference mode are active. The upload page previews the chosen image in the browser before sending it. The result page shows the verdict in large type (green for Real, red for AI-generated), a confidence bar, the probability and threshold, the model and version, the latency and an explicit caveat that the verdict is a hint, not proof. The History page reads from the database and shows totals plus a table of every request (time, file, verdict, confidence, P(AI), model/version, mode, latency); each row links back to its result page. One plain CSS file, no framework; it adapts to dark mode and phone widths.

> **Screenshot 4.5** — Upload page.  **Screenshot 4.6** — Result page for an AI image.  **Screenshot 4.7** — Result page for a real photo.  **Screenshot 4.8** — History page.

## 4.4 Configuration and production server

Every setting is read from environment variables in `app/config.py`: `MODEL_PATH`, `THRESHOLD` (empty = use the validated threshold stored in `champion.json`), `DATABASE_URL`, `INFERENCE_MODE`, `MODEL_API_URL`, `MODEL_API_KEY`, `MAX_UPLOAD_MB`, `SECRET_KEY`, `FLASK_DEBUG`. Locally they come from `.env`; in Compose from `env_file`; in the cloud from the VM's `.env`. The same code therefore runs everywhere and no secret or machine-specific path is in Git.

`python -m app` starts the built-in Flask server with `debug=False` and the reloader off; inside Docker the app runs under **Gunicorn** (`gunicorn app.wsgi:app`, 1 worker × 4 threads). Debug mode stays off because: the Werkzeug debugger gives anyone who can trigger an error an interactive Python console (remote code execution); detailed tracebacks leak paths, settings and code; and the auto-reloader starts the process twice, loading the model twice. The built-in server is single-process and not hardened, which is why a production WSGI server is used in the container.

Example `/health` response (Gunicorn + PostgreSQL 16, champion model):

```json
{
    "database": {
        "backend": "postgresql",
        "ok": true
    },
    "debug": false,
    "inference_mode": "local",
    "model": {
        "api_url": null,
        "loaded": true,
        "name": "visionguard",
        "path": "models/champion.keras",
        "threshold": 0.71,
        "version": "2"
    },
    "status": "ok"
}
```

> **Screenshot 4.9** — Browser at `http://localhost:8000/health` showing `"status": "ok"`, `"debug": false`, model version and `"database": {"backend": "postgresql", "ok": true}`.

# Task 5 — Database Trail

**Choice: PostgreSQL in Docker Compose (SQLite for local development and tests).** Both are reached through SQLAlchemy Core (`app/db.py`), so only `DATABASE_URL` changes. PostgreSQL was chosen for the deployed stack because it is a real server that several app processes can write to concurrently and it lives in its own container with a persistent named volume; SQLite is a single file, perfect for tests.

The table `predictions` is created automatically at start-up. **One row is inserted in the `/predict` view after every successful prediction**, before redirecting to the result page; the result and history pages read back from it.

| Column | Answers |
|---|---|
| `id`, `created_at` (UTC, timezone-aware) | **when** |
| `label`, `is_ai`, `p_ai`, `threshold` | **what was decided** (and the raw score behind it) |
| `confidence` | **how confidently** |
| `model_name`, `model_version`, `inference_mode` | **by which model** (registry version, local or remote API) |
| `filename` (sanitised), `file_sha256`, `file_size_bytes`, `image_width`, `image_height` | which input — without storing the image itself |
| `latency_ms` | how long it took |

Storing the SHA-256 instead of the image lets me detect repeated uploads and audit decisions without keeping user content. The **History** menu item renders this table (25 rows per page, newest first) with totals; `/api/history` returns the same rows as JSON. Because PostgreSQL publishes no port, the trail can only be queried from inside the Compose network, e.g.:

```bash
docker compose exec db psql -U visionguard -d visionguard \
  -c "select id, created_at, label, round(confidence::numeric,3), model_version, inference_mode from predictions order by id desc limit 5;"
```

> **Screenshot 5.1** — History page after several uploads.
> **Screenshot 5.2** — The same rows queried with `psql` inside the `db` container.

# Task 6 — Containerization with Docker and Docker Compose

## 6.1 Docker warm-up

`labs/docker_warmup/warmup.sh` practises every basic action with the ready-made `nginx:alpine` image: `docker pull`; `docker run -d --name vg-warmup -p 8080:80 -v "$PWD/site":/usr/share/nginx/html:ro nginx:alpine`; `curl localhost:8080` (the page comes from my mounted folder, and an edit on the host shows up immediately); `docker logs vg-warmup`; `docker exec -it vg-warmup sh` (shell inside); `docker ps` / `docker image ls` / `docker history`; `docker stop`, `docker rm`, `docker rmi`.

- **Image** — a read-only, versioned template: a filesystem snapshot plus metadata (command, ports, environment). Built once, run many times.
- **Container** — a running (or stopped) instance of an image: an isolated process with its own filesystem view, network and a thin writable layer on top of the image.
- **Layer** — each Dockerfile instruction adds a layer, a diff of files on top of the previous ones. Layers are content-addressed, cached and shared between images, which makes rebuilds and downloads fast.
- **Volume** — storage that lives outside the container's writable layer, so data survives when the container is deleted (named volumes are managed by Docker; bind mounts map a host folder).
- **Port mapping** — `-p HOST:CONTAINER` forwards a port on my machine to a port inside the container's private network; without it the service is unreachable from outside.

> **Screenshots 6.1–6.6** — run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`.

## 6.2 Dockerfile for the web app

`docker/Dockerfile` (one image used for both the web app and the cloud API; `APP_MODULE` selects which):

1. `FROM python:3.11-slim` — Debian slim base (~50 MB compressed) instead of the 1 GB full image.
2. `ENV` — no `.pyc` files, unbuffered logs, `PIP_NO_CACHE_DIR=1`, quiet TensorFlow.
3. `COPY requirements-app.txt` then `RUN pip install …` — **dependencies before source code.** Docker reuses a cached layer as long as the instruction and its inputs are unchanged. The requirements file changes rarely, so the slow ~1 GB TensorFlow install is cached; when I edit Python code only the final `COPY` layers are rebuilt (seconds, not minutes). If the code were copied first, every code edit would invalidate the cache and reinstall TensorFlow.
4. **TensorFlow package chosen deliberately:** `tensorflow-cpu==2.19.0` on amd64 — the servers have no GPU, and the CPU build omits the CUDA/cuDNN libraries (hundreds of MB). On arm64 (Apple Silicon, Oracle Ampere) the standard `tensorflow` wheel is already CPU-only. `keras==3.15.1` is pinned to the exact version that saved the model.
5. Delete `__pycache__` and bundled test folders in the same `RUN`, so they never enter a layer.
6. `useradd --uid 10001 vg` and `USER vg` — **non-root**: if the app is compromised, the attacker cannot modify the image, install packages or write to the read-only model mount.
7. `COPY --chown=vg:vg visionguard/ app/ api/` — source last.
8. `HEALTHCHECK` calls `/health` every 30 s (90 s start period for model loading); Docker/Compose mark the container healthy/unhealthy.
9. `CMD gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 $APP_MODULE`.

`.dockerignore` excludes `.git`, `.env*`, `data/`, `models/`, `mlflow.db`, reports, tests and labs. This keeps the build context small and — importantly — **keeps secrets out of the image**: `.env` can never be copied in, and secrets are only injected at run time through `env_file`.

> **Screenshot 6.7** — `docker compose exec web sh -c 'whoami; id'` showing `vg`, uid 10001.

## 6.3 Compose stack

`docker-compose.yml` defines two services on a private bridge network `backend`:

- **`db`** — `postgres:16-alpine`; credentials from `.env`; data in the **named volume `pgdata`**; a health check runs `pg_isready` every 5 s; **no `ports:` section**, so it is reachable only from containers on `backend`.
- **`web`** — built from `docker/Dockerfile`; `DATABASE_URL=postgresql+psycopg2://…@db:5432/…` — the database is addressed by its **service name `db`**, which Compose's internal DNS resolves; `depends_on: db: condition: service_healthy` so the app starts only after Postgres accepts connections (plus a small connection-retry loop in the app); port 8000 published.

All passwords come from an untracked `.env` (template: `.env.example`).

**How the model reaches the container: mounted, not built in.** `./models` is bind-mounted read-only at `/models`. Reasons: the image stays independent of any particular model (one image, many model versions); promoting a new champion means replacing one file and restarting — no rebuild or re-push of a 1.5 GB image; the model file is versioned by DVC/MLflow, not by Docker layers; and the read-only mount means the app cannot alter it. `models/champion.keras` arrives on a machine via `dvc pull` (or `python scripts/register_model.py`, which exports the `@champion` version from the registry once). The serving app never calls DagsHub per request. The trade-off — the container is not self-contained — is acceptable because Compose declares the mount explicitly.

> **Screenshot 6.8** — `docker compose up -d --build` followed by `docker compose ps` with both services `(healthy)`.
> **Screenshot 6.9** — Proof the DB port is closed: `docker compose port db 5432` prints nothing / `nc -vz localhost 5432` fails, while `docker compose exec web python -c "import socket; socket.create_connection(('db',5432)); print('db reachable from web')"` succeeds.

## 6.4 Image analysis

**Optimised image** (`docker/Dockerfile`): `python:3.11-slim`, `tensorflow-cpu`, `PIP_NO_CACHE_DIR=1`, `__pycache__`/test folders deleted in the same layer, `.dockerignore`, dependencies before code.
**Naive image** (`docker/Dockerfile.naive`): full `python:3.11` base, the generic `tensorflow` wheel, pip cache kept, no `.dockerignore`, `COPY . .` first.

Table 6.1 — Image sizes (`docker image ls`)

| Image | Size |
|---|---|
| `vg-naive` (before) | NAIVE_SIZE |
| `23l-0517_a2_ops-web` (after) | OPT_SIZE |

**Largest layers** (`docker history`): by far the biggest is the `RUN pip install … tensorflow…` layer (TensorFlow, Keras and their dependencies — well over 1 GB uncompressed), followed by the Debian base layers; our own source code layers are only kilobytes, which confirms that ordering dependencies before code is what makes rebuilds fast. In the naive image the base layer alone is ~1 GB (compilers and dev headers we never use), the build context includes data, models and `.git` because there is no `.dockerignore`, and the pip cache stays in the layer.

**Conclusion.** The size is dominated by TensorFlow itself, so the slim base, the CPU-only wheel and not keeping caches give the largest savings; the next step would be serving with TFLite / LiteRT instead of full TensorFlow.

> **Screenshot 6.10** — `docker image ls` showing both images and `docker history 23l-0517_a2_ops-web` showing the largest layers.

## 6.5 Operations and troubleshooting

`scripts/ops_demo.sh ops` demonstrates `docker compose up -d`, `stop web`, `start web`, `restart web`, `logs web` / `logs db`, and a shell inside each service (`exec web sh`, `exec db psql …`).

**Deliberate failure — wrong database password.** `scripts/ops_demo.sh break-password` changes `POSTGRES_PASSWORD` in `.env` and recreates only the web container. The database keeps its original password, because Postgres reads `POSTGRES_PASSWORD` only the first time the volume is initialised.

*Diagnosis from logs only.* `docker compose ps` shows `web` restarting, and `docker compose logs web` (Screenshot 6.12) ends with the lines below. The excerpt here is from my rehearsal of the same failure — the same Gunicorn command against a local PostgreSQL 16 on port 5433 — before running it in Compose, where the host reads `db` and the port `5432`:

```text
[2026-11-10 01:32:57 +0500] [9183] [INFO] Booting worker with pid: 9183
[2026-11-10 01:33:20 +0500] [9183] [ERROR] Exception in worker process
psycopg2.OperationalError: connection to server at "127.0.0.1", port 5433 failed: FATAL:  password authentication failed for user "visionguard"
[2026-11-10 01:33:20 +0500] [9182] [ERROR] Shutting down: Master
[2026-11-10 01:33:20 +0500] [9182] [ERROR] Reason: Worker failed to boot.
```

The key line is `FATAL: password authentication failed for user "visionguard"`: the network and the DB container are fine (the connection reached Postgres), but the credentials are rejected — so the problem is configuration, not code. *Fix:* restore the original password in `.env` (`ops_demo.sh fix-password`) and recreate `web`; `docker compose ps` shows both services healthy again. (If the password really had to change, the correct fix is `ALTER USER … PASSWORD …` inside Postgres, or recreating the volume.) A second scenario, `break-model`, renames `champion.keras`: the logs show `MODEL LOAD FAILED: FileNotFoundError` and `/health` returns 503 with the error, and the container turns `unhealthy`.

> **Screenshots 6.11–6.13** — ops commands; the failure logs; healthy again after the fix.

## 6.6 Tool lab: GitHub Actions (CI)

**Toy part.** Throwaway repository `23L-0517_A2_Ops-ci-toy` (`labs/ci_toy/`): `calc.py`, `tests/test_calc.py` and `.github/workflows/test.yml`, triggered on every push, with a **matrix of Python 3.10 and 3.12** — the job runs twice in parallel. I pushed a bug (`add` subtracting) → **red** run, both matrix legs failing on `test_add`; reverted it → **green** run.

**Apply.** `.github/workflows/ci.yml` for VisionGuard runs on every push and pull request:

1. `lint-and-test`: checkout → Python 3.11 with pip cache → install `requirements-app.txt` + pytest + flake8 → `flake8` → `pytest` (20 tests; a fake predictor replaces the Keras model, so CI does not download TensorFlow for the tests).
2. `docker-build` (after tests pass): Buildx builds `docker/Dockerfile` with GitHub Actions layer cache, then smoke-tests the image (`whoami`, `import tensorflow, app, api`).

The status badge is at the top of `README.md`.

**Where secrets live.** In the repository's *Settings → Secrets and variables → Actions*; a workflow reads them as `${{ secrets.NAME }}` and GitHub masks them in logs. They must never be written in the workflow file because that file is committed: anyone who can read the repository (or its history, forever, even after deletion) could read the secret, and forks would copy it. This workflow needs none (it builds but does not push the image); a registry token for pushing would be stored as a secret.

> **Screenshot 6.14** — Toy repository Actions tab: red run (matrix 3.10 / 3.12) and green run.
> **Screenshot 6.15** — VisionGuard Actions run: both jobs green. **Screenshot 6.16** — README with the passing badge.
