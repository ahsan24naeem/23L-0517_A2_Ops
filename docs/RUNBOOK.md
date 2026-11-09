# VisionGuard — runbook for the steps on your Mac (23L-0517)

Everything that could be done without your accounts is already done: code, data audit, EDA,
splits, 10 training runs + Keras Tuner (stored in the local MLflow store `mlflow.db` + `mlruns/`),
figures, the registered champion model (`models/champion.keras`), tests, Dockerfiles, Compose,
CI, Oracle deploy files and the report draft (`report/23L-0517_VisionGuard.docx`).

What is left needs **your accounts or your machine**, and the report needs **your screenshots**
(terminal prompt with your username + date visible). Screenshot IDs below match the grey boxes in
the report. Take them with `Cmd+Shift+4` and paste each one under its box in the .docx.

Replace `ahsan24naeem`, `ahsan24naeem`, `ORACLE_IP` with your real values everywhere.

---

## 0. One-time setup (≈10 min)

```bash
cd ~/Desktop/FAST/MLops/"Assignment 22"/23L-0517_A2_Ops
brew install python@3.11            # if `python3.11 --version` fails
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/relocate_mlflow_store.py      # fixes artifact paths in mlflow.db for this folder
python scripts/check_env.py                  # Screenshot 1.1
pytest -q                                    # should print 20 passed
```

Install Docker Desktop (you have it) and Postman (https://www.postman.com/downloads/).

## 1. GitHub (≈5 min)

1. github.com → **New repository** → name `23L-0517_A2_Ops`, **Public**, *no* README/.gitignore.
2. Push the existing history (commit hashes are preserved):
   ```bash
   git remote add origin https://github.com/ahsan24naeem/23L-0517_A2_Ops.git
   git push -u origin main
   ```
3. Open the **Actions** tab — the CI workflow starts automatically. Screenshot 6.15 when green.

## 2. DagsHub project + secrets (≈10 min)

1. dagshub.com → sign in with GitHub → **Create → New Repository → Connect a repository →
   GitHub** → pick `23L-0517_A2_Ops`. (Screenshot 3.1: the DagsHub project page.)
2. Avatar → **Your settings → Tokens** → copy the default token (or create one).
3. On the repo page: **Remote → Experiments** shows the MLflow URI
   `https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops.mlflow`.
4. Create `.env` (never committed — proof for the report):
   ```bash
   cp .env.example .env
   open -e .env          # fill MLFLOW_TRACKING_URI / USERNAME / PASSWORD(=token), POSTGRES_PASSWORD,
                         # SECRET_KEY and MODEL_API_KEY (python -c "import secrets;print(secrets.token_urlsafe(32))")
   git check-ignore -v .env && git status --short    # Screenshot 3.2: .env is ignored
   ```

## 3. DVC → DagsHub storage (Task 1.4 apply, ≈10 min + upload time ~500 MB)

```bash
dvc remote add -d origin https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user ahsan24naeem
dvc remote modify origin --local password <your-dagshub-token>
dvc add data/raw                       # re-fills the local cache; md5 stays 8f23d6dc…
dvc add models/champion.keras          # version the champion model too
git add .dvc/config data/raw.dvc models/champion.keras.dvc models/.gitignore
git commit -m "DVC remote on DagsHub; track champion model" && git push
dvc push                               # Screenshot 1.6 (cat data/raw.dvc + dvc push)
```

DVC toy (Task 1.4 toy) — first create an **empty** DagsHub repo named `23L-0517_A2_Ops-dvc-toy`
(*Create → New Repository*, not connected), then:

```bash
ahsan24naeem=ahsan24naeem DAGSHUB_TOKEN=<token> bash labs/dvc_toy/run_dvc_toy.sh   # Screenshots 1.3, 1.4
```
Open the toy repo on DagsHub (Screenshot 1.5).

## 4. MLflow (Task 3, ≈15 min)

```bash
# 3.1 local toy: two runs with different learning rates
python labs/mlflow_toy.py
mlflow ui --backend-store-uri sqlite:///mlflow_toy.db --port 5001
#   http://127.0.0.1:5001 → experiment toy-fashion-mnist → Screenshot 3.3 (both runs),
#   tick both → Compare → Screenshot 3.4 (parallel coordinates / metric chart). Ctrl+C to stop.
# 3.2 same toy run against DagsHub
python labs/mlflow_toy.py --remote                     # Screenshot 3.5 (toy runs on DagsHub)
# 3.3-3.5 copy the 10 VisionGuard runs + tuner run to DagsHub (no retraining)
python scripts/sync_to_dagshub.py --experiments VisionGuard VisionGuard-tuning
#   DagsHub → Experiments → open mnv2_finetune (manual) and mnv2_finetune_autolog:
#   Screenshots 3.6/3.7 (params, metric charts, tags), 3.8 (Artifacts tab), 3.9 (compare all runs)
# 3.6 model registry on DagsHub
python scripts/register_model.py                       # v1 challenger, v2 champion
python scripts/predict_by_alias.py data/raw/RealArt/<any real image>   # Screenshot 3.11
#   DagsHub → Models (or MLflow UI → Models) → visionguard with @champion  Screenshot 3.10
```

You can also look at the local store: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## 5. Flask app locally (Task 4/5, ≈10 min)

```bash
python -m app            # http://127.0.0.1:5000 (debug off, SQLite in instance/)
```
Screenshots: 4.5 upload page; 4.6/4.7 results (one AI, one real image from `data/raw`); 4.2 upload a
`.txt` (rename any text file to `notes.txt`); 4.3 an image > 5 MB (e.g. the largest file in
`data/raw/RealArt`, `ls -S data/raw/RealArt | head -1`); 4.4 `echo hello > fake.jpg` and upload it;
4.8 History page.

## 6. Docker (Task 6, ≈30 min, first build downloads ~1 GB)

```bash
bash labs/docker_warmup/warmup.sh                   # Screenshots 6.1-6.6 (also do `docker exec -it vg-warmup sh`)
docker compose up -d --build                        # web + db
docker compose ps                                   # Screenshot 6.8: both (healthy)
docker compose exec web sh -c 'whoami; id'          # Screenshot 6.7
docker compose port db 5432; nc -vz localhost 5432  # nothing published / connection refused
docker compose exec web python -c "import socket; socket.create_connection(('db',5432)); print('db reachable from web')"
#   Screenshot 6.9
open http://localhost:8000          # upload 2-3 images, then /history and /health (Screenshot 4.9, 5.1)
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "select id,created_at,label,confidence,model_version from predictions order by id desc limit 5;"'   # Screenshot 5.2
# 6.4 image size before/after
docker image ls 23l-0517_a2_ops-web
docker history --format "table {{.Size}}\t{{.CreatedBy}}" 23l-0517_a2_ops-web | head -12
mkdir -p /tmp/naive && rsync -a --exclude .venv --exclude data --exclude mlruns --exclude .git ./ /tmp/naive/ && rm /tmp/naive/.dockerignore
docker build -f /tmp/naive/docker/Dockerfile.naive -t vg-naive /tmp/naive
docker image ls | grep -E "vg-naive|a2_ops-web"     # Screenshot 6.10 - send me the two sizes
# 6.5 operations + deliberate failure
bash scripts/ops_demo.sh ops                        # Screenshot 6.11
bash scripts/ops_demo.sh break-password             # Screenshot 6.12 (logs show password authentication failed)
bash scripts/ops_demo.sh fix-password               # Screenshot 6.13
```

## 7. GitHub Actions toy (Task 6.6, ≈10 min)

Create an empty public repo `23L-0517_A2_Ops-ci-toy`, then follow `labs/ci_toy/README.md`
(push → green, push the bug → red, push the fix → green). Screenshot 6.14 = Actions tab with red
and green runs. Screenshot 6.16 = the README badge on the main repo.

## 8. Oracle Cloud (Task 7, ≈40 min)

1. Console → *Governance → Limits, Quotas and Usage* → filter "A1" (Screenshot 7.1).
2. *Compute → Instances → Create instance*: name `23L-0517_A2_Ops`, **Ubuntu 22.04 aarch64**,
   shape **VM.Standard.A1.Flex 1 OCPU / 6 GB**, public IP, upload `~/.ssh/id_ed25519.pub`
   (create with `ssh-keygen -t ed25519` if missing). Screenshot 7.2.
   If "Out of host capacity": switch availability domain or retry later.
3. Instance → subnet → **Default Security List → Add Ingress Rule**: `0.0.0.0/0`, TCP, port `8000`
   (Screenshot 7.3).
4. From your Mac:
   ```bash
   ssh ubuntu@ORACLE_IP
   git clone https://github.com/ahsan24naeem/23L-0517_A2_Ops.git && cd 23L-0517_A2_Ops
   bash deploy/oracle/setup_vm.sh && exit          # Screenshot 7.4, then log in again
   ```
5. Copy the model (from the Mac, project folder): `scp models/champion.keras models/champion.json ubuntu@ORACLE_IP:~/23L-0517_A2_Ops/models/`
6. On the VM:
   ```bash
   cd ~/23L-0517_A2_Ops
   printf 'MODEL_API_KEY=%s\nMAX_UPLOAD_MB=5\n' '<same key as in your Mac .env>' > .env
   docker compose -f deploy/oracle/docker-compose.yml up -d --build    # ~10 min on first build
   docker compose -f deploy/oracle/docker-compose.yml ps                # Screenshot 7.5
   docker compose -f deploy/oracle/docker-compose.yml logs --tail 20
   ```
7. From your Mac **on a different network (phone hotspot)**: `curl http://ORACLE_IP:8000/health` (Screenshot 7.6).
8. Postman → Import `deploy/postman/VisionGuard.postman_collection.json` → collection Variables:
   `base_url = http://ORACLE_IP:8000`, `api_key = <key>` → for requests 2, 3, 5 choose a file in
   Body → form-data → image (5 = any .txt). Send each, then *Run collection* (Screenshots 7.7-7.9).
   On the VM `docker compose -f deploy/oracle/docker-compose.yml logs --tail 20` (Screenshot 7.10).

## 9. Flask → cloud API (Task 8, ≈10 min)

```bash
INFERENCE_MODE=remote MODEL_API_URL=http://ORACLE_IP:8000 python -m app     # key comes from .env
```
Upload at least 6 images (3 real, 3 AI) at http://127.0.0.1:5000 → Screenshots 8.1-8.6 (result
pages show "Served by remote"), 8.7 History, 8.8 `/health`.

## 10. Finish the report

1. Send me (or fill in yourself): GitHub/DagsHub usernames, Oracle IP, the two image sizes from
   step 6, Postman results — I will regenerate the report with them.
2. Paste each screenshot under its grey box in `report/23L-0517_VisionGuard.docx`.
3. Put the final commit hash (`git rev-parse --short HEAD`) on the front page.
4. File → Export as PDF → `23L-0517_VisionGuard.pdf`. Submit the PDF, DagsHub link and API URL.
