#!/usr/bin/env bash
# Semi-automatic evidence capture for the report (macOS, run in Terminal.app from the project folder).
#   bash scripts/capture.sh cli       # Task 1/2/4 terminal evidence          (~2 min)
#   bash scripts/capture.sh web       # DagsHub + GitHub pages in your browser (~3 min)
#   bash scripts/capture.sh mlflowui  # local MLflow UI (toy runs)             (~1 min)
#   bash scripts/capture.sh dvctoy    # DVC toy lab -> DagsHub                  (~2 min)
#   bash scripts/capture.sh docker    # Docker warm-up, Compose stack, ops      (~15 min first time)
#   bash scripts/capture.sh failure   # wrong DB password -> logs -> fix        (~3 min)
#   bash scripts/capture.sh remote    # local app -> AWS API, 6 uploads         (~2 min)
#   bash scripts/capture.sh vmlogs    # API logs on the EC2 instance            (~10 s)
# Terminal shots capture the Terminal window; web shots capture the whole screen (address bar + clock).
# First run: macOS asks to allow Screen Recording for Terminal -> allow, then re-run.
set -u
cd "$(dirname "$0")/.."
mkdir -p report/shots
DH=https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops
ML=$DH.mlflow
GH=https://github.com/ahsan24naeem/23L-0517_A2_Ops
TUNED=6920fcc0421e4b06964eb0cd1610b14c
AUTOLOG=91894c33aee947a2a43cc661cde7edde
ALL='"67fdec04244c44fdbe73c130524d9c42","a88564aefb2d462b9d9486ca8b97307e","af6e78fc494d4ee78c5c633458b25475","5b8ca37b6ab64ad690c9e06dfccf0a26","1d37e45ea29f4e4b9f44d88ca170af5b","578b1934abc444f38e3076ef8710b52e","b83189014d12479b8c56459b632bb809","91894c33aee947a2a43cc661cde7edde","97f2974a435044bfa84ed9b915b3ccaa","6920fcc0421e4b06964eb0cd1610b14c"'

banner() { clear; printf '\033[1;34m[Screenshot %s] %s@%s  %s  -  23L-0517_A2_Ops\033[0m\n$ %s\n' "$1" "$(whoami)" "$(hostname -s)" "$(date '+%Y-%m-%d %H:%M:%S')" "$2"; }
tshot() { sleep 1; local id; id=$(osascript -e 'tell application "Terminal" to id of front window' 2>/dev/null)
  if [ -n "$id" ]; then screencapture -x -o -l"$id" "report/shots/$1.png"; else screencapture -x "report/shots/$1.png"; fi
  echo "   -> saved report/shots/$1.png"; }
run() { banner "$1" "$2"; eval "$2"; tshot "$1"; }            # run a command, capture the terminal
pause() { printf '\n\033[1;33m%s - press Enter to continue\033[0m' "${1:-}"; read -r _; }
wshot() { # open a page, wait, capture the screen
  echo "[Screenshot $1] opening $2"; open "$2"; sleep "${3:-9}"
  if [ "${4:-}" = ask ]; then printf '   arrange the page (scroll/click), then come back here and press Enter'; read -r _; osascript -e 'tell application "Terminal" to set miniaturized of front window to true'; sleep 1; fi
  screencapture -x "report/shots/$1.png"; echo "   -> saved report/shots/$1.png"
  [ "${4:-}" = ask ] && osascript -e 'tell application "Terminal" to set miniaturized of every window to false'
  osascript -e 'tell application "Terminal" to activate'; }

case "${1:-help}" in
cli)
  run 1.1 "python scripts/check_env.py 2>&1 | grep -v -E 'oneDNN|cpu_feature|rebuild|absl'"
  run 1.2 "python scripts/audit_dataset.py | head -45"
  run 2.1 "python -c \"import keras; keras.models.load_model('models/cnn_baseline.keras').summary()\" 2>&1 | grep -v -E 'WARNING|saveable'"
  run 3.2 "git check-ignore -v .env; git status --short; echo '(.env is not listed -> ignored)'"
  run 4.1 "python scripts/demo_inference.py 2>&1 | grep -v -E 'WARNING|oneDNN|saveable|UserWarning'"
  run 1.6 "cat data/raw.dvc; dvc status -c 2>&1 | tail -3"
  ;;
web)
  wshot 3.1 "$DH"
  wshot 1.6b "$DH/src/main/data"
  wshot 3.5 "$ML/#/experiments/0" 12
  wshot 3.6 "$ML/#/experiments/1/runs/$TUNED" 12 ask
  wshot 3.6b "$ML/#/metric?runs=%5B%22$TUNED%22%5D&metric=%22val_loss%22&experiments=%5B%221%22%5D&plot_metric_keys=%5B%22val_loss%22,%22loss%22%5D" 12
  wshot 3.7 "$ML/#/experiments/1/runs/$AUTOLOG" 12 ask
  wshot 3.8 "$ML/#/experiments/1/runs/$TUNED/artifacts" 12 ask
  wshot 3.9 "$ML/#/compare-runs?runs=%5B$(printf %s "$ALL" | sed 's/"/%22/g')%5D&experiments=%5B%221%22%5D" 15 ask
  wshot 3.10 "$ML/#/models/visionguard" 12
  wshot 6.15 "$GH/actions" 8
  wshot 6.16 "$GH" 8
  ;;
mlflowui)
  mlflow ui --backend-store-uri sqlite:///mlflow_toy.db --port 5001 >/tmp/mlflowui.log 2>&1 & PID=$!; sleep 6
  wshot 3.3 "http://127.0.0.1:5001/#/experiments/1" 6 ask
  echo "In the browser: tick both runs -> Compare."; wshot 3.4 "http://127.0.0.1:5001/#/experiments/1" 3 ask
  kill $PID
  ;;
dvctoy)
  : "${DAGSHUB_TOKEN:?run as: DAGSHUB_TOKEN=<token> bash scripts/capture.sh dvctoy}"
  DAGSHUB_USER=ahsan24naeem DAGSHUB_REPO=23L-0517_A2_Ops-dvc-toy STOP_AFTER_CHECKOUT=1 bash labs/dvc_toy/run_dvc_toy.sh; tshot 1.3
  pause "next: remote, push, delete, pull"
  clear; DAGSHUB_USER=ahsan24naeem DAGSHUB_REPO=23L-0517_A2_Ops-dvc-toy RESUME=1 bash labs/dvc_toy/run_dvc_toy.sh; tshot 1.4
  wshot 1.5 "https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops-dvc-toy/commits/main" 8
  ;;
docker)
  docker info >/dev/null 2>&1 || { echo "Start Docker Desktop first"; exit 1; }
  mkdir -p labs/docker_warmup/site && echo "<h1>Hello from a mounted volume - 23L-0517 $(date)</h1>" > labs/docker_warmup/site/index.html
  docker rm -f vg-warmup >/dev/null 2>&1
  run 6.1 "docker pull nginx:alpine && docker run -d --name vg-warmup -p 8080:80 -v \"\$PWD/labs/docker_warmup/site\":/usr/share/nginx/html:ro nginx:alpine && docker ps"
  run 6.2 "curl -s http://localhost:8080; echo '<p>edited on the host</p>' >> labs/docker_warmup/site/index.html; curl -s http://localhost:8080"
  run 6.3 "docker logs vg-warmup 2>&1 | tail -8"
  run 6.4 "docker exec vg-warmup sh -c 'whoami; hostname; ls /usr/share/nginx/html; head -2 /etc/os-release'"
  run 6.5 "docker image ls nginx; docker history nginx:alpine | head -8"
  run 6.6 "docker stop vg-warmup && docker rm vg-warmup && docker rmi nginx:alpine && docker ps -a | grep vg-warmup || echo 'container and image removed'"
  [ -f .env ] || { echo ".env missing (cp .env.example .env)"; exit 1; }
  chmod -R a+rX models   # the container runs as uid 10001 and must be able to read the mounted model
  banner 6.8 "docker compose up -d --build  (first build ~5-10 min)"; docker compose up -d --build
  echo "waiting for healthy..."; for i in $(seq 1 40); do docker compose ps | grep -q "web.*healthy" && break; sleep 5; done
  run 6.8 "docker compose ps"
  run 6.7 "docker compose exec web sh -c 'whoami; id'"
  run 6.9 "docker compose port db 5432 || echo '(no published port)'; nc -vz -G 2 localhost 5432; docker compose exec web python -c \"import socket; socket.create_connection(('db',5432)); print('db reachable from web on the internal network')\""
  for f in data/raw/RealArt/1032249.jpg data/raw/AiArtData/midjourney20images.jpg data/raw/RealArt/152752.jpg; do curl -s -o /dev/null -F "image=@$f" localhost:8000/predict; done
  run 4.9 "curl -s localhost:8000/health | python -m json.tool"
  wshot 5.1 "http://localhost:8000/history" 4
  run 5.2 "docker compose exec db sh -c 'psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -c \"select id, created_at, filename, label, round(confidence::numeric,3) conf, model_version, inference_mode from predictions order by id;\"'"
  run 6.10 "docker image ls | grep -E 'REPOSITORY|a2_ops|vg-naive'; docker history --format 'table {{.Size}}\t{{.CreatedBy}}' 23l-0517_a2_ops-web:latest | cut -c1-110 | head -12"
  run 6.11 "docker compose stop web && docker compose start web && docker compose restart web && docker compose ps && docker compose logs --tail 6 web && docker compose exec db sh -c 'psql -U \"\$POSTGRES_USER\" -c \"select version();\"' | head -3"
  ;;
naive)
  rm -rf /tmp/naive && mkdir -p /tmp/naive && rsync -a --exclude .venv --exclude data --exclude mlruns --exclude .git --exclude report --exclude reports ./ /tmp/naive/ && rm -f /tmp/naive/.dockerignore
  docker build -f /tmp/naive/docker/Dockerfile.naive -t vg-naive /tmp/naive
  run 6.10 "docker image ls | grep -E 'REPOSITORY|a2_ops|vg-naive'; docker history --format 'table {{.Size}}\t{{.CreatedBy}}' 23l-0517_a2_ops-web:latest | cut -c1-110 | head -12"
  ;;
failure)
  cp .env .env.bak && sed -i '' 's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=wrong-password-123/' .env
  banner 6.12 "docker compose up -d --force-recreate web; sleep 30; docker compose ps; docker compose logs --tail 15 web"
  docker compose up -d --force-recreate web >/dev/null 2>&1; sleep 30; docker compose ps; docker compose logs --tail 15 web | cut -c1-160; tshot 6.12
  pause "diagnosis: password authentication failed -> restore .env"
  mv .env.bak .env
  banner 6.13 "restore .env; docker compose up -d --force-recreate web"
  docker compose up -d --force-recreate web >/dev/null 2>&1; for i in $(seq 1 30); do docker compose ps | grep -q "web.*healthy" && break; sleep 5; done
  docker compose ps; curl -s localhost:8000/health; echo; tshot 6.13
  ;;
remote)
  rm -f instance/remote.db
  DATABASE_URL=sqlite:///instance/remote.db INFERENCE_MODE=remote PORT=5000 python -m app >/tmp/remote_app.log 2>&1 & PID=$!; sleep 6
  n=0
  for f in data/raw/RealArt/1032249.jpg data/raw/RealArt/152752.jpg "data/raw/RealArt/12-reasons-why-people-like-your-posts-on-social-media-001.jpg" \
           data/raw/AiArtData/midjourney20images.jpg data/raw/AiArtData/Midjourney-Mishra-abstract-dancing-painting.jpg data/raw/AiArtData/2022-11-10-image-12.jpg; do
    n=$((n+1)); curl -s -o /dev/null -w "upload $n: %{http_code}\n" -F "image=@$f" localhost:5000/predict
    wshot 8.$n "http://127.0.0.1:5000/result/$n" 3
  done
  wshot 8.7 "http://127.0.0.1:5000/history" 3
  wshot 8.8 "http://127.0.0.1:5000/health" 3
  curl -s localhost:5000/api/history | python -m json.tool > reports/remote_history.json
  kill $PID; echo "saved reports/remote_history.json"
  ;;
vmlogs)
  run 7.10 "ssh vg 'cd ~/23L-0517_A2_Ops && docker compose -f deploy/oracle/docker-compose.yml logs --tail 20 | cut -c1-150'"
  ;;
*) sed -n 2,13p "$0" ;;
esac
