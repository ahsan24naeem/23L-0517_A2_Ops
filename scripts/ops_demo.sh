#!/usr/bin/env bash
# Task 6.5 - operations and a deliberate failure, diagnosed only from logs.
# Usage: bash scripts/ops_demo.sh [ops|break-password|fix-password|break-model|fix-model]
set -x
cd "$(dirname "$0")/.."
case "${1:-ops}" in
  ops)
    docker compose up -d --build           # start (builds the web image if needed)
    docker compose ps                      # both services should become "healthy"
    docker compose stop web && docker compose ps
    docker compose start web
    docker compose restart web
    docker compose logs --tail 20 web
    docker compose logs --tail 10 db
    docker compose exec web sh -c 'whoami; id; python -c "import tensorflow as tf; print(tf.__version__)"; ls -l /models'
    docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "select id, created_at, label, confidence, model_version from predictions order by id desc limit 5;"'
    ;;
  break-password)
    # Postgres only reads POSTGRES_PASSWORD when the volume is first initialised, so changing it
    # now makes the web app use a password the database does not know.
    cp .env .env.bak && sed -i.tmp 's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=wrong-password-123/' .env && rm -f .env.tmp
    docker compose up -d --force-recreate web
    sleep 25; docker compose ps; docker compose logs --tail 30 web
    ;;
  fix-password)
    mv .env.bak .env && docker compose up -d --force-recreate web && sleep 40 && docker compose ps && curl -s localhost:8000/health
    ;;
  break-model)
    mv models/champion.keras models/champion.keras.moved
    docker compose restart web; sleep 20; docker compose logs --tail 20 web; curl -s localhost:8000/health; docker compose ps
    ;;
  fix-model)
    mv models/champion.keras.moved models/champion.keras && docker compose restart web && sleep 40 && curl -s localhost:8000/health
    ;;
esac
