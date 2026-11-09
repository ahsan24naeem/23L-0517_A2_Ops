#!/usr/bin/env bash
# Task 1.4 toy part - DVC basics on a tiny CSV, in a NEW empty folder.
# Usage:
#   bash labs/dvc_toy/run_dvc_toy.sh                      # local-folder remote
#   DAGSHUB_USER=<you> DAGSHUB_TOKEN=<token> bash labs/dvc_toy/run_dvc_toy.sh   # DagsHub remote
# Each step is echoed so the terminal output can be screenshotted for the report.
set -euo pipefail
TOY=${TOY_DIR:-$HOME/23L-0517_A2_Ops-dvc-toy}
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
run()  { printf '\033[0;90m$ %s\033[0m\n' "$*"; eval "$@"; }

step "0. fresh, empty folder: $TOY   ($(whoami) @ $(date '+%Y-%m-%d %H:%M'))"
rm -rf "$TOY" && mkdir -p "$TOY" && cd "$TOY"

step "1. initialise Git and DVC"
run git init -q -b main
run git config user.name "\"${GIT_NAME:-Ahsan Naeem}\""
run dvc init -q
run git commit -q -m "\"init dvc\""

step "2. small CSV, tracked by DVC (not Git)"
printf 'id,city,temp_c\n1,Lahore,31\n2,Karachi,33\n3,Islamabad,27\n' > weather.csv
run cat weather.csv
run dvc add weather.csv
run cat weather.csv.dvc
run cat .gitignore
run git add weather.csv.dvc .gitignore
run git commit -q -m "\"data v1: 3 cities\""

step "3. change the data -> version 2"
printf '4,Peshawar,29\n5,Quetta,22\n' >> weather.csv
run dvc add weather.csv
run git add weather.csv.dvc
run git commit -q -m "\"data v2: 5 cities\""
run git log --oneline
run "git diff HEAD~1 -- weather.csv.dvc"

step "4. go back to version 1 (Git restores the pointer, DVC restores the data)"
run git checkout -q HEAD~1 -- weather.csv.dvc
run dvc checkout
run cat weather.csv
step "   ...and return to the latest version"
run git checkout -q HEAD -- weather.csv.dvc
run dvc checkout
run wc -l weather.csv

step "5. configure remote storage"
if [[ -n "${DAGSHUB_USER:-}" && -n "${DAGSHUB_TOKEN:-}" ]]; then
  REPO=${DAGSHUB_REPO:-23L-0517_A2_Ops-dvc-toy}
  run dvc remote add -d origin "https://dagshub.com/$DAGSHUB_USER/$REPO.dvc"
  # credentials go to .dvc/config.local, which is NOT committed
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USER"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
  echo "(credentials stored in .dvc/config.local - git-ignored)"
  git remote add origin "https://dagshub.com/$DAGSHUB_USER/$REPO.git" || true
else
  run mkdir -p "$TOY-remote"
  run dvc remote add -d localremote "$TOY-remote"
fi
run cat .dvc/config
run git add .dvc/config
run git commit -q -m "\"configure dvc remote\""

step "6. push data to the remote"
run dvc push
if [[ -n "${DAGSHUB_USER:-}" && -n "${DAGSHUB_TOKEN:-}" ]]; then
  git push -q "https://$DAGSHUB_USER:$DAGSHUB_TOKEN@dagshub.com/$DAGSHUB_USER/$REPO.git" main && echo "git history pushed to DagsHub"
fi

step "7. delete the local data AND the local cache"
run rm -rf weather.csv .dvc/cache
run "ls -la | grep weather || true"

step "8. pull it back from the remote"
run dvc pull
run cat weather.csv
step "done"
