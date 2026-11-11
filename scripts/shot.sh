#!/usr/bin/env bash
# Take a report screenshot on macOS and save it under its ID, e.g.:
#   bash scripts/shot.sh 4.6        -> drag a rectangle (or press Space, then click a window)
#   bash scripts/shot.sh 7.5 5      -> same, after a 5-second delay (time to switch windows)
# Files land in report/shots/<ID>.png; the report build inserts them under the matching box.
set -e
cd "$(dirname "$0")/.."
[ -z "$1" ] && { echo "usage: bash scripts/shot.sh <ID e.g. 4.6> [delay-seconds]"; exit 1; }
mkdir -p report/shots
f="report/shots/$1.png"
[ -n "$2" ] && { echo "capturing in $2 s..."; sleep "$2"; }
screencapture -i -x "$f" && [ -f "$f" ] && echo "saved $f" || echo "cancelled"
