#!/usr/bin/env bash
# Build report/23L-0517_VisionGuard.docx and .pdf from the Markdown chapters.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=23L-0517_VisionGuard
for f in 01_front_task1.md 02_task2.md 03_task3.md 04_tasks4_6.md 07_tasks7_8.md 09_conclusion.md; do [ -f "$f" ] && { cat "$f"; echo; }; done > build/_raw.md
python3 build/prep_md.py build/_raw.md build/_all.md
pandoc build/_all.md -o $OUT.docx --reference-doc=build/reference.docx --resource-path=.:..:build \
  --highlight-style=tango
python3 build/postprocess.py $OUT.docx
soffice --headless --convert-to pdf $OUT.docx >/dev/null 2>&1
ls -la $OUT.*
