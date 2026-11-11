# Screenshots — how each one is produced (23L-0517)

Run from the project folder in **Terminal.app** with `conda activate vg`. First time, macOS asks to allow
*Screen Recording* for Terminal: allow it (System Settings → Privacy & Security → Screen Recording), quit and reopen Terminal.
Everything lands in `report/shots/<ID>.png`; Claude rebuilds the report with them.

| Command | Produces |
|---|---|
| `bash scripts/capture.sh cli` | 1.1 env, 1.2 audit, 2.1 CNN summary, 3.2 .env ignored, 4.1 inference module, 1.6 data/raw.dvc |
| `bash scripts/capture.sh web` | 3.1 DagsHub repo, 1.6b data folder, 3.5 toy runs, 3.6/3.6b champion run + curves, 3.7 autolog run, 3.8 artifacts, 3.9 compare, 3.10 registry, 6.15 Actions, 6.16 README badge |
| `bash scripts/capture.sh mlflowui` | 3.3 local MLflow runs, 3.4 compare view (tick both runs → Compare when asked) |
| `DAGSHUB_TOKEN=<token> bash scripts/capture.sh dvctoy` | 1.3 two data versions + checkout, 1.4 push/delete/pull, 1.5 toy repo on DagsHub (create the empty DagsHub repo `23L-0517_A2_Ops-dvc-toy` first) |
| `bash scripts/capture.sh docker` | 6.1–6.6 warm-up, 6.8 compose healthy, 6.7 non-root, 6.9 DB port closed, 4.9 health, 5.1 history, 5.2 psql, 6.10 image layers, 6.11 ops |
| `bash scripts/capture.sh naive` | 6.10 again with the naive image for the before/after size |
| `bash scripts/capture.sh failure` | 6.12 wrong password in the logs, 6.13 healthy after the fix |
| `bash scripts/capture.sh remote` | 8.1–8.6 six uploads through the local app to the AWS API, 8.7 history, 8.8 health |
| `bash scripts/capture.sh vmlogs` | 7.10 API logs on the EC2 instance (run after Postman) |
| `bash scripts/shot.sh <ID>` (manual) | 4.4, 4.6, 4.7, 4.8 web app pages; 7.1 Oracle capacity error (optional); 7.2 EC2 instance; 7.3 security group; 7.7–7.9 Postman; 6.14 CI toy repo |

Already captured: 3.2, 3.11, 4.2, 4.3, 4.5, 7.4, 7.5, 7.6.
