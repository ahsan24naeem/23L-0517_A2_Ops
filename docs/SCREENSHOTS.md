# Screenshot checklist (23L-0517)

Take each one with `bash scripts/shot.sh <ID>` (add a delay: `bash scripts/shot.sh <ID> 5`).
Drag a rectangle, or press Space and click a window. Make sure your terminal prompt / browser
address bar and the date are visible. Files go to `report/shots/<ID>.png`.

| Done | ID | What to capture |
|---|---|---|
| [ ] | 1.1 | Terminal on my Mac: `source .venv/bin/activate && python scripts/check_env.py` showing user@host, date, all versions and "GPU available: False". |
| [ ] | 1.2 | Terminal: `python scripts/audit_dataset.py` printing the summary tables. |
| [ ] | 1.3 | Toy run: `git log --oneline` with the "data v1" and "data v2" commits and the pointer diff. |
| [ ] | 1.4 | Toy run: `dvc checkout` back to v1 (3 rows), then `rm -rf weather.csv .dvc/cache` → `dvc pull` restoring the file. |
| [ ] | 1.5 | DagsHub web page of the toy repository showing the commits and the DVC-tracked `weather.csv`. |
| [ ] | 1.6 | `cat data/raw.dvc` and `dvc push` for the real dataset; DagsHub project page showing the `data/raw` folder tracked by DVC. |
| [ ] | 2.1 | `model.summary()` of the baseline CNN (text also in `reports/runs/cnn_baseline/model_summary.txt` and attached to the MLflow run). |
| [ ] | 3.1 | DagsHub project page (connected to GitHub). |
| [ ] | 3.2 | `git check-ignore -v .env` → `.gitignore:2:.env` and `git status` not listing `.env`. |
| [ ] | 3.3 | Local MLflow UI listing the two toy runs (lr 0.001 and 0.01) with their metrics. |
| [ ] | 3.4 | Compare view of the two runs (metric chart over epochs / parallel coordinates). |
| [ ] | 3.5 | The two toy runs in DagsHub's MLflow UI. |
| [ ] | 3.6 | Run page of `mnv2_finetune` (manual): parameters, metric curves (train/val loss, val AUC, learning rate across both stages) and tags (git commit, DVC version, seed). |
| [ ] | 3.7 | Run page of `mnv2_finetune_autolog`: the parameters and metrics autologging produced. |
| [ ] | 3.8 | Artifacts tab of the chosen run showing these files. |
| [ ] | 3.9 | DagsHub/MLflow comparison of all runs (table + chart of val AUC / test F1). |
| [ ] | 3.10 | Registry page: `visionguard` with versions 1 and 2 and the `@champion` alias. |
| [ ] | 3.11 | Terminal output of `python scripts/predict_by_alias.py <image>`. |
| [ ] | 4.1 | Console output of the inference module for one known real image and one known AI image. |
| [ ] | 4.2 | Error page "Unsupported file type" (uploading a `.txt`). |
| [ ] | 4.3 | Error page "File too large" (uploading a > 5 MB image). |
| [ ] | 4.4 | Error page "Not a valid image" (a text file renamed `.jpg`). |
| [ ] | 4.5 | Upload page. |
| [ ] | 4.6 | Result page for an AI image. |
| [ ] | 4.7 | Result page for a real photo. |
| [ ] | 4.8 | History page. |
| [ ] | 4.9 | Browser at `http://localhost:8000/health` showing `"status": "ok"`, `"debug": false`, model version and `"database": {"backend": "postgresql", "ok": true}`. |
| [ ] | 5.1 | History page after several uploads. |
| [ ] | 5.2 | The same rows queried with `psql` inside the `db` container. |
| [ ] | 6.1 | run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`. |
| [ ] | 6.2 | run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`. |
| [ ] | 6.3 | run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`. |
| [ ] | 6.4 | run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`. |
| [ ] | 6.5 | run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`. |
| [ ] | 6.6 | run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`. |
| [ ] | 6.7 | `docker compose exec web sh -c 'whoami; id'` showing `vg`, uid 10001. |
| [ ] | 6.8 | `docker compose up -d --build` followed by `docker compose ps` with both services `(healthy)`. |
| [ ] | 6.9 | Proof the DB port is closed: `docker compose port db 5432` prints nothing / `nc -vz localhost 5432` fails, while `docker compose exec web python -c "import socket; socket.create_connection(('db',5432)); print('db reachable from web')"` succeeds. |
| [ ] | 6.10 | `docker image ls` showing both images and `docker history 23l-0517_a2_ops-web` showing the largest layers. |
| [ ] | 6.11 | ops commands; the failure logs; healthy again after the fix. |
| [ ] | 6.12 | ops commands; the failure logs; healthy again after the fix. |
| [ ] | 6.13 | ops commands; the failure logs; healthy again after the fix. |
| [ ] | 6.14 | Toy repository Actions tab: red run (matrix 3.10 / 3.12) and green run. |
| [ ] | 6.15 | VisionGuard Actions run: both jobs green. |
| [ ] | 6.16 | README with the passing badge. |
| [ ] | 7.7 | Postman: `/predict` with the API key, response JSON and passing tests. |
| [ ] | 7.8 | Postman: 401 without the key. |
| [ ] | 7.9 | Postman Runner / Console with all five results. |
| [ ] | 7.10 | `docker compose logs api` on the VM showing one log line per request. |
| [ ] | 8.1 | Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*. |
| [ ] | 8.2 | Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*. |
| [ ] | 8.3 | Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*. |
| [ ] | 8.4 | Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*. |
| [ ] | 8.5 | Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*. |
| [ ] | 8.6 | Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*. |
| [ ] | 8.7 | History page listing those requests with model version and mode `remote`. |
| [ ] | 8.8 | `/health` of the local app in remote mode (`"inference_mode": "remote"`, `api_url`), and the EC2 instance's API logs for the same requests. |
