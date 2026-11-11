# Task 7 — API Deployment (AWS EC2, pathway C)

**What runs where.** As the task specifies, the **model API** is deployed in the cloud and the **Flask web app runs on my laptop** and calls it over the internet. The API (`api/__init__.py`) is a small Flask service served by Gunicorn in the same Docker image as the web app (`APP_MODULE=api.wsgi:app`), with the champion model mounted read-only. It is stateless — the history database stays with the web app.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /` | none | service info: model name, registry version, threshold, input size |
| `GET /health` | none | `status`, model info, uptime, requests served (200 / 503) |
| `POST /predict` | `X-API-Key` header | multipart field `image` → `label`, `is_ai`, `confidence`, `p_ai`, `threshold`, `model_name`, `model_version`, `latency_ms`, sanitised `filename`, `sha256`, size |

The key is compared in constant time (`hmac.compare_digest`) and comes from the VM's `.env`; without it the API answers 401. The same upload validation as the web app (type 415, size 413, undecodable 422) applies.

## Research before deploying

- **First attempt: Oracle Cloud Always Free (pathway A).** Oracle's documentation (November 2026) lists Ampere A1 Always Free as 2 OCPUs / 12 GB in total (halved from 4 / 24 in June 2026) plus up to two x86 **E2.1.Micro** VMs (1 GB RAM), 200 GB block storage and 10 TB/month egress; idle Always Free instances can be reclaimed if CPU, network and memory stay below 20 % for 7 days. I created a VCN with the *VCN with Internet Connectivity* wizard (the inline "create public subnet" option in the instance form left the public-IP switch disabled), but every launch then failed with **"Out of capacity for shape"** for both A1.Flex and E2.1.Micro. My home region, Mumbai, has a single availability domain, so there was nowhere else to try. (An earlier instance also used an Oracle Linux image and an SSH key I did not have, so I terminated it.) The setup script supports Oracle Linux (`dnf`, `firewalld`) and Ubuntu, so this work was not lost.
- **Switch to AWS EC2 (pathway C).** Same design — one VM running the API with Docker Compose — on a **t3.micro** (2 vCPU burstable, **1 GB RAM**, x86-64), **Ubuntu Server 24.04**, **20 GiB gp3** root volume (the 8 GB default is too small for the ~2 GB TensorFlow image; free-tier storage covers up to 30 GB). AWS's current free plan gives new accounts credits rather than a permanent free VM, and a t3.micro costs only cents per hour, so I stop the instance after grading.
- **Memory.** TensorFlow import alone needs ~600 MB, so `setup_vm.sh` adds a **2 GB swap file** and the API runs a single Gunicorn worker. The Dockerfile picks the TensorFlow package per architecture (`tensorflow-cpu` on amd64, `tensorflow` on arm64) and the image is built on the VM.
- **Firewalls.** On AWS the **Security Group** is the network firewall (Ubuntu's own iptables has no REJECT rule there; on Oracle images the VM firewall must be opened as well, which the script also handles).

| Compose element | AWS equivalent |
|---|---|
| `api` service, `restart: unless-stopped` | Docker Engine + Compose on the EC2 instance; restarts after reboot |
| published port `8000:8000` | Security Group inbound rule TCP 8000 from 0.0.0.0/0 |
| `.env` (API key) | file on the instance only (could be AWS Secrets Manager / SSM Parameter Store) |
| bind-mounted `models/` | file on the instance's EBS volume, copied once at deploy time |
| health check | Docker `HEALTHCHECK` (an ALB target-group health check in a larger setup) |

## Step-by-step deployment

1. **Key pair.** EC2 → Key Pairs → *Import key pair* `23L-0517-key` with my existing public key (`~/.ssh/id_ed25519_university.pub`).
2. **Launch.** EC2 → *Launch instances*: name `23L-0517_A2_Ops`, AMI *Ubuntu Server 24.04 LTS*, type *t3.micro*, key pair `23L-0517-key`, auto-assign public IP **enabled**, new security group `23L-0517-sg` with **SSH 22 from My IP** and **Custom TCP 8000 from Anywhere**, 20 GiB gp3.
3. **SSH alias** on my Mac (`~/.ssh/config`, host `vg` → the public IP, user `ubuntu`, that key), so every later command is `ssh vg …`.
4. **Copy code + model.** `rsync -av --exclude data --exclude mlruns --exclude .git … ./ vg:~/23L-0517_A2_Ops/` — copies the code and `models/champion.keras` (the registry's `@champion`, exported once). The API never contacts DagsHub while serving.
5. **Prepare the VM.** `ssh vg "cd ~/23L-0517_A2_Ops && bash deploy/oracle/setup_vm.sh"` installs Docker Engine 29 + Compose, adds the user to the `docker` group and a 2 GB swap file, and makes the model world-readable for the container's non-root user.
6. **Secret.** On the VM: a random key (`secrets.token_urlsafe(32)`) written to `~/23L-0517_A2_Ops/.env` as `MODEL_API_KEY` — never committed.
7. **Build and start.** `docker compose -f deploy/oracle/docker-compose.yml up -d --build` (image built in ~3 min); `ps` shows the container `Up (healthy)` and the logs show `model loaded: visionguard v2`.
8. **Test from outside**, including from my phone hotspot (a different network):

```text
$ curl -s http://3.26.70.239:8000/health
{"error":null,"img_size":224,"model_name":"visionguard","model_version":"2","project":"23L-0517_A2_Ops",
 "requests_served":0,"service":"visionguard-model-api","status":"ok","threshold":0.71,"uptime_s":302}
$ curl -s -H "X-API-Key: ****" -F "image=@data/raw/RealArt/1032249.jpg" http://3.26.70.239:8000/predict
{"confidence":0.9753,"filename":"1032249.jpg","height":1198,"is_ai":false,"label":"Real","latency_ms":158.4,
 "model_name":"visionguard","model_version":"2","p_ai":0.0247,"sha256":"04e99a55…","threshold":0.71,"width":1797}
```

> **Screenshot 7.1** — Oracle Cloud "Out of capacity for shape" error when launching the Always Free instance (Mumbai).

> **Screenshot 7.2** — EC2 instance summary: `23L-0517_A2_Ops`, t3.micro, running, public IPv4 3.26.70.239.

> **Screenshot 7.3** — Security group `23L-0517-sg` inbound rules: SSH 22 from my IP, TCP 8000 from anywhere.

> **Screenshot 7.4** — End of `setup_vm.sh` on the instance: Docker 29.9 installed, 2 GB swap active.

> **Screenshot 7.5** — `docker compose ps` (healthy), logs with `model loaded: visionguard v2`, and `curl localhost:8000/health` on the instance.

> **Screenshot 7.6** — From my Mac over the internet: `/health`, `/predict` with the key (Real, 97.5 %) and without the key (`missing or invalid API key`).

**API URL:** `http://3.26.70.239:8000` — health check `http://3.26.70.239:8000/health`, prediction `POST http://3.26.70.239:8000/predict` (multipart field `image`, header `X-API-Key`).

## Testing with Postman

`deploy/postman/VisionGuard.postman_collection.json` is imported into Postman with collection variables `base_url` and `api_key`. Each request has test scripts:

| # | Request | Expected | Logged result |
|---|---|---|---|
| 1 | `GET /health` | 200, `status: ok` | **[fill in]** |
| 2 | `POST /predict` real photo + key | 200, label + confidence | **[fill in]** |
| 3 | `POST /predict` AI image + key | 200 | **[fill in]** |
| 4 | `POST /predict` without key | 401 | **[fill in]** |
| 5 | `POST /predict` a `.txt` file | 415 | **[fill in]** |

> **Screenshot 7.7** — Postman: `/predict` with the API key, response JSON and passing tests.  **Screenshot 7.8** — Postman: 401 without the key.  **Screenshot 7.9** — Postman Runner / Console with all five results.  **Screenshot 7.10** — `docker compose logs api` on the VM showing one log line per request.

**What went wrong / trade-offs.** The API is plain HTTP, so the API key travels unencrypted; for a real product I would put Caddy or Nginx with a TLS certificate in front (e.g. a free `*.sslip.io` hostname with Let's Encrypt). The 1 GB t3.micro needs swap for TensorFlow; a 2 GB instance would build and start faster. Oracle's free capacity being unavailable cost me an evening — for a real deadline a paid or credit-backed VM is the more reliable choice.

# Task 8 — Integrating Flask and the API

The web app has two interchangeable back-ends behind one interface (`app/predictors.py`): `LocalPredictor` loads the Keras model in-process (used in Docker Compose) and `RemotePredictor` forwards the upload to the cloud API with the `X-API-Key` header and a timeout. Setting

```bash
INFERENCE_MODE=remote
MODEL_API_URL=http://3.26.70.239:8000
MODEL_API_KEY=<same key as on the VM>
python -m app            # Flask on http://127.0.0.1:5000, debug off
```

switches the running app to the cloud model without any code change. At start-up the app calls the API's `/health` to learn the model name, version and threshold. For every upload the Flask view validates the file locally (rejecting bad files before they cost a network round trip), sends the bytes to `POST /predict`, stores the API's answer in the local database with `inference_mode = remote` and the measured round-trip latency, and redirects to the result page. If the API is down or the key is wrong, the user sees a friendly "Model service error" page (502) instead of a crash, and nothing is written to the history.

Table 8.1 — Inputs sent through the local Flask app to the cloud model (fill in from the result pages)

| # | Image (file) | True class | Verdict | Confidence | P(AI) | Model version | Round-trip latency |
|---|---|---|---|---|---|---|---|
| 1 | | Real | | | | | |
| 2 | | Real | | | | | |
| 3 | | Real | | | | | |
| 4 | | AI | | | | | |
| 5 | | AI | | | | | |
| 6 | | AI | | | | | |

> **Screenshots 8.1–8.6** — Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*.
> **Screenshot 8.7** — History page listing those requests with model version and mode `remote`.
> **Screenshot 8.8** — `/health` of the local app in remote mode (`"inference_mode": "remote"`, `api_url`), and the EC2 instance's API logs for the same requests.
