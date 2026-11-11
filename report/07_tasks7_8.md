# Task 7 — API Deployment (Oracle Cloud, pathway A)

**What runs where.** As the task specifies, the **model API** is deployed in the cloud and the **Flask web app runs on my laptop** and calls it over the internet. The API (`api/__init__.py`) is a small Flask service served by Gunicorn in the same Docker image as the web app (`APP_MODULE=api.wsgi:app`), with the champion model mounted read-only. It is stateless — the history database stays with the web app.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /` | none | service info: model name, registry version, threshold, input size |
| `GET /health` | none | `status`, model info, uptime, requests served (200 / 503) |
| `POST /predict` | `X-API-Key` header | multipart field `image` → `label`, `is_ai`, `confidence`, `p_ai`, `threshold`, `model_name`, `model_version`, `latency_ms`, sanitised `filename`, `sha256`, size |

The key is compared in constant time (`hmac.compare_digest`) and comes from the VM's `.env`; without it the API answers 401. The same upload validation as the web app (type 415, size 413, undecodable 422) applies.

## Research before deploying

- **Current free limits (checked in November 2026).** Oracle's documentation now lists Ampere A1 Always Free as **2 OCPUs and 12 GB RAM** in total (1,500 OCPU-hours and 9,000 GB-hours per month) — halved from 4 OCPU / 24 GB in June 2026. Also free: up to two **E2.1.Micro** x86 VMs (1/8 OCPU, 1 GB RAM, 50 Mbps), **200 GB** block storage (boot volumes included) and **10 TB/month** outbound traffic. My console's *Limits, Quotas and Usage* page confirmed the numbers for my tenancy (Screenshot 7.1).
- **Arm vs x86.** A1 is Arm64 (Ampere Altra); E2.1.Micro is x86-64. Free A1 capacity was not available in my region (India West, Mumbai), so I used an **x86 VM.Standard.E2.1.Micro** (1 GB RAM) with **Ubuntu 22.04**. TensorFlow import alone needs ~600 MB, so `setup_vm.sh` adds a **2 GB swap file**, and the API runs a single Gunicorn worker. The Dockerfile handles both architectures (`TARGETARCH`: `tensorflow-cpu` on amd64, `tensorflow` on arm64), and the image is built on the VM itself.
- **Free-instance availability.** A1 capacity is often exhausted ("Out of host capacity"); the fix is to try another availability domain or retry later.
- **Idle-instance reclamation.** Oracle may reclaim an Always Free instance if, over 7 days, its 95th-percentile CPU, network and (for A1) memory utilisation are all below 20 %. A demo API is mostly idle, so I keep the instance only for the grading period (a cron job generating synthetic load would be against the spirit of the rule).
- **The two firewalls.** Traffic must pass (1) the **VCN Security List** (or a Network Security Group) in the OCI console — an ingress rule for TCP 8000 from 0.0.0.0/0 — *and* (2) the **VM's own iptables**, because Oracle's Ubuntu images ship rules that reject everything except SSH. Opening only one of them is the most common reason the API "does not respond".

## Step-by-step deployment

1. **Create the VM.** First *Networking → Virtual cloud networks → Start VCN Wizard → Create VCN with Internet Connectivity* (`23L-0517_A2_Ops-vcn`, which creates the public subnet, internet gateway and route table). Then *Compute → Instances → Create instance*: image *Canonical Ubuntu 22.04*, shape *VM.Standard.E2.1.Micro* (Always Free), existing VCN + **public** subnet, *assign public IPv4* on, my SSH public key. *(Screenshot 7.2)*
2. **Open port 8000 in the VCN.** Instance → Subnet → Default Security List → *Add Ingress Rule*: source `0.0.0.0/0`, TCP, destination port `8000`. *(Screenshot 7.3)*
3. **SSH in and prepare the VM.** `ssh ubuntu@141.148.203.241`, then `bash deploy/oracle/setup_vm.sh` (from the repo) installs Docker Engine + Compose, adds the user to the `docker` group, opens TCP 8000 in iptables (`iptables -I INPUT 6 … --dport 8000 -j ACCEPT` + `netfilter-persistent save`) and adds a 2 GB swap file. *(Screenshot 7.4)*
4. **Get the code and the model.** From my laptop: `rsync -av --exclude data --exclude mlruns --exclude .git … ./ ubuntu@141.148.203.241:~/23L-0517_A2_Ops/` — this copies the code and `models/champion.keras` (the registry's `@champion` version, exported once). The API never contacts DagsHub while serving.
5. **Secrets.** `cp .env.example .env` and set `MODEL_API_KEY` to a long random value (`python3 -c "import secrets;print(secrets.token_urlsafe(32))"`). `.env` stays on the VM only.
6. **Build and start.** `docker compose -f deploy/oracle/docker-compose.yml up -d --build`, then `docker compose -f deploy/oracle/docker-compose.yml ps` (healthy) and `logs` (`model loaded: visionguard v2`). *(Screenshot 7.5)*
7. **Test from the VM and from outside.** `curl http://localhost:8000/health` on the VM; `curl http://141.148.203.241:8000/health` from my laptop **on a different network (mobile hotspot)**. *(Screenshot 7.6)*

**API URL:** `http://141.148.203.241:8000` — health check `http://141.148.203.241:8000/health`, prediction `POST http://141.148.203.241:8000/predict`.

**Mapping of Compose elements to the provider.** Compose service → Docker on the VM; published port → VCN security-list rule + iptables rule; `.env` → file on the VM (could be OCI Vault); bind-mounted model → file on the VM's boot volume; restart policy `unless-stopped` → containers return after a VM reboot.

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

**What went wrong / trade-offs.** The API is plain HTTP, so the API key travels unencrypted; for a real product I would put Caddy or Nginx with a TLS certificate in front (e.g. a free `*.sslip.io` hostname with Let's Encrypt). The A1 instance is built natively for arm64, which is why the Dockerfile selects the TensorFlow package per architecture.

# Task 8 — Integrating Flask and the API

The web app has two interchangeable back-ends behind one interface (`app/predictors.py`): `LocalPredictor` loads the Keras model in-process (used in Docker Compose) and `RemotePredictor` forwards the upload to the cloud API with the `X-API-Key` header and a timeout. Setting

```bash
INFERENCE_MODE=remote
MODEL_API_URL=http://141.148.203.241:8000
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
> **Screenshot 8.8** — `/health` of the local app in remote mode (`"inference_mode": "remote"`, `api_url`), and the VM's API logs for the same requests.
