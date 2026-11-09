# syntax=docker/dockerfile:1
# VisionGuard image - used for the web app (Compose) and for the model API (Oracle Cloud).
# Order of instructions = layer-cache friendliness: things that change rarely (base image,
# system settings, Python dependencies) come first; our source code, which changes on
# every commit, is copied last so a code edit only rebuilds the final small layers.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/models/champion.keras

WORKDIR /app

# 1) dependencies (cached until requirements-app.txt changes)
COPY requirements-app.txt .
# CPU-only TensorFlow: no GPU on the target machines, and tensorflow-cpu skips the
# CUDA libraries. On ARM (Oracle Ampere, Apple Silicon) the regular wheel is CPU-only.
ARG TARGETARCH
RUN pip install -r requirements-app.txt \
 && if [ "$TARGETARCH" = "amd64" ] || [ -z "$TARGETARCH" ]; then pip install tensorflow-cpu==2.19.0 keras==3.15.1; \
    else pip install tensorflow==2.19.0 keras==3.15.1; fi \
 && find /usr/local/lib/python3.11 -name "__pycache__" -prune -exec rm -rf {} + \
 && find /usr/local/lib/python3.11/site-packages -name "tests" -type d -prune -exec rm -rf {} +

# 2) unprivileged user - a compromised process cannot modify the image or the host mounts
RUN useradd --create-home --uid 10001 vg && mkdir -p /app/instance && chown vg:vg /app/instance

# 3) source code (changes most often -> last)
COPY --chown=vg:vg visionguard/ visionguard/
COPY --chown=vg:vg app/ app/
COPY --chown=vg:vg api/ api/

USER vg
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Gunicorn = production WSGI server. One worker process (the model is ~10 MB but TF
# itself uses a few hundred MB per process) with 4 threads for concurrent requests.
ENV APP_MODULE=app.wsgi:app
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 120 --access-logfile - $APP_MODULE"]
