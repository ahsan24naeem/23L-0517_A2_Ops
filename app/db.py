"""Database trail: one row per prediction request.

Works with PostgreSQL (Docker Compose / cloud) and SQLite (local dev / tests)
through SQLAlchemy Core. Each row answers: what was decided, when, by which
model (name + version), and how confidently.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, func, select, text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("filename", String(255), nullable=False),
    Column("file_sha256", String(64), nullable=False, index=True),
    Column("file_size_bytes", Integer, nullable=False),
    Column("image_width", Integer),
    Column("image_height", Integer),
    Column("label", String(32), nullable=False),
    Column("is_ai", Boolean, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("p_ai", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_name", String(128), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("inference_mode", String(16), nullable=False),
    Column("latency_ms", Float),
)


def make_engine(url: str, retries: int = 10, delay: float = 2.0) -> Engine:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True, future=True)
    last = None
    for _ in range(retries):  # extra safety on top of the Compose health check
        try:
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
            metadata.create_all(engine)
            return engine
        except OperationalError as e:
            last = e
            time.sleep(delay)
    raise last  # type: ignore[misc]


def insert_prediction(engine: Engine, **row) -> int:
    row.setdefault("created_at", datetime.now(timezone.utc))
    with engine.begin() as c:
        res = c.execute(predictions.insert().values(**row))
        return int(res.inserted_primary_key[0])


def get_prediction(engine: Engine, pid: int) -> dict | None:
    with engine.connect() as c:
        r = c.execute(select(predictions).where(predictions.c.id == pid)).mappings().first()
        return dict(r) if r else None


def list_predictions(engine: Engine, limit: int = 50, offset: int = 0) -> list[dict]:
    q = select(predictions).order_by(predictions.c.id.desc()).limit(limit).offset(offset)
    with engine.connect() as c:
        return [dict(r) for r in c.execute(q).mappings()]


def stats(engine: Engine) -> dict:
    with engine.connect() as c:
        total = c.execute(select(func.count()).select_from(predictions)).scalar() or 0
        ai = c.execute(select(func.count()).where(predictions.c.is_ai.is_(True))).scalar() or 0
        avg = c.execute(select(func.avg(predictions.c.confidence))).scalar()
    return {"total": int(total), "ai": int(ai), "real": int(total - ai), "avg_confidence": float(avg or 0)}


def ping(engine: Engine) -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
