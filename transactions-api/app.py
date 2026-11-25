"""
Fraud Transactions API — a self-hosted stream of card payments.

Replays a fixed holdout sample of the simulated "Sparkov" dataset
(fraudTest.csv) as if the payments were happening right now.

Endpoints
- GET /health                 service status and dataset stats
- GET /v1/transactions?n=1    n random transactions, WITHOUT the label
- GET /v1/labels?trans_num=…  ground truth for given transactions
                              (simulates chargeback feedback arriving later)
- GET /current-transactions   drop-in replacement for the former Jedha API
                              (pandas orient=split, JSON-encoded string)

Labels are deliberately served apart from the transactions, so the label can
never leak into the features sent to the model.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

DATA_PATH = Path(
    os.getenv(
        "TRANSACTIONS_PATH",
        Path(__file__).parent / "data" / "transactions_holdout.parquet",
    )
)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MAX_BATCH = 100
LABEL_COLUMN = "is_fraud"
# Columns the former Jedha API did not expose
JEDHA_DROPPED = ["trans_date_trans_time", "unix_time", LABEL_COLUMN]


def load_transactions(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = {"trans_num", LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df.reset_index(drop=True)


TRANSACTIONS = load_transactions()
LABELS = TRANSACTIONS.set_index("trans_num")[LABEL_COLUMN].astype(int).to_dict()
FEATURES = TRANSACTIONS.drop(columns=[LABEL_COLUMN])
_rng = np.random.default_rng()
_rng_lock = threading.Lock()

app = FastAPI(
    title="Fraud Transactions API",
    description=__doc__,
    version="1.0.0",
)


# ── Minimal per-client rate limit (protects a free-tier host) ─────────────────
_hits: dict[str, list[float]] = defaultdict(list)
_hits_lock = threading.Lock()


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path not in ("/", "/health", "/docs", "/openapi.json"):
        client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        now = time.monotonic()
        with _hits_lock:
            recent = [t for t in _hits[client] if now - t < 60]
            if len(recent) >= RATE_LIMIT_PER_MINUTE:
                _hits[client] = recent
                return JSONResponse(
                    {"detail": f"Rate limit: {RATE_LIMIT_PER_MINUTE} requests/minute"},
                    status_code=429,
                )
            recent.append(now)
            _hits[client] = recent
    return await call_next(request)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sample(n: int, seed: int | None = None) -> pd.DataFrame:
    if seed is not None:
        idx = np.random.default_rng(seed).integers(0, len(FEATURES), size=n)
    else:
        with _rng_lock:
            idx = _rng.integers(0, len(FEATURES), size=n)
    return FEATURES.iloc[idx]


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    # Round-trip through pandas JSON to get native Python types
    return json.loads(df.to_json(orient="records"))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Fraud Transactions API — replaying card payments in real time",
        "docs": "/docs",
        "endpoints": ["/health", "/v1/transactions", "/v1/labels", "/current-transactions"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "transactions": len(TRANSACTIONS),
        "fraud_count": int(TRANSACTIONS[LABEL_COLUMN].sum()),
        "fraud_rate": round(float(TRANSACTIONS[LABEL_COLUMN].mean()), 5),
        "columns": list(FEATURES.columns),
    }


@app.get("/v1/transactions")
def transactions(
    n: int = Query(1, ge=1, le=MAX_BATCH, description="Number of transactions"),
    seed: int | None = Query(None, description="Seed for a reproducible draw"),
) -> dict[str, Any]:
    """Return n transactions as JSON records, stamped with the time they were emitted.

    The original fields (including `unix_time` and `trans_date_trans_time`,
    which the model was trained on) are kept as-is; `event_time` is the
    emission time, useful for freshness monitoring downstream.
    """
    event_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = _to_records(_sample(n, seed))
    for r in records:
        r["event_time"] = event_time
    return {"count": len(records), "transactions": records}


@app.get("/v1/labels")
def labels(
    trans_num: list[str] = Query(..., description="One or more transaction ids"),
) -> dict[str, Any]:
    """Return the ground truth (is_fraud) for known transaction ids.

    Unknown ids are listed in `unknown` instead of failing the whole call.
    """
    if len(trans_num) > MAX_BATCH:
        raise HTTPException(422, f"At most {MAX_BATCH} ids per call")
    found = [{"trans_num": t, LABEL_COLUMN: LABELS[t]} for t in trans_num if t in LABELS]
    unknown = [t for t in trans_num if t not in LABELS]
    return {"labels": found, "unknown": unknown}


@app.get("/current-transactions")
def current_transactions() -> str:
    """Drop-in replacement for the former Jedha endpoint.

    Same contract: one transaction, pandas `to_json(orient="split")`, returned
    as a JSON-encoded string, without timestamps nor label, plus `current_time`
    (epoch milliseconds).
    """
    df = _sample(1).drop(columns=[c for c in JEDHA_DROPPED if c in FEATURES.columns])
    df = df.assign(current_time=int(time.time() * 1000))
    return df.to_json(orient="split")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
