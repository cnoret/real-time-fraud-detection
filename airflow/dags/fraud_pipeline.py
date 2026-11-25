"""
Shared fraud scoring logic, independent of any orchestrator.

Used by both runners of the same pipeline:
- the Airflow DAG (`fraud_detection_dag_v2.py`), one transaction per task run;
- the scheduled GitHub Actions job (`scripts/score_batch.py`), a batch per run.

Everything here is plain Python (requests + psycopg2), so it can be unit tested
without Airflow and reused by any scheduler.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Sequence

import requests

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Self-hosted transactions source (transactions-api/ in this repo, deployed on Render).
TRANSACTIONS_API_DEFAULT = "https://fraud-transactions-api.onrender.com"
# Fraud detection model (model-api/ in this repo, deployed on Render).
MODEL_API_DEFAULT = "https://fraud-model-api.onrender.com"
# Free Render services sleep after 15 min idle and take about a minute to wake up.
REQUEST_TIMEOUT = 90
LABEL_TIMEOUT = 30
# The Transactions API serves at most 100 transactions or labels per call.
MAX_BATCH = 100
# Transactions per call to the model API's /predict/batch.
PREDICT_CHUNK = 20

DEFAULTS: Dict[str, Any] = {
    "amt": 100.0,
    "cc_num": 4000000000000002,
    "zip": 12345,
    "city_pop": 50000,
    "lat": 40.7128,
    "long": -74.0060,
    "merch_lat": 40.7128,
    "merch_long": -74.0060,
    "first": "John",
    "last": "Doe",
    "gender": "M",
    "street": "123 Main St",
    "city": "Anytown",
    "state": "NY",
    "job": "Engineer",
    "dob": "1990-01-01",
    "merchant": "Generic Store",
    "category": "misc_pos",
    "trans_date_trans_time": "2020-06-15 12:00:00",
    "unix_time": 1_592_222_400,
}

FALLBACK_SCHEMA: List[str] = [
    "cc_num", "amt", "zip", "lat", "long", "city_pop", "unix_time",
    "merch_lat", "merch_long", "trans_date_trans_time", "merchant",
    "category", "first", "last", "gender", "street", "city", "state",
    "job", "dob", "trans_num",
]

# Idempotent DDL: creates the table on a fresh database and migrates older ones.
SCHEMA_SQL: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS fraud_transactions (
      transaction_id TEXT,
      amount DOUBLE PRECISION,
      merchant TEXT,
      fraud_probability DOUBLE PRECISION,
      prediction INTEGER
    );
    """,
    """
    ALTER TABLE fraud_transactions
      ADD COLUMN IF NOT EXISTS inserted_at TIMESTAMPTZ DEFAULT NOW();
    """,
    """
    ALTER TABLE fraud_transactions
      ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ,
      ADD COLUMN IF NOT EXISTS is_fraud INTEGER,
      ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ;
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS fraud_transactions_transaction_id_uidx
    ON fraud_transactions (transaction_id);
    """,
]

UPSERT_SQL = """
    INSERT INTO fraud_transactions (
        transaction_id, amount, merchant, fraud_probability, prediction,
        is_fraud, event_time
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (transaction_id) DO UPDATE SET
        amount = EXCLUDED.amount,
        merchant = EXCLUDED.merchant,
        fraud_probability = EXCLUDED.fraud_probability,
        prediction = EXCLUDED.prediction,
        is_fraud = EXCLUDED.is_fraud,
        event_time = EXCLUDED.event_time,
        updated_at = NOW();
"""

FRAUD_ALERT_PROBABILITY = 0.001


# ── Pure helpers ──────────────────────────────────────────────────────────────

def parse_transactions_batch(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a `/v1/transactions` response and return all its transactions."""
    txs = body.get("transactions") if isinstance(body, dict) else None
    if not txs:
        raise ValueError("Invalid Transactions API response: no transaction")
    for tx in txs:
        if not tx.get("trans_num"):
            raise ValueError("Invalid Transactions API response: missing trans_num")
        if "is_fraud" in tx:
            raise ValueError("Label leaked into transaction payload")
    return txs


def parse_transactions_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the first transaction from a `/v1/transactions` response."""
    return parse_transactions_batch(body)[0]


def parse_labels_response(body: Dict[str, Any]) -> Dict[str, int]:
    """Map trans_num -> is_fraud from a `/v1/labels` response (unknown ids are absent)."""
    return {
        item["trans_num"]: int(item["is_fraud"])
        for item in body.get("labels", [])
        if item.get("trans_num")
    }


def parse_label_response(body: Dict[str, Any], trans_num: str) -> int | None:
    """Return is_fraud for trans_num from a `/v1/labels` response, or None if unknown."""
    return parse_labels_response(body).get(trans_num)


def expected_columns_from_health(health: Dict[str, Any]) -> List[str]:
    """Model input columns as advertised by the model API `/health` endpoint."""
    cols = list(health.get("expected_numeric", [])) + list(health.get("expected_categorical", []))
    return cols or list(FALLBACK_SCHEMA)


def fix_timestamps(
    transaction: Dict[str, Any], expected_columns: List[str]
) -> Dict[str, Any]:
    """Normalize timestamps into the model's 2020 training range."""
    tx = dict(transaction)

    if "current_time" in tx and "unix_time" in expected_columns:
        raw_ms = tx["current_time"]
        if raw_ms > 1_640_995_200_000:  # > 2022-01-01 in ms → rescale into 2020
            training_start = 1_577_836_800  # 2020-01-01 (s)
            year_seconds = 31_536_000
            tx["unix_time"] = training_start + (raw_ms // 1000) % year_seconds
        else:
            tx["unix_time"] = raw_ms // 1000

    if "unix_time" in tx and "trans_date_trans_time" in expected_columns:
        try:
            dt = datetime.fromtimestamp(tx["unix_time"])
            tx["trans_date_trans_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            tx["trans_date_trans_time"] = DEFAULTS["trans_date_trans_time"]

    return tx


def build_payload(
    transaction: Dict[str, Any], expected_columns: List[str]
) -> Dict[str, Any]:
    """Align transaction fields with model schema, filling defaults for missing fields."""
    payload: Dict[str, Any] = {}
    for col in expected_columns:
        if col in transaction and transaction[col] is not None:
            payload[col] = transaction[col]
        elif col in DEFAULTS:
            payload[col] = DEFAULTS[col]
        else:
            payload[col] = f"default_{int(time.time())}" if col == "trans_num" else 0
    return payload


def prepare_payload(
    transaction: Dict[str, Any], expected_columns: List[str]
) -> Dict[str, Any]:
    """Timestamp fix + schema alignment, as sent to the model."""
    return build_payload(fix_timestamps(transaction, expected_columns), expected_columns)


def parse_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what the pipeline stores from a model API prediction."""
    return {
        "probability": float(pred.get("probability", 0.0)),
        "prediction": int(pred.get("prediction", 0)),
    }


def build_row(
    transaction: Dict[str, Any], pred: Dict[str, Any], is_fraud: int | None
) -> tuple:
    """Values for UPSERT_SQL, in column order."""
    return (
        transaction.get("trans_num") or f"tx_{int(datetime.now().timestamp())}",
        float(transaction.get("amt", 0)),
        str(transaction.get("merchant", "Unknown"))[:255],
        float(pred.get("probability", 0.0)),
        int(pred.get("prediction", 0)),
        is_fraud,
        transaction.get("event_time"),
    )


# ── I/O: APIs ─────────────────────────────────────────────────────────────────

def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def fetch_transactions(base: str, n: int = 1) -> List[Dict[str, Any]]:
    """Fetch n transactions (without their label) from the Transactions API."""
    if not 1 <= n <= MAX_BATCH:
        raise ValueError(f"n must be between 1 and {MAX_BATCH}")
    r = requests.get(_url(base, "/v1/transactions"), params={"n": n}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return parse_transactions_batch(r.json())


def fetch_labels(base: str, trans_nums: Sequence[str]) -> Dict[str, int]:
    """Fetch the ground truth for the given ids (simulated chargeback feedback).

    A failure here must not block the predictions: missing labels stay NULL.
    """
    if not trans_nums:
        return {}
    try:
        r = requests.get(
            _url(base, "/v1/labels"),
            params={"trans_num": list(trans_nums)},
            timeout=LABEL_TIMEOUT,
        )
        r.raise_for_status()
        return parse_labels_response(r.json())
    except Exception as e:
        log.warning("Label fetch failed (%s); storing NULL.", e)
        return {}


def fetch_expected_columns(base: str) -> List[str]:
    """Ask the model API for its input schema, falling back to a known schema."""
    try:
        r = requests.get(_url(base, "/health"), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return expected_columns_from_health(r.json())
    except Exception as e:
        log.warning("Health check failed (%s). Using fallback schema.", e)
        return list(FALLBACK_SCHEMA)


def predict(base: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Score one prepared payload with the model API."""
    r = requests.post(_url(base, "/predict"), json={"data": payload}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return parse_prediction(r.json())


def predict_batch(
    base: str, payloads: Sequence[Dict[str, Any]], chunk_size: int = PREDICT_CHUNK
) -> List[Dict[str, Any]]:
    """Score prepared payloads with the model API, a chunk per call.

    Small chunks keep every call well under REQUEST_TIMEOUT on a free-tier CPU.
    """
    preds: List[Dict[str, Any]] = []
    for start in range(0, len(payloads), chunk_size):
        chunk = list(payloads[start:start + chunk_size])
        r = requests.post(
            _url(base, "/predict/batch"),
            json=[{"data": p} for p in chunk],
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        got = r.json().get("predictions", [])
        if len(got) != len(chunk):
            raise ValueError(f"Model API returned {len(got)} predictions for {len(chunk)} inputs")
        preds.extend(parse_prediction(p) for p in got)
    return preds


def wait_until_healthy(base: str, attempts: int = 6, delay: float = 20.0) -> bool:
    """Wake a sleeping free-tier service up by polling `/health`."""
    for i in range(1, attempts + 1):
        try:
            r = requests.get(_url(base, "/health"), timeout=REQUEST_TIMEOUT)
            if r.ok:
                return True
            log.info("%s/health answered %s (attempt %d/%d)", base, r.status_code, i, attempts)
        except requests.RequestException as e:
            log.info("%s not reachable yet (%s, attempt %d/%d)", base, e, i, attempts)
        if i < attempts:
            time.sleep(delay)
    return False


# ── I/O: database ─────────────────────────────────────────────────────────────

def store_rows(conn, rows: Sequence[tuple]) -> int:
    """Create/migrate the table, then upsert rows on transaction_id in one transaction."""
    with conn.cursor() as cur:
        for stmt in SCHEMA_SQL:
            cur.execute(stmt)
        cur.executemany(UPSERT_SQL, list(rows))
    conn.commit()
    return len(rows)


def log_fraud_alerts(rows: Sequence[tuple]) -> None:
    for row in rows:
        if row[3] > FRAUD_ALERT_PROBABILITY:
            log.warning("FRAUD ALERT: prob=%.6f, amount=$%s", row[3], row[1])
