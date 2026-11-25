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
from datetime import datetime, timezone
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

# Raw fields the model reads (model-api/fraud_features.py INPUT_COLUMNS), used when
# the model API /health is unreachable. Identifiers are deliberately absent.
FALLBACK_SCHEMA: List[str] = [
    "trans_date_trans_time", "amt", "category", "gender", "dob",
    "city_pop", "lat", "long", "merch_lat", "merch_long",
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
    ALTER TABLE fraud_transactions
      ADD COLUMN IF NOT EXISTS model_version TEXT,
      ADD COLUMN IF NOT EXISTS decision_threshold DOUBLE PRECISION;
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS fraud_transactions_transaction_id_uidx
    ON fraud_transactions (transaction_id);
    """,
]

UPSERT_SQL = """
    INSERT INTO fraud_transactions (
        transaction_id, amount, merchant, fraud_probability, prediction,
        is_fraud, event_time, model_version, decision_threshold
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (transaction_id) DO UPDATE SET
        amount = EXCLUDED.amount,
        merchant = EXCLUDED.merchant,
        fraud_probability = EXCLUDED.fraud_probability,
        prediction = EXCLUDED.prediction,
        is_fraud = EXCLUDED.is_fraud,
        event_time = EXCLUDED.event_time,
        model_version = EXCLUDED.model_version,
        decision_threshold = EXCLUDED.decision_threshold,
        updated_at = NOW();
"""



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
    cols = list(health.get("input_columns") or [])
    return cols or list(FALLBACK_SCHEMA)


def _utc_text(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fix_timestamps(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Make sure `trans_date_trans_time` is set, the only time field the model reads.

    The source's own value wins. Otherwise it is derived from `current_time`
    (epoch ms, Jedha-style payloads) or from `unix_time` (epoch s), in UTC.
    """
    tx = dict(transaction)
    if tx.get("trans_date_trans_time"):
        return tx
    try:
        if tx.get("current_time") is not None:
            tx["trans_date_trans_time"] = _utc_text(tx["current_time"] / 1000)
        elif tx.get("unix_time") is not None:
            tx["trans_date_trans_time"] = _utc_text(tx["unix_time"])
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    return tx


def build_payload(
    transaction: Dict[str, Any], expected_columns: List[str]
) -> Dict[str, Any]:
    """Keep only the model's input columns; a missing field is sent as null.

    No made-up defaults: the model handles missing values itself, and the
    API reports them in `details.missing_fields`.
    """
    return {col: transaction.get(col) for col in expected_columns}


def prepare_payload(
    transaction: Dict[str, Any], expected_columns: List[str]
) -> Dict[str, Any]:
    """Timestamp fix + schema alignment, as sent to the model."""
    return build_payload(fix_timestamps(transaction), expected_columns)


def parse_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what the pipeline stores from a model API prediction."""
    threshold = pred.get("threshold_used")
    return {
        "probability": float(pred.get("probability", 0.0)),
        "prediction": int(pred.get("prediction", 0)),
        "model_version": pred.get("model_version"),
        "threshold": None if threshold is None else float(threshold),
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
        pred.get("model_version"),
        pred.get("threshold"),
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


RETRY_STATUSES = {429, 502, 503, 504}


def _post_with_retries(url: str, body: Any, attempts: int = 3, delay: float = 30.0):
    """POST, retrying timeouts and gateway errors (free-tier restarts and redeploys)."""
    for i in range(1, attempts + 1):
        try:
            r = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
            if r.status_code not in RETRY_STATUSES or i == attempts:
                r.raise_for_status()
                return r
            log.warning("%s answered %s (attempt %d/%d), retrying", url, r.status_code, i, attempts)
        except (requests.ConnectionError, requests.Timeout) as e:
            if i == attempts:
                raise
            log.warning("%s failed (%s, attempt %d/%d), retrying", url, e, i, attempts)
        time.sleep(delay)


def predict_batch(
    base: str, payloads: Sequence[Dict[str, Any]], chunk_size: int = PREDICT_CHUNK
) -> List[Dict[str, Any]]:
    """Score prepared payloads with the model API, a chunk per call.

    Small chunks keep every call well under REQUEST_TIMEOUT on a free-tier CPU.
    """
    preds: List[Dict[str, Any]] = []
    for start in range(0, len(payloads), chunk_size):
        chunk = list(payloads[start:start + chunk_size])
        r = _post_with_retries(_url(base, "/predict/batch"), [{"data": p} for p in chunk])
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
    """Log the transactions the model flagged (prediction = 1)."""
    for row in rows:
        if row[4] == 1:
            log.warning("FRAUD ALERT: prob=%.6f, amount=$%s", row[3], row[1])
