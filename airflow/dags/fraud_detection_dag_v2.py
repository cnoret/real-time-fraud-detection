"""
Fraud Detection MVP (Airflow 3)
Graph-friendly DAG with clear task bubbles:
- Ingestion: fetch a transaction from our own Transactions API
- Labeling: fetch its ground truth (is_fraud), served separately
- Inference: prepare payload + call the model API
- Persistence: store result into NeonDB
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

import requests
import psycopg2

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator

# ── Config ────────────────────────────────────────────────────────────────────
# Self-hosted transactions source (transactions-api/ in this repo, deployed on Render).
# Can be overridden with the Airflow Variable TRANSACTIONS_API.
TRANSACTIONS_API_DEFAULT = "https://fraud-transactions-api.onrender.com"
# Fraud detection model (model-api/ in this repo, deployed on Render).
# Can be overridden with the Airflow Variable MODEL_API.
MODEL_API_DEFAULT = "https://fraud-model-api.onrender.com"
# Free Render services sleep after 15 min idle and take about a minute to wake up.
REQUEST_TIMEOUT = 90

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

DOC_MD = """
# Fraud Detection MVP

**Flow**
1. **Ingestion** – fetch a live transaction from our Transactions API.
2. **Labeling** – fetch its ground truth (`is_fraud`), never sent to the model.
3. **Inference** – prepare payload (schema align + timestamp fix) then call the **model API**.
4. **Persistence** – store results into **NeonDB**.

**Key Vars**
- `TRANSACTIONS_API`, `MODEL_API`, `NEON_DB_URI` (Airflow Variables)

**Table**
```sql
CREATE TABLE IF NOT EXISTS fraud_transactions (
  transaction_id TEXT PRIMARY KEY,
  amount         DOUBLE PRECISION,
  merchant       TEXT,
  fraud_probability DOUBLE PRECISION,
  prediction     INTEGER,
  is_fraud       INTEGER,          -- ground truth, NULL if unknown
  event_time     TIMESTAMPTZ,      -- emission time from the source API
  inserted_at    TIMESTAMPTZ DEFAULT NOW()
);
```
"""


# ── Module-level helpers (pure Python, fully testable) ────────────────────────

def _parse_transactions_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the first transaction from a `/v1/transactions` response."""
    txs = body.get("transactions") if isinstance(body, dict) else None
    if not txs:
        raise ValueError("Invalid Transactions API response: no transaction")
    tx = txs[0]
    if not tx.get("trans_num"):
        raise ValueError("Invalid Transactions API response: missing trans_num")
    if "is_fraud" in tx:
        raise ValueError("Label leaked into transaction payload")
    return tx


def _parse_label_response(body: Dict[str, Any], trans_num: str) -> int | None:
    """Return is_fraud for trans_num from a `/v1/labels` response, or None if unknown."""
    for item in body.get("labels", []):
        if item.get("trans_num") == trans_num:
            return int(item["is_fraud"])
    return None


def _fix_timestamps(
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


def _build_payload(
    transaction: Dict[str, Any], expected_columns: List[str]
) -> Dict[str, Any]:
    """Align transaction fields with model schema, filling defaults for missing fields."""
    import time

    payload: Dict[str, Any] = {}
    for col in expected_columns:
        if col in transaction and transaction[col] is not None:
            payload[col] = transaction[col]
        elif col in DEFAULTS:
            payload[col] = DEFAULTS[col]
        else:
            payload[col] = f"default_{int(time.time())}" if col == "trans_num" else 0
    return payload


# ── DAG ───────────────────────────────────────────────────────────────────────

@dag(
    dag_id="automatic_fraud_detection",
    description="Fraud detection pipeline (fetch → prepare → predict + label → store)",
    start_date=datetime(2025, 1, 1),
    schedule=timedelta(minutes=1),
    catchup=False,
    default_args={
        "owner": "christophe_noret",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["fraud", "mvp", "graph"],
    doc_md=DOC_MD,
)
def automatic_fraud_detection():
    """Orchestrate fetch → prepare → predict → store pipeline."""
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    with TaskGroup(group_id="ingestion", tooltip="Fetch a transaction from the Transactions API") as ingestion:

        @task(task_id="fetch_transaction")
        def fetch_transaction() -> Dict[str, Any]:
            """Fetch 1 transaction (without its label) from our Transactions API."""
            base = Variable.get("TRANSACTIONS_API", default_var=TRANSACTIONS_API_DEFAULT)
            logging.info("Fetching transaction from %s…", base)
            r = requests.get(f"{base.rstrip('/')}/v1/transactions", params={"n": 1}, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            tx = _parse_transactions_response(r.json())
            logging.info("Fetched transaction: $%s at %s", tx.get("amt"), tx.get("merchant"))
            return tx

        tx_dict = fetch_transaction()

    # ── Labeling ──────────────────────────────────────────────────────────────
    with TaskGroup(group_id="labeling", tooltip="Fetch the ground truth label") as labeling:

        @task(task_id="fetch_ground_truth")
        def fetch_ground_truth(transaction: Dict[str, Any]) -> int | None:
            """Fetch is_fraud for this transaction (simulated chargeback feedback).

            A failure here must not block the prediction: the label stays NULL.
            """
            base = Variable.get("TRANSACTIONS_API", default_var=TRANSACTIONS_API_DEFAULT)
            trans_num = transaction["trans_num"]
            try:
                r = requests.get(
                    f"{base.rstrip('/')}/v1/labels",
                    params={"trans_num": trans_num},
                    timeout=30,
                )
                r.raise_for_status()
                return _parse_label_response(r.json(), trans_num)
            except Exception as e:
                logging.warning("Label fetch failed (%s); storing NULL.", e)
                return None

        label = fetch_ground_truth(tx_dict)

    # ── Inference ─────────────────────────────────────────────────────────────
    with TaskGroup(group_id="inference", tooltip="Prepare payload and predict") as inference:

        @task(task_id="prepare_payload")
        def prepare_payload(transaction: Dict[str, Any]) -> Dict[str, Any]:
            """Align fields with model schema; fix timestamps; fill defaults."""
            base = Variable.get("MODEL_API", default_var=MODEL_API_DEFAULT)
            try:
                resp = requests.get(f"{base.rstrip('/')}/health", timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                health = resp.json()
                expected_columns = (
                    health.get("expected_numeric", []) + health.get("expected_categorical", [])
                )
            except Exception as e:
                logging.warning("Health check failed (%s). Using fallback schema.", e)
                expected_columns = FALLBACK_SCHEMA

            tx = _fix_timestamps(transaction, expected_columns)
            return _build_payload(tx, expected_columns)

        @task(task_id="predict_fraud")
        def predict_fraud(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Call the model API and return prediction + prob."""
            base = Variable.get("MODEL_API", default_var=MODEL_API_DEFAULT)
            logging.info("Calling model endpoint %s…", base)
            resp = requests.post(
                f"{base.rstrip('/')}/predict", json={"data": payload}, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            pred = resp.json()
            result = {
                "probability": float(pred.get("probability", 0.0)),
                "prediction": int(pred.get("prediction", 0)),
            }
            logging.info(
                "Model → prob=%.6f, pred=%s", result["probability"], result["prediction"]
            )
            return result

        prepared = prepare_payload(tx_dict)
        pred_out = predict_fraud(prepared)

    # ── Persistence ───────────────────────────────────────────────────────────
    with TaskGroup(group_id="persistence", tooltip="Store results into NeonDB") as persistence:

        @task(task_id="store_in_neon")
        def store_in_neon(
            transaction: Dict[str, Any], pred: Dict[str, Any], is_fraud: int | None
        ) -> None:
            neon_db_uri = Variable.get("NEON_DB_URI")
            logging.info("Storing into NeonDB…")
            conn = psycopg2.connect(neon_db_uri)
            cur = conn.cursor()

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS fraud_transactions (
                  transaction_id TEXT,
                  amount DOUBLE PRECISION,
                  merchant TEXT,
                  fraud_probability DOUBLE PRECISION,
                  prediction INTEGER
                );
                """
            )
            cur.execute(
                """
                ALTER TABLE fraud_transactions
                  ADD COLUMN IF NOT EXISTS inserted_at TIMESTAMPTZ DEFAULT NOW();
                """
            )
            cur.execute(
                """
                ALTER TABLE fraud_transactions
                  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ,
                  ADD COLUMN IF NOT EXISTS is_fraud INTEGER,
                  ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ;
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS fraud_transactions_transaction_id_uidx
                ON fraud_transactions (transaction_id);
                """
            )
            cur.execute(
                """
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
                """,
                (
                    transaction.get("trans_num") or f"tx_{int(datetime.now().timestamp())}",
                    float(transaction.get("amt", 0)),
                    str(transaction.get("merchant", "Unknown"))[:255],
                    float(pred.get("probability", 0.0)),
                    int(pred.get("prediction", 0)),
                    is_fraud,
                    transaction.get("event_time"),
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            if float(pred.get("probability", 0.0)) > 0.001:
                logging.warning(
                    "FRAUD ALERT: prob=%.6f, amount=$%s",
                    float(pred["probability"]),
                    transaction.get("amt", 0),
                )

        store_in_neon(tx_dict, pred_out, label)

    start >> ingestion >> [inference, labeling] >> persistence >> end


dag = automatic_fraud_detection()
