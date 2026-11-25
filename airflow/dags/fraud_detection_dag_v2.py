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
from typing import Dict, Any

import psycopg2

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator

# The scoring logic lives in fraud_pipeline.py (same folder), shared with the
# scheduled GitHub Actions job (scripts/score_batch.py).
import fraud_pipeline as fp
from fraud_pipeline import DEFAULTS, FALLBACK_SCHEMA  # noqa: F401  (re-exported)

# Kept under their historical names for readability and tests.
_parse_transactions_response = fp.parse_transactions_response
_parse_label_response = fp.parse_label_response
_fix_timestamps = fp.fix_timestamps
_build_payload = fp.build_payload

# Defaults can be overridden with the Airflow Variables TRANSACTIONS_API / MODEL_API.
TRANSACTIONS_API_DEFAULT = fp.TRANSACTIONS_API_DEFAULT
MODEL_API_DEFAULT = fp.MODEL_API_DEFAULT

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
            tx = fp.fetch_transactions(base, n=1)[0]
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
            return fp.fetch_labels(base, [trans_num]).get(trans_num)

        label = fetch_ground_truth(tx_dict)

    # ── Inference ─────────────────────────────────────────────────────────────
    with TaskGroup(group_id="inference", tooltip="Prepare payload and predict") as inference:

        @task(task_id="prepare_payload")
        def prepare_payload(transaction: Dict[str, Any]) -> Dict[str, Any]:
            """Align fields with model schema; fix timestamps; fill defaults."""
            base = Variable.get("MODEL_API", default_var=MODEL_API_DEFAULT)
            expected_columns = fp.fetch_expected_columns(base)
            return fp.prepare_payload(transaction, expected_columns)

        @task(task_id="predict_fraud")
        def predict_fraud(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Call the model API and return prediction + prob."""
            base = Variable.get("MODEL_API", default_var=MODEL_API_DEFAULT)
            logging.info("Calling model endpoint %s…", base)
            result = fp.predict(base, payload)
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
            rows = [fp.build_row(transaction, pred, is_fraud)]
            conn = psycopg2.connect(neon_db_uri)
            try:
                fp.store_rows(conn, rows)
            finally:
                conn.close()
            fp.log_fraud_alerts(rows)

        store_in_neon(tx_dict, pred_out, label)

    start >> ingestion >> [inference, labeling] >> persistence >> end


dag = automatic_fraud_detection()
