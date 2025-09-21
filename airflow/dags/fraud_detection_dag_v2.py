"""
Fraud Detection MVP (Airflow 3)
Graph-friendly DAG with clear task bubbles:
- Ingestion: fetch transaction from Jedha
- Inference: prepare payload + call HF Space
- Persistence: store result into NeonDB
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import requests
import psycopg2

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator

# Config
JEDHA_API = "https://charlestng-real-time-fraud-detection.hf.space/current-transactions"
HF_SPACE_API = "https://cnoret-fraud-detection-api.hf.space/predict"
#NEON_DB_URI = Variable.get("NEON_DB_URI")
NEON_DB_URI = "postgresql://neondb_owner:npg_3gTXEcYOI2bV@ep-steep-butterfly-aba5n020-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"
DEFAULTS = {
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

DOC_MD = """
# Fraud Detection MVP

**Flow**
1. **Ingestion** – fetch a live transaction from Jedha API.  
2. **Inference** – prepare payload (schema align + timestamp fix) then call **HF Space** for prediction.  
3. **Persistence** – store results into **NeonDB**.

**Key Vars**
- `JEDHA_API`, `HF_SPACE_API`, `HEALTH_URL`, `NEON_DB_URI` (Airflow Variables)

**Table**
```sql
CREATE TABLE IF NOT EXISTS fraud_transactions (
  transaction_id TEXT PRIMARY KEY,
  amount         DOUBLE PRECISION,
  merchant       TEXT,
  fraud_probability DOUBLE PRECISION,
  prediction     INTEGER,
  inserted_at    TIMESTAMPTZ DEFAULT NOW()
);
```
"""


@dag(
    dag_id="automatic_fraud_detection",
    description="Fraud detection pipeline (fetch → prepare → predict → store)",
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
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # ------------- #
    # Ingestion     #
    # ------------- #
    with TaskGroup(
        group_id="ingestion", tooltip="Fetch a transaction from Jedha API"
    ) as ingestion:

        @task(task_id="fetch_transaction")
        def fetch_transaction() -> Dict[str, Any]:
            """Fetch 1 transaction from Jedha (Pandas orient=split)."""
            logging.info("Fetching transaction from Jedha…")
            r = requests.get(JEDHA_API, timeout=30)
            r.raise_for_status()
            raw_text = r.text.strip().strip('"').replace('\\"', '"')
            data = json.loads(raw_text)

            if "data" in data and "columns" in data:
                import pandas as pd

                df = pd.DataFrame(data["data"], columns=data["columns"])
                tx = df.iloc[0].to_dict()
                logging.info(
                    "Fetched transaction: $%s at %s", tx.get("amt"), tx.get("merchant")
                )
                return tx
            raise ValueError("Invalid Jedha API response format")

        tx_dict = fetch_transaction()

    # ------------- #
    # Inference     #
    # ------------- #
    with TaskGroup(
        group_id="inference", tooltip="Prepare payload and predict"
    ) as inference:

        @task(task_id="prepare_payload")
        def prepare_payload(transaction: Dict[str, Any]) -> Dict[str, Any]:
            """Align fields with model schema; fix timestamps; fill defaults."""
            # Get expected columns from model's /health
            try:
                resp = requests.get(HEALTH_URL, timeout=10)
                resp.raise_for_status()
                health = resp.json()
                expected_columns = health.get("expected_numeric", []) + health.get(
                    "expected_categorical", []
                )
            except Exception as e:
                logging.warning("Health check failed (%s). Using fallback schema.", e)
                expected_columns = [
                    "cc_num",
                    "amt",
                    "zip",
                    "lat",
                    "long",
                    "city_pop",
                    "unix_time",
                    "merch_lat",
                    "merch_long",
                    "trans_date_trans_time",
                    "merchant",
                    "category",
                    "first",
                    "last",
                    "gender",
                    "street",
                    "city",
                    "state",
                    "job",
                    "dob",
                    "trans_num",
                ]

            # Timestamp normalization (mirror of local client)
            if "current_time" in transaction and "unix_time" in expected_columns:
                raw_ms = transaction["current_time"]
                if (
                    raw_ms > 1_640_995_200_000
                ):  # > 2022-01-01 in ms → rescale into 2020 range
                    training_start = 1_577_836_800  # 2020-01-01 (s)
                    year_seconds = 31_536_000
                    transaction["unix_time"] = (
                        training_start + (raw_ms // 1000) % year_seconds
                    )
                else:
                    transaction["unix_time"] = raw_ms // 1000

            if (
                "unix_time" in transaction
                and "trans_date_trans_time" in expected_columns
            ):
                try:
                    dt = datetime.fromtimestamp(transaction["unix_time"])
                    transaction["trans_date_trans_time"] = dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except Exception:
                    transaction["trans_date_trans_time"] = DEFAULTS[
                        "trans_date_trans_time"
                    ]

            payload = {}
            for col in expected_columns:
                if col in transaction and transaction[col] is not None:
                    payload[col] = transaction[col]
                elif col in DEFAULTS:
                    payload[col] = DEFAULTS[col]
                else:
                    import time

                    payload[col] = (
                        f"default_{int(time.time())}" if col == "trans_num" else 0
                    )
            return payload

        @task(task_id="predict_fraud")
        def predict_fraud(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Call HF Space model endpoint and return prediction + prob."""
            logging.info("Calling model endpoint…")
            resp = requests.post(HF_SPACE_API, json={"data": payload}, timeout=30)
            resp.raise_for_status()
            pred = resp.json()
            result = {
                "probability": float(pred.get("probability", 0.0)),
                "prediction": int(pred.get("prediction", 0)),
            }
            logging.info(
                "Model → prob=%.6f, pred=%s",
                result["probability"],
                result["prediction"],
            )
            return result

        prepared = prepare_payload(tx_dict)
        pred_out = predict_fraud(prepared)

    # ------------- #
    # Persistence   #
    # ------------- #
    with TaskGroup(
        group_id="persistence", tooltip="Store results into NeonDB"
    ) as persistence:

        @task(task_id="store_in_neon")
        def store_in_neon(transaction: Dict[str, Any], pred: Dict[str, Any]) -> None:
            logging.info("Storing into NeonDB…")
            conn = psycopg2.connect(NEON_DB_URI)
            cur = conn.cursor()

            # 1) Create table if not exists (base columns)
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

            # 2) Ensure audit columns + uniqueness
            cur.execute(
                """
                ALTER TABLE fraud_transactions
                  ADD COLUMN IF NOT EXISTS inserted_at TIMESTAMPTZ DEFAULT NOW();
                """
            )
            cur.execute(
                """
                ALTER TABLE fraud_transactions
                  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
                """
            )
            # Ensure a unique index so ON CONFLICT works even if table was created earlier without PK
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS fraud_transactions_transaction_id_uidx
                ON fraud_transactions (transaction_id);
                """
            )

            # 3) Upsert with safe column set
            cur.execute(
                """
                INSERT INTO fraud_transactions (
                    transaction_id, amount, merchant, fraud_probability, prediction
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO UPDATE SET
                    amount = EXCLUDED.amount,
                    merchant = EXCLUDED.merchant,
                    fraud_probability = EXCLUDED.fraud_probability,
                    prediction = EXCLUDED.prediction,
                    updated_at = NOW();
                """,
                (
                    transaction.get("trans_num")
                    or f"tx_{int(datetime.now().timestamp())}",
                    float(transaction.get("amt", 0)),
                    str(transaction.get("merchant", "Unknown"))[:255],
                    float(pred.get("probability", 0.0)),
                    int(pred.get("prediction", 0)),
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            # Simple alert threshold
            if float(pred.get("probability", 0.0)) > 0.001:
                logging.warning(
                    "FRAUD ALERT: prob=%.6f, amount=$%s",
                    float(pred["probability"]),
                    transaction.get("amt", 0),
                )

        store_in_neon(tx_dict, pred_out)

    # Chain
    start >> ingestion >> inference >> persistence >> end


dag = automatic_fraud_detection()
