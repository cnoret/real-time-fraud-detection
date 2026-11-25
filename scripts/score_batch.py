"""
Score a batch of transactions end to end, without Airflow.

Same logic as the Airflow DAG (airflow/dags/fraud_detection_dag_v2.py), reused
from airflow/dags/fraud_pipeline.py, but one batch per run so a scheduler that
fires rarely (GitHub Actions cron) still feeds the dashboard:

  wake APIs → fetch N transactions → fetch their labels → predict → upsert in Neon

Environment:
  NEON_DB_URI        required, PostgreSQL connection string
  TRANSACTIONS_API   optional, defaults to the Render deployment
  MODEL_API          optional, defaults to the Render deployment

Usage:
  python scripts/score_batch.py --n 30
  python scripts/score_batch.py --n 5 --dry-run   # score but write nothing (no NEON_DB_URI needed)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "airflow" / "dags"))

import fraud_pipeline as fp  # noqa: E402

log = logging.getLogger("score_batch")


def run(
    n: int, transactions_api: str, model_api: str, db_uri: str | None, connect=None
) -> int:
    """Run one batch; returns the number of rows written (0 when db_uri is None)."""
    for name, base in (("Transactions API", transactions_api), ("Model API", model_api)):
        if not fp.wait_until_healthy(base):
            raise RuntimeError(f"{name} ({base}) did not wake up")

    txs = fp.fetch_transactions(transactions_api, n=n)
    log.info("Fetched %d transactions", len(txs))

    labels = fp.fetch_labels(transactions_api, [tx["trans_num"] for tx in txs])
    log.info("Fetched %d labels (%d frauds)", len(labels), sum(labels.values()))

    expected_columns = fp.fetch_expected_columns(model_api)
    payloads = [fp.prepare_payload(tx, expected_columns) for tx in txs]
    preds = fp.predict_batch(model_api, payloads)
    log.info("Scored %d transactions (%d flagged)", len(preds), sum(p["prediction"] for p in preds))

    rows = [fp.build_row(tx, p, labels.get(tx["trans_num"])) for tx, p in zip(txs, preds)]

    if db_uri is None:
        for row in rows:
            log.info("dry run: %s", row)
        return 0

    if connect is None:
        import psycopg2

        connect = psycopg2.connect
    conn = connect(db_uri)
    try:
        written = fp.store_rows(conn, rows)
    finally:
        conn.close()
    log.info("Upserted %d rows into fraud_transactions", written)
    fp.log_fraud_alerts(rows)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=30, help=f"transactions per run (1-{fp.MAX_BATCH})")
    parser.add_argument("--dry-run", action="store_true", help="score but do not write to the database")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_uri = None if args.dry_run else os.environ.get("NEON_DB_URI")
    if not db_uri and not args.dry_run:
        log.error("NEON_DB_URI is not set")
        return 2

    run(
        n=args.n,
        transactions_api=os.environ.get("TRANSACTIONS_API") or fp.TRANSACTIONS_API_DEFAULT,
        model_api=os.environ.get("MODEL_API") or fp.MODEL_API_DEFAULT,
        db_uri=db_uri,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
