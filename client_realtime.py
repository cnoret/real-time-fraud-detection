"""Minimal interactive client: fetch a live transaction, score it, print the verdict.

Reuses the pipeline's own helpers (airflow/dags/fraud_pipeline.py), so it sends
the model exactly what the scheduled pipeline sends.

    python client_realtime.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "airflow" / "dags"))

import fraud_pipeline as fp  # noqa: E402

TRANSACTIONS_API = fp.TRANSACTIONS_API_DEFAULT
MODEL_API = fp.MODEL_API_DEFAULT

VERDICTS = {
    "CRITICAL": "FRAUD - block the payment",
    "HIGH": "FRAUD ALERT - send to review",
    "MEDIUM": "Suspicious - review if capacity allows",
    "LOW": "Legitimate - approve",
}


def score_one(expected_columns: list[str]) -> None:
    tx = fp.fetch_transactions(TRANSACTIONS_API, n=1)[0]
    payload = fp.prepare_payload(tx, expected_columns)
    r = requests.post(f"{MODEL_API}/predict", json={"data": payload}, timeout=fp.REQUEST_TIMEOUT)
    r.raise_for_status()
    result = r.json()
    truth = fp.fetch_labels(TRANSACTIONS_API, [tx["trans_num"]]).get(tx["trans_num"])

    print(f"\n{VERDICTS.get(result['risk_level'], result['risk_level'])}")
    print(f"  amount       ${tx.get('amt')} ({tx.get('category')})")
    print(f"  probability  {result['probability']:.4f} (alert threshold {result['threshold_used']:.2f})")
    print(f"  model        {result.get('model_version')}")
    print(f"  ground truth {'fraud' if truth == 1 else 'legitimate' if truth == 0 else 'unknown'}")


def main() -> None:
    print("Waking the APIs up (free hosting can take a minute)...")
    if not (fp.wait_until_healthy(TRANSACTIONS_API) and fp.wait_until_healthy(MODEL_API)):
        print("APIs unavailable")
        return
    expected_columns = fp.fetch_expected_columns(MODEL_API)
    while True:
        choice = input("\n1. Score one transaction  2. Score five  0. Exit\nChoice: ").strip()
        if choice == "0":
            return
        for i in range({"1": 1, "2": 5}.get(choice, 0)):
            score_one(expected_columns)
            if choice == "2" and i < 4:
                time.sleep(2)


if __name__ == "__main__":
    main()
