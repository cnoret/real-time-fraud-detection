"""Build the replay set served by the Transactions API.

Source: fraudTest.csv (simulated "Sparkov" dataset, 555,719 transactions, 0.39% fraud).
A fixed sample (seed 42) of 50,000 rows keeps the natural fraud rate. The model
is trained on fraudTrain.csv only, which ends before fraudTest.csv starts, so these
rows are unseen data; training/train.py checks that none of their trans_num is in
its training data.

Usage:
    python transactions-api/scripts/build_holdout.py [path_or_url_of_the_csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
)
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "transactions_holdout.parquet"
SAMPLE_SIZE = 50_000
SEED = 42


def build(source: str = SOURCE_URL, output: Path = OUTPUT) -> pd.DataFrame:
    df = pd.read_csv(source, index_col=0)
    holdout = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_parquet(output, index=False, compression="zstd")
    print(
        f"{len(holdout)} rows, {int(holdout.is_fraud.sum())} frauds "
        f"({holdout.is_fraud.mean():.2%}) -> {output}"
    )
    return holdout


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else SOURCE_URL)
