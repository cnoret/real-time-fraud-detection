"""Construit le jeu de rejeu servi par l'API de transactions.

Source : fraudTest.csv (jeu simulé « Sparkov », 555 719 transactions, 0,39 % de fraudes).
On tire un échantillon fixe (graine 42) de 50 000 lignes qui conserve le taux
de fraude naturel. Ces lignes forment un jeu « holdout » : tout futur
réentraînement du modèle doit exclure leurs trans_num pour que la mesure de
performance en production reste honnête.

Usage :
    python transactions-api/scripts/build_holdout.py [chemin_ou_url_du_csv]
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
        f"{len(holdout)} lignes, {int(holdout.is_fraud.sum())} fraudes "
        f"({holdout.is_fraud.mean():.2%}) -> {output}"
    )
    return holdout


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else SOURCE_URL)
