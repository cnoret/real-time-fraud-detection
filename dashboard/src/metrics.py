"""Pure metric helpers for the monitoring dashboard (pandas only, no Streamlit, no I/O).

Every function takes a DataFrame shaped like `fraud_transactions`:
transaction_id, amount, merchant, fraud_probability, prediction, is_fraud,
event_time, inserted_at (timestamps tz-aware, UTC).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

# Offline results on the notebook's test set (EDA_training.ipynb).
OFFLINE_METRICS = {"precision": 0.45, "recall": 0.59, "roc_auc": 0.967}

# Threshold applied by the pipeline when it stores `prediction` ("balanced" in model-api).
SERVED_THRESHOLD = 0.1

# The GitHub Actions pipeline runs hourly; GitHub may delay a scheduled run.
EXPECTED_RUN_EVERY = timedelta(hours=1)


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def labelled(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        frauds = self.tp + self.fn
        return self.tp / frauds if frauds else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return None if p is None or r is None else 0.0
        return 2 * p * r / (p + r)


def predicted(df: pd.DataFrame, threshold: float | None = None) -> pd.Series:
    """Model decision per row: the stored `prediction`, or `fraud_probability >= threshold`."""
    if threshold is None:
        return df["prediction"].fillna(0).astype(int)
    return (df["fraud_probability"] >= threshold).astype(int)


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose ground truth (`is_fraud`) is known."""
    return df[df["is_fraud"].notna()]


def confusion(df: pd.DataFrame, threshold: float | None = None) -> Confusion:
    """Confusion matrix on labelled rows only."""
    lab = labelled(df)
    if lab.empty:
        return Confusion(0, 0, 0, 0)
    pred = predicted(lab, threshold)
    truth = lab["is_fraud"].astype(int)
    return Confusion(
        tp=int(((pred == 1) & (truth == 1)).sum()),
        fp=int(((pred == 1) & (truth == 0)).sum()),
        fn=int(((pred == 0) & (truth == 1)).sum()),
        tn=int(((pred == 0) & (truth == 0)).sum()),
    )


def threshold_curve(df: pd.DataFrame, thresholds: list[float] | None = None) -> pd.DataFrame:
    """Precision, recall and F1 for each threshold (long format, for charts)."""
    if thresholds is None:
        thresholds = [round(0.01 * i, 2) for i in range(1, 100)]
    rows = []
    for t in thresholds:
        c = confusion(df, t)
        for metric, value in (("Precision", c.precision), ("Recall", c.recall), ("F1", c.f1)):
            if value is not None:
                rows.append({"threshold": t, "metric": metric, "value": value})
    return pd.DataFrame(rows, columns=["threshold", "metric", "value"])


def roc_auc(df: pd.DataFrame) -> float | None:
    """ROC AUC from ranks (Mann-Whitney U); None when one class is missing."""
    lab = labelled(df)
    truth = lab["is_fraud"].astype(int)
    n_pos, n_neg = int(truth.sum()), int((truth == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = lab["fraud_probability"].rank(method="average")
    return float((ranks[truth == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def label_coverage(df: pd.DataFrame) -> float | None:
    """Share of rows whose label is known."""
    if df.empty:
        return None
    return float(df["is_fraud"].notna().mean())


@dataclass(frozen=True)
class Freshness:
    level: str  # "ok" | "warn" | "down" | "empty"
    age: timedelta | None


def freshness(
    last_inserted: datetime | None, now: datetime, every: timedelta = EXPECTED_RUN_EVERY
) -> Freshness:
    """ok within 1.5 runs, warn within 3 runs, down beyond (or empty table)."""
    if last_inserted is None or pd.isna(last_inserted):
        return Freshness("empty", None)
    age = now - last_inserted
    if age <= every * 1.5:
        return Freshness("ok", age)
    if age <= every * 3:
        return Freshness("warn", age)
    return Freshness("down", age)


def format_age(age: timedelta | None) -> str:
    """Human-readable duration: '45 s', '12 min', '3 h 05 min', '4 d'."""
    if age is None:
        return "n/a"
    seconds = max(0, int(age.total_seconds()))
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    return f"{hours // 24} d"


def hourly_volume(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Rows scored, alerts raised and actual frauds per hour, empty hours included as 0."""
    hours = pd.date_range(
        pd.Timestamp(start).floor("h"), pd.Timestamp(end).floor("h"), freq="h"
    )
    out = pd.DataFrame({"hour": hours})
    if df.empty:
        return out.assign(scored=0, alerts=0, frauds=0)
    tmp = df.assign(
        hour=df["inserted_at"].dt.floor("h"),
        alert=predicted(df),
        fraud=df["is_fraud"].fillna(0).astype(int),
    )
    agg = tmp.groupby("hour").agg(
        scored=("transaction_id", "size"), alerts=("alert", "sum"), frauds=("fraud", "sum")
    )
    out = out.merge(agg, left_on="hour", right_index=True, how="left").fillna(0)
    return out.astype({"scored": int, "alerts": int, "frauds": int})


def batches(df: pd.DataFrame, gap: timedelta = timedelta(minutes=5)) -> pd.DataFrame:
    """Group rows into ingestion batches: a new batch starts after `gap` without inserts."""
    cols = ["started", "rows", "alerts", "frauds", "labelled", "duration_s"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    tmp = df.sort_values("inserted_at").assign(alert=lambda d: predicted(d))
    batch_id = (tmp["inserted_at"].diff() > gap).cumsum()
    out = tmp.groupby(batch_id).agg(
        started=("inserted_at", "min"),
        ended=("inserted_at", "max"),
        rows=("transaction_id", "size"),
        alerts=("alert", "sum"),
        frauds=("is_fraud", lambda s: int(s.fillna(0).sum())),
        labelled=("is_fraud", lambda s: int(s.notna().sum())),
    )
    out["duration_s"] = (out["ended"] - out["started"]).dt.total_seconds()
    return out[cols].sort_values("started", ascending=False).reset_index(drop=True)


def ingestion_lag(df: pd.DataFrame) -> timedelta | None:
    """Median delay between a transaction's emission (event_time) and its insert."""
    if df.empty or df["event_time"].isna().all():
        return None
    lag = (df["inserted_at"] - df["event_time"]).dropna()
    return None if lag.empty else lag.median().to_pytimedelta()


def outcome(df: pd.DataFrame) -> pd.Series:
    """Per-row outcome label: Caught, Missed, False alert, Passed, or Label pending."""
    pred = predicted(df)
    truth = df["is_fraud"]
    out = pd.Series("Label pending", index=df.index)
    known = truth.notna()
    t = truth.fillna(0).astype(int)
    out[known & (pred == 1) & (t == 1)] = "Caught"
    out[known & (pred == 0) & (t == 1)] = "Missed"
    out[known & (pred == 1) & (t == 0)] = "False alert"
    out[known & (pred == 0) & (t == 0)] = "Passed"
    return out
