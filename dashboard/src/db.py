"""Database access for the dashboard: one cached connection, cached queries (Neon PostgreSQL)."""

from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal

import pandas as pd
import psycopg2
import streamlit as st

from quality import CHECKS

TRANSACTION_COLUMNS = [
    "transaction_id", "amount", "merchant", "fraud_probability",
    "prediction", "is_fraud", "event_time", "inserted_at",
]
# Added by a later pipeline migration: selected only once they exist.
OPTIONAL_COLUMNS = ["model_version", "decision_threshold"]


def db_uri() -> str | None:
    return os.getenv("NEON_DB_URI")


@st.cache_resource(show_spinner=False)
def _connection():
    conn = psycopg2.connect(db_uri())
    conn.autocommit = True  # read-only queries: no transaction left open between reruns
    return conn


def _query(sql: str, params: tuple = ()) -> tuple[list[tuple], list[str]]:
    """Run a query, reconnecting once if Neon closed the idle connection."""
    for attempt in (1, 2):
        conn = _connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall(), [d[0] for d in cur.description]
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            _connection.clear()
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


def _to_float(value):
    return float(value) if isinstance(value, Decimal) else value


@st.cache_data(ttl=300, show_spinner=False)
def _existing_columns() -> set[str]:
    rows, _ = _query(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'fraud_transactions'"
    )
    return {r[0] for r in rows}


@st.cache_data(ttl=60, show_spinner=False)
def load_transactions(since: datetime | None) -> pd.DataFrame:
    """All rows inserted since `since` (or all rows), newest first."""
    existing = _existing_columns()
    cols = TRANSACTION_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in existing]
    sql = f"SELECT {', '.join(cols)} FROM fraud_transactions"
    params: tuple = ()
    if since is not None:
        sql += " WHERE inserted_at >= %s"
        params = (since,)
    rows, cols = _query(sql + " ORDER BY inserted_at DESC", params)
    df = pd.DataFrame(rows, columns=cols)
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    for col in ("amount", "fraud_probability", "decision_threshold"):
        df[col] = pd.to_numeric(df[col].map(_to_float), errors="coerce").astype(float)
    for col in ("prediction", "is_fraud"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("event_time", "inserted_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_overall() -> dict:
    """Table-wide figures, independent of the selected window."""
    rows, _ = _query(
        "SELECT COUNT(*), MIN(inserted_at), MAX(inserted_at), NOW() FROM fraud_transactions"
    )
    total, first, last, now = rows[0]
    return {
        "total": int(total),
        "first": pd.Timestamp(first).tz_convert("UTC") if first else None,
        "last": pd.Timestamp(last).tz_convert("UTC") if last else None,
        "now": pd.Timestamp(now).tz_convert("UTC"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def run_quality_checks() -> list[tuple[str, float | None]]:
    """(check name, measured value) for every check in quality.CHECKS."""
    results = []
    for check in CHECKS:
        try:
            rows, _ = _query(check.sql)
            value = rows[0][0]
            results.append((check.name, None if value is None else float(value)))
        except psycopg2.Error:
            results.append((check.name, None))
    return results
