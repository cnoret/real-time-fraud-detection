"""Data quality checks on `fraud_transactions`, written in SQL and run by the dashboard.

Each check is one query returning a single number, compared with a bound.
`evaluate` is pure so the pass / warn / fail rules are unit-tested without a database.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    sql: str
    rule: str  # "eq" | "lte" | "gte": how the value compares with `bound`
    bound: float
    severity: str = "fail"  # status when the rule is broken: "fail" or "warn"
    unit: str = "rows"


CHECKS: list[Check] = [
    Check(
        name="Unique transaction ids",
        description="Each transaction is stored once (the pipeline upserts on transaction_id).",
        sql="SELECT COUNT(*) - COUNT(DISTINCT transaction_id) FROM fraud_transactions",
        rule="eq",
        bound=0,
        unit="duplicates",
    ),
    Check(
        name="Probability within [0, 1]",
        description="Every stored score is a valid probability.",
        sql=(
            "SELECT COUNT(*) FROM fraud_transactions "
            "WHERE fraud_probability IS NULL OR fraud_probability NOT BETWEEN 0 AND 1"
        ),
        rule="eq",
        bound=0,
    ),
    Check(
        name="Binary prediction",
        description="The model decision is always 0 or 1.",
        sql=(
            "SELECT COUNT(*) FROM fraud_transactions "
            "WHERE prediction IS NULL OR prediction NOT IN (0, 1)"
        ),
        rule="eq",
        bound=0,
    ),
    Check(
        name="Valid amount and merchant",
        description="No missing merchant, no missing or negative amount.",
        sql=(
            "SELECT COUNT(*) FROM fraud_transactions "
            "WHERE amount IS NULL OR amount < 0 OR merchant IS NULL OR merchant = ''"
        ),
        rule="eq",
        bound=0,
    ),
    Check(
        name="Label coverage, last 7 days",
        description="Share of recent rows whose ground truth (is_fraud) is known.",
        sql=(
            "SELECT COALESCE(100.0 * COUNT(is_fraud) / NULLIF(COUNT(*), 0), 100) "
            "FROM fraud_transactions WHERE inserted_at > NOW() - INTERVAL '7 days'"
        ),
        rule="gte",
        bound=95,
        severity="warn",
        unit="%",
    ),
    Check(
        name="Fresh data",
        description="Minutes since the latest insert (the pipeline runs hourly).",
        sql="SELECT EXTRACT(EPOCH FROM NOW() - MAX(inserted_at)) / 60 FROM fraud_transactions",
        rule="lte",
        bound=90,
        unit="min",
    ),
    Check(
        name="Hourly runs, last 24 h",
        description="Distinct hours with at least one insert (GitHub may skip a busy slot).",
        sql=(
            "SELECT COUNT(DISTINCT date_trunc('hour', inserted_at)) FROM fraud_transactions "
            "WHERE inserted_at > NOW() - INTERVAL '24 hours'"
        ),
        rule="gte",
        bound=22,
        severity="warn",
        unit="hours",
    ),
    Check(
        name="Model version recorded, last 24 h",
        description="Every recent row names the model release that scored it.",
        sql=(
            "SELECT COUNT(*) FROM fraud_transactions "
            "WHERE inserted_at > NOW() - INTERVAL '24 hours' AND model_version IS NULL"
        ),
        rule="eq",
        bound=0,
        severity="warn",
    ),
    Check(
        name="No event from the future",
        description="No transaction is emitted after it was stored.",
        sql=(
            "SELECT COUNT(*) FROM fraud_transactions "
            "WHERE event_time > inserted_at + INTERVAL '5 minutes'"
        ),
        rule="eq",
        bound=0,
        severity="warn",
    ),
]


def evaluate(check: Check, value: float | None) -> str:
    """Return "pass", "warn" or "fail" for a check's measured value."""
    if value is None:
        return check.severity
    value = float(value)
    if check.rule == "eq":
        ok = value == check.bound
    elif check.rule == "lte":
        ok = value <= check.bound
    elif check.rule == "gte":
        ok = value >= check.bound
    else:
        raise ValueError(f"Unknown rule: {check.rule}")
    return "pass" if ok else check.severity


def format_value(check: Check, value: float | None) -> str:
    if value is None:
        return "no data"
    value = float(value)
    shown = f"{value:.1f}" if check.unit in ("%", "min") else f"{value:,.0f}"
    return f"{shown}%" if check.unit == "%" else f"{shown} {check.unit}"


def expectation(check: Check) -> str:
    sign = {"eq": "=", "lte": "≤", "gte": "≥"}[check.rule]
    bound = f"{check.bound:g}"
    return f"{sign} {bound}%" if check.unit == "%" else f"{sign} {bound} {check.unit}"
