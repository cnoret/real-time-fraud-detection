"""Tests for the Streamlit dashboard: pure metrics, data quality rules, and a smoke run of every page."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import metrics
import quality

NOW = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
APP = Path(__file__).resolve().parents[1] / "dashboard" / "src" / "streamlit_app.py"


COLUMNS = [
    "transaction_id", "amount", "merchant", "fraud_probability",
    "prediction", "is_fraud", "event_time", "inserted_at",
    "model_version", "decision_threshold",
]


def _frame(rows):
    """rows: (probability, prediction, is_fraud, minutes_ago)."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": f"tx{i}",
                "amount": 10.0 + i,
                "merchant": f"fraud_Merchant {i % 3}",
                "fraud_probability": p,
                "prediction": pred,
                "is_fraud": label,
                "event_time": NOW - timedelta(minutes=ago, seconds=30),
                "inserted_at": NOW - timedelta(minutes=ago),
                "model_version": metrics.CURRENT_MODEL,
                "decision_threshold": metrics.SERVED_THRESHOLD,
            }
            for i, (p, pred, label, ago) in enumerate(rows)
        ],
        columns=COLUMNS,
    )
    df["is_fraud"] = df["is_fraud"].astype("Int64")
    df["prediction"] = df["prediction"].astype("Int64")
    return df


@pytest.fixture
def sample():
    return _frame([
        (0.95, 1, 1, 1),     # caught
        (0.60, 1, 1, 2),     # caught
        (0.05, 0, 1, 3),     # missed
        (0.40, 1, 0, 70),    # false alert
        (0.02, 0, 0, 71),    # passed
        (0.01, 0, 0, 72),    # passed
        (0.30, 1, None, 73), # label pending
    ])


# ── Confusion matrix and rates ────────────────────────────────────────────────

def test_confusion_uses_stored_prediction_on_labelled_rows(sample):
    c = metrics.confusion(sample)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 2)
    assert c.labelled == 6
    assert c.precision == pytest.approx(2 / 3)
    assert c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)


def test_confusion_with_custom_threshold(sample):
    c = metrics.confusion(sample, threshold=0.5)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 0, 1, 3)
    assert c.precision == 1.0


def test_for_model_treats_unversioned_rows_as_legacy(sample):
    mixed = sample.assign(model_version=[None, None] + [metrics.CURRENT_MODEL] * 5)
    assert len(metrics.for_model(mixed, metrics.LEGACY_MODEL)) == 2
    assert len(metrics.for_model(mixed, metrics.CURRENT_MODEL)) == 5
    assert len(metrics.for_model(sample.drop(columns="model_version"), metrics.LEGACY_MODEL)) == 7


def test_served_threshold_prefers_stored_value(sample):
    assert metrics.served_threshold(sample) == metrics.SERVED_THRESHOLD
    legacy = sample.assign(decision_threshold=None)
    assert metrics.served_threshold(legacy, metrics.LEGACY_MODEL) == metrics.LEGACY_THRESHOLD


def test_offline_reference_is_at_the_served_threshold():
    assert metrics.OFFLINE_THRESHOLD == pytest.approx(metrics.SERVED_THRESHOLD)


def test_rates_are_none_without_denominator():
    c = metrics.Confusion(tp=0, fp=0, fn=0, tn=5)
    assert c.precision is None and c.recall is None and c.f1 is None
    assert metrics.Confusion(tp=0, fp=2, fn=3, tn=5).f1 == 0.0


def test_confusion_of_empty_frame():
    empty = _frame([])
    assert metrics.confusion(empty).labelled == 0
    assert metrics.label_coverage(empty) is None
    assert metrics.batches(empty).empty


def test_threshold_curve_long_format(sample):
    curve = metrics.threshold_curve(sample, [0.1, 0.5])
    assert set(curve["metric"]) == {"Precision", "Recall", "F1"}
    recall = curve[(curve.metric == "Recall")].set_index("threshold")["value"]
    assert recall[0.1] == pytest.approx(2 / 3) and recall[0.5] == pytest.approx(2 / 3)


def test_roc_auc(sample):
    # frauds 0.95, 0.60, 0.05 vs legit 0.40, 0.02, 0.01: 8 of 9 pairs ranked right
    assert metrics.roc_auc(sample) == pytest.approx(8 / 9)
    only_legit = sample[sample["is_fraud"] == 0]
    assert metrics.roc_auc(only_legit) is None


def test_label_coverage(sample):
    assert metrics.label_coverage(sample) == pytest.approx(6 / 7)


def test_outcome_labels(sample):
    assert list(metrics.outcome(sample)) == [
        "Caught", "Caught", "Missed", "False alert", "Passed", "Passed", "Label pending",
    ]


# ── Freshness, volume, batches ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "minutes, level",
    [(10, "ok"), (90, "ok"), (91, "warn"), (180, "warn"), (181, "down")],
)
def test_freshness_levels(minutes, level):
    assert metrics.freshness(NOW - timedelta(minutes=minutes), NOW).level == level


def test_freshness_of_empty_table():
    assert metrics.freshness(None, NOW) == metrics.Freshness("empty", None)


@pytest.mark.parametrize(
    "delta, text",
    [
        (timedelta(seconds=45), "45 s"),
        (timedelta(minutes=12), "12 min"),
        (timedelta(hours=3, minutes=5), "3 h 05 min"),
        (timedelta(days=4, hours=2), "4 d"),
        (None, "n/a"),
    ],
)
def test_format_age(delta, text):
    assert metrics.format_age(delta) == text


def test_hourly_volume_keeps_empty_hours(sample):
    hourly = metrics.hourly_volume(sample, NOW - timedelta(hours=3), NOW)
    assert len(hourly) == 4  # 15:00, 16:00, 17:00, 18:00
    assert hourly["scored"].sum() == len(sample)
    assert (hourly["scored"] == 0).sum() == 2


def test_batches_split_on_gaps(sample):
    runs = metrics.batches(sample)
    assert list(runs["rows"]) == [3, 4]  # newest batch first
    assert runs.loc[0, "frauds"] == 3 and runs.loc[1, "labelled"] == 3
    assert runs.loc[1, "alerts"] == 2


def test_ingestion_lag(sample):
    assert metrics.ingestion_lag(sample) == timedelta(seconds=30)


# ── Data quality rules ────────────────────────────────────────────────────────

def _check(rule, bound, severity="fail", unit="rows"):
    return quality.Check("c", "d", "SELECT 1", rule, bound, severity, unit)


@pytest.mark.parametrize(
    "rule, bound, value, expected",
    [
        ("eq", 0, 0, "pass"),
        ("eq", 0, 3, "fail"),
        ("lte", 90, 45, "pass"),
        ("lte", 90, 120, "fail"),
        ("gte", 95, 97.5, "pass"),
        ("gte", 95, 80, "fail"),
        ("eq", 0, None, "fail"),
    ],
)
def test_evaluate(rule, bound, value, expected):
    assert quality.evaluate(_check(rule, bound), value) == expected


def test_warn_severity_and_formatting():
    check = _check("gte", 95, severity="warn", unit="%")
    assert quality.evaluate(check, 50) == "warn"
    assert quality.format_value(check, 96.25) == "96.2%"
    assert quality.expectation(check) == "≥ 95%"


def test_every_check_queries_the_table():
    names = [c.name for c in quality.CHECKS]
    assert len(names) == len(set(names))
    for c in quality.CHECKS:
        assert "FROM fraud_transactions" in c.sql
        assert c.rule in {"eq", "lte", "gte"} and c.severity in {"fail", "warn"}


# ── Smoke run of the app, database replaced by the sample frame ───────────────

@pytest.fixture
def app(sample, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("altair")
    from streamlit.testing.v1 import AppTest

    import db

    monkeypatch.setenv("NEON_DB_URI", "postgresql://test")
    monkeypatch.setattr(db, "load_transactions", lambda since: sample)
    monkeypatch.setattr(
        db, "load_overall",
        lambda: {"total": len(sample), "first": sample["inserted_at"].min(),
                 "last": sample["inserted_at"].max(), "now": pd.Timestamp(NOW)},
    )
    monkeypatch.setattr(
        db, "run_quality_checks", lambda: [(c.name, 0.0 if c.rule != "gte" else 100.0) for c in quality.CHECKS]
    )
    return AppTest.from_file(str(APP), default_timeout=30)


def test_app_entry_renders_overview(app):
    app.run()
    assert not app.exception, app.exception
    assert [t.value for t in app.title] == ["Overview"]


@pytest.mark.parametrize(
    "page, title",
    [
        ("overview", "Overview"),
        ("model_performance", "Model performance"),
        ("pipeline_runs", "Pipeline runs"),
        ("data_quality", "Data quality"),
        ("transactions", "Transactions"),
    ],
)
def test_every_page_renders(app, page, title):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(
        f"import charts, ui, views\nui.inject_css()\nviews.{page}()\n", default_timeout=30
    )
    at.run()
    assert not at.exception, at.exception
    assert [t.value for t in at.title] == [title]
