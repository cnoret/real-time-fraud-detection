"""The dashboard's pages: Overview, Model performance, Pipeline runs, Data quality, Transactions."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

import charts
import db
import metrics
import quality
import ui

WINDOWS = {"24 h": timedelta(days=1), "7 d": timedelta(days=7), "30 d": timedelta(days=30), "All": None}
DEFAULT_WINDOW = "7 d"
ACTIONS_URL = "https://github.com/cnoret/real-time-fraud-detection/actions/workflows/pipeline.yml"
FRESHNESS_WORDS = {"ok": "Fresh", "warn": "Late", "down": "Stale", "empty": "No data"}


# ── Shared context ────────────────────────────────────────────────────────────

MODELS = {
    metrics.CURRENT_MODEL: f"{metrics.CURRENT_MODEL} (served)",
    metrics.LEGACY_MODEL: f"{metrics.LEGACY_MODEL} (retired, leaky)",
}


def _window() -> str:
    return st.session_state.get("window") or DEFAULT_WINDOW


def _model() -> str:
    return st.session_state.get("model") or metrics.CURRENT_MODEL


def _context():
    """(table-wide figures, rows of the selected model in the selected window, window label)."""
    overall = db.load_overall()
    delta = WINDOWS[_window()]
    since = None if delta is None else (overall["now"] - delta).floor("min").to_pydatetime()
    return overall, metrics.for_model(db.load_transactions(since), _model()), _window()


def _threshold(df) -> float:
    return metrics.served_threshold(df, _model())


def _empty_window(overall: dict, label: str) -> bool:
    """Explain an empty window; returns True when there is nothing to show."""
    if overall["total"] == 0:
        st.info("The table is empty: no transaction has been scored yet.")
        return True
    age = metrics.format_age(overall["now"] - overall["last"])
    st.info(
        f"No transaction scored by {_model()} in the last {label}. The latest insert is {age} old: "
        "pick a wider window or another model in the sidebar."
    )
    return True


def _freshness_tile(overall: dict) -> dict:
    f = metrics.freshness(overall["last"], overall["now"])
    last = overall["last"]
    return {
        "label": "Ingestion",
        "value": f"{FRESHNESS_WORDS[f.level]} · {metrics.format_age(f.age)}",
        "detail": "no insert yet" if last is None else f"last insert {last:%d %b %H:%M} UTC",
        "level": f.level,
    }


def _quality_summary() -> tuple[int, int, str, str]:
    """(passing, total, level, detail) over all data quality checks."""
    results = dict(db.run_quality_checks())
    statuses = [(c, quality.evaluate(c, results.get(c.name))) for c in quality.CHECKS]
    passing = sum(s == "pass" for _, s in statuses)
    failing = [c.name for c, s in statuses if s == "fail"]
    warning = [c.name for c, s in statuses if s == "warn"]
    level = "down" if failing else "warn" if warning else "ok"
    detail = (failing + warning)[0] if failing or warning else "all checks passing"
    return passing, len(statuses), level, detail


def _money(x: float) -> str:
    return f"${x:,.2f}" if x < 1000 else f"${x:,.0f}"


def _section(title: str) -> None:
    st.markdown(f"**{title}**")


# ── Pages ─────────────────────────────────────────────────────────────────────

def overview() -> None:
    ui.header("Overview", "Card transactions scored by an hourly pipeline · Transactions API → Model API → Neon PostgreSQL")
    overall, df, label = _context()
    if df.empty and _empty_window(overall, label):
        return

    conf = metrics.confusion(df)
    coverage = metrics.label_coverage(df)
    alerts = int(metrics.predicted(df).sum())
    passing, total_checks, q_level, q_detail = _quality_summary()
    ui.tiles([
        _freshness_tile(overall),
        {
            "label": "Alerts",
            "value": f"{alerts:,} raised",
            "detail": f"{ui.pct(alerts / len(df))} of {len(df):,} scored · {label}",
            "level": "ok",
        },
        {
            "label": "Labels",
            "value": f"{ui.pct(coverage, 0)} joined",
            "detail": f"is_fraud known for {conf.labelled:,} rows",
            "level": "ok" if (coverage or 0) >= 0.95 else "warn",
        },
        {
            "label": "Data quality",
            "value": f"{passing} / {total_checks} checks",
            "detail": q_detail,
            "level": q_level,
        },
    ])

    left, right = st.columns([1.6, 1], gap="medium")
    with left, st.container(border=True):
        _section("Transactions per hour, last 48 h · orange dots are alerts")
        start = max(df["inserted_at"].min(), overall["now"] - timedelta(hours=48))
        recent = df[df["inserted_at"] >= start]
        hourly = metrics.hourly_volume(recent, start, overall["now"])
        st.altair_chart(charts.hourly_volume(hourly), theme=None, width="stretch")
    with right, st.container(border=True):
        _section(f"Latest transactions · {df['inserted_at'].iloc[0]:%d %b} UTC")
        latest = df.head(9)
        ui.feed(
            [
                {
                    "time": f"{r.inserted_at:%H:%M}",
                    "merchant": str(r.merchant).removeprefix("fraud_"),
                    "amount": _money(r.amount),
                    "probability": float(r.fraud_probability),
                }
                for r in latest.itertuples()
            ],
            hot_from=_threshold(df),
        )

    left, right = st.columns([1, 2], gap="medium")
    with left, st.container(border=True):
        _section("Model vs ground truth")
        ui.meters([
            ("Recall", conf.recall, ui.pct(conf.recall)),
            ("Precision", conf.precision, ui.pct(conf.precision)),
            ("F1", conf.f1, "n/a" if conf.f1 is None else f"{conf.f1:.2f}"),
        ])
        st.altair_chart(charts.confusion_matrix(conf), theme=None, width="stretch")
        st.caption(f"Stored decisions of {_model()}, threshold {_threshold(df):.2f}, on {conf.labelled:,} labelled rows.")
    with right, st.container(border=True):
        _section("Recent ingestion batches")
        _batches_table(metrics.batches(df).head(8))


def model_performance() -> None:
    ui.header("Model performance", "Live predictions joined with their ground-truth label (is_fraud)")
    overall, df, label = _context()
    if df.empty and _empty_window(overall, label):
        return
    lab = metrics.labelled(df)
    if lab.empty:
        st.info("No labelled transaction in this window yet: labels arrive with each batch.")
        return

    served = _threshold(df)
    threshold = st.slider(
        "Decision threshold",
        0.01, 0.99, round(served, 2), 0.01,
        help=f"The pipeline stores decisions at {served:.2f} (the API's 'balanced' setting). "
        "Move it to see the trade-off between missed frauds and false alerts.",
    )
    conf = metrics.confusion(df, threshold)
    auc = metrics.roc_auc(df)
    current = _model() == metrics.CURRENT_MODEL
    off = metrics.OFFLINE_METRICS if current else {}
    # Rates move with the threshold: compare with offline only at the threshold they were measured at.
    same_threshold = current and abs(threshold - metrics.OFFLINE_THRESHOLD) < 0.006

    def delta(live, key):
        ref = off.get(key)
        if live is None or ref is None or not same_threshold:
            return None
        return f"{(live - ref) * 100:+.1f} pts vs offline"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Recall", ui.pct(conf.recall), delta(conf.recall, "recall"),
        help="Share of actual frauds the model flagged.", border=True,
    )
    c2.metric(
        "Precision", ui.pct(conf.precision), delta(conf.precision, "precision"),
        help="Share of alerts that were actual frauds.", border=True,
    )
    c3.metric(
        "F1 score", "n/a" if conf.f1 is None else f"{conf.f1:.2f}",
        None if conf.f1 is None or off.get("f1") is None or not same_threshold
        else f"{conf.f1 - off['f1']:+.2f} vs offline",
        help="Harmonic mean of precision and recall.", border=True,
    )
    c4.metric(
        "ROC AUC",
        "n/a" if auc is None else f"{auc:.3f}",
        None if auc is None or off.get("roc_auc") is None else f"{auc - off['roc_auc']:+.3f} vs offline",
        help="Threshold-free: how well scores rank frauds above legitimate payments.",
        border=True,
    )
    if not current:
        st.warning(
            f"{metrics.LEGACY_MODEL} was trained on the very transactions the API replays, "
            "so these live figures are inflated. Kept for comparison only."
        )
    frauds = conf.tp + conf.fn
    if frauds < 10:
        st.warning(f"Only {frauds} labelled frauds in this window: these rates are noisy. Pick a wider window.")

    left, right = st.columns([1, 1.4], gap="medium")
    with left, st.container(border=True):
        _section(f"Confusion matrix · threshold {threshold:.2f}")
        st.altair_chart(charts.confusion_matrix(conf), theme=None, width="stretch")
        st.html(
            f'<p class="nc-note">{conf.tp + conf.fp:,} alerts for {frauds:,} actual frauds '
            f"among {conf.labelled:,} labelled transactions.</p>"
        )
    with right, st.container(border=True):
        _section("Precision, recall and F1 by threshold")
        st.altair_chart(charts.threshold_curve(metrics.threshold_curve(df), threshold), theme=None, width="stretch")

    with st.container(border=True):
        _section("How scores separate fraud from legit")
        st.altair_chart(charts.score_distribution(df, threshold), theme=None, width="stretch")
        if off.get("precision") is not None:
            st.caption(
                f"Offline reference for {metrics.CURRENT_MODEL} (fraudTest.csv, never seen in training, "
                f"threshold {metrics.OFFLINE_THRESHOLD:.2f}): precision {off['precision']:.0%}, "
                f"recall {off['recall']:.0%}, F1 {off['f1']:.2f}, ROC AUC {off['roc_auc']:.3f}, "
                f"PR AUC {off['pr_auc']:.2f}."
            )


def pipeline_runs() -> None:
    ui.header("Pipeline runs", "Hourly GitHub Actions run: wake the APIs, fetch transactions and labels, score, upsert")
    overall, df, label = _context()
    if df.empty and _empty_window(overall, label):
        return
    runs = metrics.batches(df)
    lag = metrics.ingestion_lag(df)
    ui.tiles([
        _freshness_tile(overall),
        {"label": "Rows", "value": f"{len(df):,} scored", "detail": f"{overall['total']:,} in the table", "level": "ok"},
        {"label": "Batches", "value": f"{len(runs):,} runs", "detail": f"{label} · median {runs['rows'].median():.0f} rows", "level": "ok"},
        {"label": "Ingestion lag", "value": metrics.format_age(lag), "detail": "median, event_time to insert", "level": "ok"},
    ])
    hourly = metrics.hourly_volume(df, df["inserted_at"].min(), overall["now"])
    with st.container(border=True):
        _section("Transactions per hour · empty hours are missed runs")
        st.altair_chart(charts.hourly_volume(hourly, show_alerts=False), theme=None, width="stretch")
    if hourly["hour"].dt.floor("D").nunique() > 2:
        with st.container(border=True):
            _section("Transactions per day")
            st.altair_chart(charts.daily_volume(hourly), theme=None, width="stretch")
    with st.container(border=True):
        _section("Ingestion batches")
        _batches_table(runs)
        st.markdown(f"Workflow logs: [GitHub Actions · Scheduled pipeline]({ACTIONS_URL})")


def data_quality() -> None:
    ui.header("Data quality", "SQL checks on fraud_transactions, run every 5 minutes")
    results = dict(db.run_quality_checks())
    rows = []
    for check in quality.CHECKS:
        value = results.get(check.name)
        rows.append({
            "Status": quality.evaluate(check, value).capitalize(),
            "Check": check.name,
            "Measured": quality.format_value(check, value),
            "Expected": quality.expectation(check),
            "Why it matters": check.description,
        })
    table = pd.DataFrame(rows)
    passing = (table["Status"] == "Pass").sum()
    st.markdown(f"**{passing} of {len(table)} checks passing**")
    st.dataframe(
        table.style.map(
            lambda s: {"Pass": "color: #6FCF97", "Warn": "color: #F0A060", "Fail": "color: #E5737A"}.get(s, ""),
            subset=["Status"],
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("SQL behind each check"):
        for check in quality.CHECKS:
            st.markdown(f"**{check.name}**")
            st.code(check.sql, language="sql")


def transactions() -> None:
    ui.header("Transactions", "Every scored transaction in the selected window, newest first")
    overall, df, label = _context()
    if df.empty and _empty_window(overall, label):
        return
    df = df.assign(outcome=metrics.outcome(df))

    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    merchant = c1.text_input("Merchant contains")
    min_prob = c2.slider("Min fraud probability", 0.0, 1.0, 0.0, 0.01)
    outcomes = c3.multiselect(
        "Outcome", ["Caught", "Missed", "False alert", "Passed", "Label pending"], placeholder="All outcomes"
    )
    view = df[df["fraud_probability"] >= min_prob]
    if merchant:
        view = view[view["merchant"].str.contains(merchant, case=False, na=False, regex=False)]
    if outcomes:
        view = view[view["outcome"].isin(outcomes)]

    st.caption(f"{len(view):,} of {len(df):,} transactions · {label}")
    shown = view.assign(merchant=view["merchant"].str.removeprefix("fraud_"))  # dataset prefix, not a label
    st.dataframe(
        shown[["inserted_at", "merchant", "amount", "fraud_probability", "outcome", "transaction_id"]],
        hide_index=True,
        width="stretch",
        height=560,
        column_config={
            "inserted_at": st.column_config.DatetimeColumn("Scored at (UTC)", format="D MMM YYYY, HH:mm"),
            "merchant": st.column_config.TextColumn("Merchant"),
            "amount": st.column_config.NumberColumn("Amount", format="dollar"),
            "fraud_probability": st.column_config.ProgressColumn("Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"),
            "outcome": st.column_config.TextColumn("Outcome"),
            "transaction_id": st.column_config.TextColumn("Transaction id"),
        },
    )
    st.download_button(
        "Download CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="fraud_transactions.csv",
        mime="text/csv",
    )


def _batches_table(runs: pd.DataFrame) -> None:
    st.dataframe(
        runs,
        hide_index=True,
        width="stretch",
        column_config={
            "started": st.column_config.DatetimeColumn("Started (UTC)", format="D MMM, HH:mm"),
            "rows": st.column_config.NumberColumn("Rows"),
            "alerts": st.column_config.NumberColumn("Alerts"),
            "frauds": st.column_config.NumberColumn("Actual frauds"),
            "labelled": st.column_config.NumberColumn("Labelled"),
            "duration_s": st.column_config.NumberColumn("Insert span", format="%.1f s"),
        },
    )
