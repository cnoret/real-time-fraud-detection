"""Altair charts and the dark chart theme used across the dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd

import metrics

# Palette (kept in sync with .streamlit/config.toml)
BG = "#0D0E12"  # page background (charts are drawn transparent on it)
GRID = "#23262F"
TEXT = "#E7E7EB"
MUTED = "#9EA1AC"
ACCENT = "#C9BBF5"  # lavender: model metrics
NEUTRAL = "#4A5268"  # slate: volume
FRAUD = "#F0A060"  # orange: fraud, alerts, warnings
LEGIT = "#79A8F0"  # blue: legitimate transactions
FONT = "IBM Plex Sans, sans-serif"


@alt.theme.register("night_console", enable=True)
def _night_console() -> alt.theme.ThemeConfig:
    return {
        "config": {
            "background": "transparent",
            "font": FONT,
            "padding": 8,
            "view": {"stroke": None},
            "axis": {
                "domainColor": GRID,
                "gridColor": GRID,
                "tickColor": GRID,
                "labelColor": MUTED,
                "titleColor": MUTED,
                "labelFontSize": 11,
                "titleFontSize": 11,
                "titleFontWeight": 400,
            },
            "legend": {
                "labelColor": MUTED,
                "titleColor": MUTED,
                "labelFontSize": 11,
                "titleFontSize": 11,
                "orient": "top",
            },
            "title": {"color": TEXT, "font": FONT, "fontSize": 13, "fontWeight": 600, "anchor": "start"},
        }
    }


def hourly_volume(hourly: pd.DataFrame, show_alerts: bool = True) -> alt.Chart | alt.LayerChart:
    """Bars of rows scored per hour, optionally with an orange dot sized by the alerts raised that hour."""
    base = alt.Chart(hourly).encode(
        x=alt.X("hour:T", title=None, axis=alt.Axis(format="%d %b %H:%M", labelAngle=0, tickCount=8))
    )
    tooltip = [
        alt.Tooltip("hour:T", title="Hour (UTC)", format="%d %b %H:%M"),
        alt.Tooltip("scored:Q", title="Scored"),
        alt.Tooltip("alerts:Q", title="Alerts"),
        alt.Tooltip("frauds:Q", title="Actual frauds"),
    ]
    bars = base.mark_bar(color=NEUTRAL, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y("scored:Q", title="Transactions scored"), tooltip=tooltip
    )
    dots = (
        base.transform_filter(alt.datum.alerts > 0)
        .mark_circle(color=FRAUD, opacity=1)
        .encode(
            y="scored:Q",
            size=alt.Size("alerts:Q", title="Alerts", scale=alt.Scale(range=[40, 260]), legend=None),
            tooltip=tooltip,
        )
    )
    chart = bars + dots if show_alerts else bars
    return chart.properties(height=260)


def daily_volume(hourly: pd.DataFrame) -> alt.Chart:
    daily = (
        hourly.assign(day=hourly["hour"].dt.floor("D"))
        .groupby("day", as_index=False)[["scored", "alerts", "frauds"]]
        .sum()
    )
    return (
        alt.Chart(daily)
        .mark_bar(color=NEUTRAL, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("yearmonthdate(day):O", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)),
            y=alt.Y("scored:Q", title="Rows per day"),
            tooltip=[
                alt.Tooltip("day:T", title="Day", format="%d %b %Y"),
                alt.Tooltip("scored:Q", title="Scored"),
                alt.Tooltip("alerts:Q", title="Alerts"),
                alt.Tooltip("frauds:Q", title="Actual frauds"),
            ],
        )
        .properties(height=220)
    )


def confusion_matrix(c: metrics.Confusion) -> alt.LayerChart:
    df = pd.DataFrame(
        [
            {"actual": "Fraud", "predicted": "Fraud", "count": c.tp, "kind": "Caught", "good": True},
            {"actual": "Fraud", "predicted": "Legit", "count": c.fn, "kind": "Missed", "good": False},
            {"actual": "Legit", "predicted": "Fraud", "count": c.fp, "kind": "False alert", "good": False},
            {"actual": "Legit", "predicted": "Legit", "count": c.tn, "kind": "Passed", "good": True},
        ]
    )
    df["label"] = df["count"].map("{:,}".format) + "\n" + df["kind"]
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", title="Predicted", sort=["Fraud", "Legit"], axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("actual:N", title="Actual", sort=["Fraud", "Legit"]),
    )
    cells = base.mark_rect(cornerRadius=6, stroke=BG, strokeWidth=6).encode(
        color=alt.Color(
            "good:N",
            scale=alt.Scale(domain=[True, False], range=["#2B2546", "#3A2A1D"]),
            legend=None,
        ),
        tooltip=[alt.Tooltip("kind:N", title="Outcome"), alt.Tooltip("count:Q", title="Transactions")],
    )
    text = base.mark_text(fontSize=15, lineBreak="\n", color=TEXT).encode(text="label:N")
    return (cells + text).properties(height=240)


def threshold_curve(curve: pd.DataFrame, current: float) -> alt.LayerChart:
    lines = (
        alt.Chart(curve)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("threshold:Q", title="Decision threshold", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric:N",
                title=None,
                scale=alt.Scale(domain=["Precision", "Recall", "F1"], range=[ACCENT, FRAUD, LEGIT]),
            ),
            strokeDash=alt.condition(alt.datum.metric == "F1", alt.value([4, 3]), alt.value([1, 0])),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                alt.Tooltip("metric:N"),
                alt.Tooltip("value:Q", format=".1%"),
            ],
        )
    )
    rule = alt.Chart(pd.DataFrame({"t": [current]})).mark_rule(color=TEXT, strokeDash=[3, 4]).encode(x="t:Q")
    return (lines + rule).properties(height=280)


def score_distribution(df: pd.DataFrame, threshold: float, bins: int = 20) -> alt.LayerChart:
    """Share of each class per score bucket: how well scores separate fraud from legit."""
    lab = metrics.labelled(df)
    edges = [i / bins for i in range(bins + 1)]
    tmp = lab.assign(
        bucket=pd.cut(lab["fraud_probability"], edges, include_lowest=True, labels=edges[:-1]).astype(float),
        label=lab["is_fraud"].astype(int).map({1: "Fraud", 0: "Legit"}),
    )
    agg = tmp.groupby(["label", "bucket"], observed=True).size().rename("n").reset_index()
    agg["share"] = agg["n"] / agg.groupby("label")["n"].transform("sum")
    agg["bucket_end"] = agg["bucket"] + 1 / bins
    bars = (
        alt.Chart(agg)
        .mark_bar(opacity=0.7, binSpacing=1)
        .encode(
            x=alt.X("bucket:Q", title="Fraud probability", bin="binned", scale=alt.Scale(domain=[0, 1])),
            x2="bucket_end:Q",
            y=alt.Y("share:Q", title="Share of class", axis=alt.Axis(format="%"), stack=None),
            color=alt.Color("label:N", title=None, scale=alt.Scale(domain=["Fraud", "Legit"], range=[FRAUD, LEGIT])),
            tooltip=[
                alt.Tooltip("label:N", title="Class"),
                alt.Tooltip("bucket:Q", title="From", format=".2f"),
                alt.Tooltip("n:Q", title="Transactions"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=TEXT, strokeDash=[3, 4]).encode(x="t:Q")
    return (bars + rule).properties(height=260)
