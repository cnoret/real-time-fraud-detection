"""Small HTML building blocks (status tiles, panels) styled for the Night console theme."""

from __future__ import annotations

from html import escape

import streamlit as st

LEVEL_COLORS = {"ok": "#6FCF97", "warn": "#F0A060", "down": "#E5737A", "empty": "#9EA1AC"}

CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1400px; }
.nc-sub { color: #9EA1AC; margin: -0.6rem 0 1.2rem; }
.nc-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 1rem; }
.nc-tile { background: #15171D; border: 1px solid #23262F; border-radius: 12px; padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }
.nc-tile.warn { border-color: #3A2F22; }
.nc-tile.down { border-color: #44262A; }
.nc-label { display: flex; align-items: center; gap: 8px; color: #9EA1AC; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
.nc-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.nc-value { font-size: 1.25rem; font-weight: 600; color: #E7E7EB; }
.nc-detail { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #9EA1AC; }
.nc-feed { list-style: none; margin: 0; padding: 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.nc-feed li { display: grid; grid-template-columns: 40px minmax(0, 1fr) 68px 36px; gap: 10px; padding: 8px 0; border-bottom: 1px solid #20232C; }
.nc-feed li:last-child { border-bottom: 0; }
.nc-feed .t { color: #9EA1AC; }
.nc-feed .m { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nc-feed .r { text-align: right; }
.nc-feed .hot { color: #F0A060; font-weight: 500; }
.nc-meter { display: grid; grid-template-columns: 84px minmax(0, 1fr) 64px; gap: 12px; align-items: center; margin: 6px 0 10px; }
.nc-meter .bar { height: 8px; background: #20232C; border-radius: 4px; }
.nc-meter .fill { height: 8px; background: #C9BBF5; border-radius: 4px; }
.nc-meter .v { font-family: 'IBM Plex Mono', monospace; text-align: right; }
.nc-meter .k { color: #A9ACB7; }
.nc-note { color: #9EA1AC; font-size: 0.8rem; }
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.html(f'<p class="nc-sub">{escape(subtitle)}</p>')


def tiles(items: list[dict]) -> None:
    """items: dicts with label, value, detail, level ("ok" | "warn" | "down" | "empty")."""
    parts = []
    for it in items:
        level = it.get("level", "ok")
        color = LEVEL_COLORS.get(level, LEVEL_COLORS["empty"])
        parts.append(
            f'<div class="nc-tile {escape(level)}">'
            f'<span class="nc-label"><span class="nc-dot" style="background:{color}"></span>{escape(it["label"])}</span>'
            f'<span class="nc-value">{escape(it["value"])}</span>'
            f'<span class="nc-detail">{escape(it["detail"])}</span>'
            "</div>"
        )
    st.html(f'<div class="nc-tiles">{"".join(parts)}</div>')


def feed(rows: list[dict], hot_from: float) -> None:
    """Latest transactions: time, merchant, amount, probability (orange at or above hot_from)."""
    items = []
    for r in rows:
        cls = "r hot" if r["probability"] >= hot_from else "r t"
        items.append(
            "<li>"
            f'<span class="t">{escape(r["time"])}</span>'
            f'<span class="m">{escape(r["merchant"])}</span>'
            f'<span class="r">{escape(r["amount"])}</span>'
            f'<span class="{cls}">{r["probability"]:.2f}</span>'
            "</li>"
        )
    st.html(f'<ul class="nc-feed">{"".join(items)}</ul>')


def meters(items: list[tuple[str, float | None, str]]) -> None:
    """Horizontal meters: (label, share between 0 and 1 or None, shown value)."""
    parts = []
    for label, share, shown in items:
        width = 0 if share is None else max(0.0, min(1.0, share)) * 100
        parts.append(
            f'<div class="nc-meter"><span class="k">{escape(label)}</span>'
            f'<div class="bar"><div class="fill" style="width:{width:.1f}%"></div></div>'
            f'<span class="v">{escape(shown)}</span></div>'
        )
    st.html("".join(parts))


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"
