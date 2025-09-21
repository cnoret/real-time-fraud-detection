# Streamlit app for reading fraud transactions from Neon (Postgres)

import os
import math
import time
import psycopg2
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# ---------- Config ----------
st.set_page_config(page_title="Fraud Monitor", page_icon="🕵️", layout="wide")
st.title("🕵️ Real-Time Fraud Monitor")

# Example: postgresql://readonly_user:*****@<neon-host>/<db>?sslmode=require
DB_URI = os.getenv("NEON_DB_URI")
if not DB_URI:
    st.error(
        "NEON_DB_URI environment variable is not set. Add it in your Space → Settings → Secrets."
    )
    st.stop()


# ---------- DB helpers ----------
@st.cache_resource(show_spinner=False)
def get_conn():
    # Connection is cached at resource level for reuse in Streamlit
    return psycopg2.connect(DB_URI)


def fetch_df(sql: str, params: tuple = ()):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def fetch_one(sql: str, params: tuple = ()):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return row


# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")

    # Date range on inserted_at (fallback to last 7 days if table empty)
    try:
        minmax = fetch_one(
            "SELECT COALESCE(MIN(inserted_at), NOW() - INTERVAL '7 days') AS min_dt, "
            "COALESCE(MAX(inserted_at), NOW()) AS max_dt FROM fraud_transactions;"
        )
        default_start = minmax[0]
        default_end = minmax[1]
    except Exception:
        default_start = datetime.utcnow() - timedelta(days=7)
        default_end = datetime.utcnow()

    date_range = st.date_input(
        "Inserted at (UTC)",
        value=(default_start.date(), default_end.date()),
        help="Filter rows by inserted_at.",
    )
    start_dt = datetime.combine(date_range[0], datetime.min.time())
    end_dt = datetime.combine(date_range[1], datetime.max.time())

    merchant_q = st.text_input("Merchant contains", value="")
    min_prob = st.slider("Min fraud probability", 0.0, 1.0, 0.0, 0.001)
    pred_select = st.selectbox("Prediction label", ["All", "Fraud (1)", "Legit (0)"])

    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    refresh_sec = st.number_input(
        "Auto-refresh (seconds, 0=off)", min_value=0, max_value=3600, value=0
    )

# ---------- Auto refresh ----------
if refresh_sec and refresh_sec > 0:
    # This will force the app to rerun every X seconds
    st.rerun()  # only triggers when page is reloaded
    st.query_params["ts"] = int(time.time() // refresh_sec)

# ---------- Query builder ----------
where = ["inserted_at BETWEEN %s AND %s", "fraud_probability >= %s"]
params = [start_dt, end_dt, min_prob]

if merchant_q:
    where.append("merchant ILIKE %s")
    params.append(f"%{merchant_q}%")

if pred_select != "All":
    where.append("prediction = %s")
    params.append(1 if "Fraud" in pred_select else 0)

where_sql = " AND ".join(where)

# Count total for pagination
count_sql = f"SELECT COUNT(*) FROM fraud_transactions WHERE {where_sql};"
try:
    total_rows = fetch_one(count_sql, tuple(params))[0]
except Exception as e:
    st.error(f"Query failed (COUNT): {e}")
    st.stop()

# Pagination state
if "page" not in st.session_state:
    st.session_state.page = 1

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    if st.button("⟲ Refresh"):
        st.session_state.page = 1
        st.cache_data.clear()
with col_b:
    total_pages = max(1, math.ceil(total_rows / page_size))
    st.write(f"Page {st.session_state.page} / {total_pages}")
with col_c:
    prev_disabled = st.session_state.page <= 1
    next_disabled = st.session_state.page >= total_pages
    btn_prev, btn_next = st.columns(2)
    with btn_prev:
        if st.button("← Prev", disabled=prev_disabled):
            st.session_state.page -= 1
    with btn_next:
        if st.button("Next →", disabled=next_disabled):
            st.session_state.page += 1

offset = (st.session_state.page - 1) * page_size

# Data query (ordered by latest insert)
data_sql = f"""
SELECT transaction_id, amount, merchant, fraud_probability, prediction, inserted_at
FROM fraud_transactions
WHERE {where_sql}
ORDER BY inserted_at DESC
LIMIT %s OFFSET %s;
"""

try:
    df = fetch_df(data_sql, tuple(params) + (page_size, offset))
except Exception as e:
    st.error(f"Query failed (SELECT): {e}")
    st.stop()

# ---------- KPI cards ----------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Rows (filtered)", value=f"{total_rows:,}")
with kpi2:
    try:
        last_prob = fetch_one(
            f"SELECT fraud_probability FROM fraud_transactions WHERE {where_sql} ORDER BY inserted_at DESC LIMIT 1;",
            tuple(params),
        )
        st.metric("Last probability", value=f"{(last_prob[0] if last_prob else 0):.4f}")
    except Exception:
        st.metric("Last probability", value="—")
with kpi3:
    try:
        fraud_rate = (
            fetch_one(
                f"SELECT AVG(CASE WHEN prediction=1 THEN 1.0 ELSE 0.0 END) FROM fraud_transactions WHERE {where_sql};",
                tuple(params),
            )[0]
            or 0.0
        )
        st.metric("Fraud rate (filtered)", value=f"{fraud_rate*100:.2f}%")
    except Exception:
        st.metric("Fraud rate (filtered)", value="—")
with kpi4:
    st.metric("Min prob filter", value=f"{min_prob:.3f}")

# ---------- Chart ----------
try:
    chart_df = fetch_df(
        f"""
        SELECT date_trunc('hour', inserted_at) AS ts_hour,
               AVG(fraud_probability) AS avg_prob,
               SUM(CASE WHEN prediction=1 THEN 1 ELSE 0 END) AS frauds,
               COUNT(*) AS total
        FROM fraud_transactions
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY 1 ASC;
        """,
        tuple(params),
    )
    if not chart_df.empty:
        st.subheader("Hourly trend (avg probability & counts)")
        # Two simple charts (avoid heavy libs)
        st.line_chart(chart_df.set_index("ts_hour")["avg_prob"], height=220)
        st.bar_chart(chart_df.set_index("ts_hour")[["total", "frauds"]], height=220)
except Exception as e:
    st.warning(f"Chart query failed: {e}")

# ---------- Table ----------
st.subheader("Transactions")
st.dataframe(df, use_container_width=True)


# ---------- CSV Export ----------
@st.cache_data(show_spinner=False)
def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


st.download_button(
    "⬇️ Download current page (CSV)",
    data=convert_df_to_csv(df),
    file_name="fraud_transactions_page.csv",
    mime="text/csv",
)

# ---------- Footer ----------
st.markdown(
    """
    <hr style="margin-top:2em; margin-bottom:1em;">
    <div style="text-align:center; color:gray; font-size:0.9em;">
        Made with ❤️ by <b>Christophe Noret</b><br>
        <a href="https://github.com/cnoret" target="_blank" style="margin-right:12px;">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" 
                 alt="GitHub" width="20">
        </a>
        <a href="https://www.linkedin.com/in/christophenoret" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 alt="LinkedIn" width="20">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
