"""Fraud Monitor: a dark, multipage Streamlit console over the fraud_transactions table (Neon PostgreSQL).

Pages live in views.py, metrics in metrics.py (pure pandas), SQL data quality checks in quality.py.
Run locally:  NEON_DB_URI=postgresql://... streamlit run dashboard/src/streamlit_app.py
"""

import streamlit as st

import charts  # noqa: F401  (registers the Altair theme)
import db
import ui
import views

st.set_page_config(page_title="Fraud Monitor", page_icon=":material/shield:", layout="wide")
ui.inject_css()

if not db.db_uri():
    st.error(
        "NEON_DB_URI is not set. Add it to the app secrets "
        "(Streamlit Community Cloud → Settings → Secrets) or export it before `streamlit run`."
    )
    st.stop()

PAGES = [
    st.Page(views.overview, title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(views.model_performance, title="Model performance", icon=":material/insights:"),
    st.Page(views.pipeline_runs, title="Pipeline runs", icon=":material/schedule:"),
    st.Page(views.data_quality, title="Data quality", icon=":material/fact_check:"),
    st.Page(views.transactions, title="Transactions", icon=":material/table_rows:"),
]
# Hidden built-in menu: the sidebar is drawn below so the brand header sits above the links.
pages = st.navigation(PAGES, position="hidden")

with st.sidebar:
    ui.brand("Fraud Monitor")
    for page in PAGES:
        st.page_link(page, width="stretch")
    st.divider()
    st.segmented_control(
        "Time window",
        list(views.WINDOWS),
        default=views.DEFAULT_WINDOW,
        key="window",
        help="Applies to every page except Data quality, which checks the whole table.",
    )
    st.selectbox(
        "Model",
        list(views.MODELS),
        format_func=views.MODELS.get,
        key="model",
        help="Rows are scored by one model release each. The retired v1 had seen the replayed data.",
    )
    if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
        st.cache_data.clear()
    st.caption("Queries are cached for 60 s. All times are UTC.")
    st.divider()
    st.caption(
        "Built by **Christophe Noret** · "
        "[GitHub](https://github.com/cnoret/real-time-fraud-detection) · "
        "[LinkedIn](https://www.linkedin.com/in/christophenoret)"
    )

pages.run()
