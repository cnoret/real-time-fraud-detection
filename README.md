# Real-Time Fraud Detection: an orchestrated scoring pipeline for card payments

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://cnoret-fraud-monitor.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cnoret/real-time-fraud-detection/blob/main/EDA_training.ipynb)
[![Tests](https://github.com/cnoret/real-time-fraud-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/cnoret/real-time-fraud-detection/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-3.0-017CEE?logo=apacheairflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An orchestrated pipeline (Airflow, plus an hourly GitHub Actions run that keeps the demo live) scores a replayed stream of card transactions with a fraud model (ROC AUC 0.967 offline), stores each prediction next to its ground-truth label in PostgreSQL, and feeds a live monitoring dashboard.**

**Data:** 555,719 simulated card transactions (Sparkov, `fraudTest.csv`) · 0.39% fraud (2,145 cases) · June to December 2020 · 50,000-transaction holdout replayed as a live stream

**Live app:** [cnoret-fraud-monitor.streamlit.app](https://cnoret-fraud-monitor.streamlit.app/)

---

## Business question

A card issuer has to decide, for each payment, whether to let it through or flag it for review. Missing a fraud costs money; flagging too many genuine payments costs analysts' time and customer trust. This project asks: **how many frauds can a simple, explainable model catch in near real time, and at what false-alarm cost?**

## Results (offline evaluation)

Trained in [`EDA_training.ipynb`](EDA_training.ipynb) on a stratified 80/20 split (111,144 test transactions, 429 frauds), tracked with MLflow.

| Model | Precision (fraud) | Recall (fraud) | F1 | ROC AUC |
|---|---|---|---|---|
| **Logistic regression (selected)** | **45.1%** | **59.2%** | **0.51** | **0.967** |
| Random forest | 4.5% | 75.5% | 0.08 | 0.937 |

At the default 0.5 threshold, the logistic regression catches about 6 frauds in 10, and roughly 1 alert in 2 is a real fraud. The random forest catches more frauds but raises about 20 false alarms per real one, which would swamp a review team. The production API uses a lower threshold (0.1, `balanced`) to favour recall.

## Architecture

```mermaid
flowchart LR
    A[Transactions API<br/>FastAPI on Render] -->|transactions| B[Airflow DAG, local<br/>or GitHub Actions, hourly]
    A -->|ground-truth label,<br/>served separately| B
    B -->|features only| C[Model API<br/>FastAPI on Render]
    C -->|probability + decision| B
    B -->|upsert| D[(Neon PostgreSQL)]
    D --> E[Streamlit dashboard<br/>Streamlit Community Cloud]
```

All hosting runs on free tiers.

| Component | Folder | Role |
|---|---|---|
| Transactions API | [`transactions-api/`](transactions-api/) | Replays a fixed 50,000-transaction holdout (195 frauds) as if payments were arriving now. The `is_fraud` label is served by a separate endpoint so it can never leak into the model's features. Live at [fraud-transactions-api.onrender.com](https://fraud-transactions-api.onrender.com/docs). |
| Airflow DAG | [`airflow/dags/fraud_detection_dag_v2.py`](airflow/dags/fraud_detection_dag_v2.py) | Runs every minute: `ingestion` → `labeling` → `inference` → `persistence`. Rejects a payload that contains the label, aligns it to the model schema, and upserts the result on `transaction_id`. |
| Scheduled run | [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml), [`scripts/score_batch.py`](scripts/score_batch.py) | Same steps as the DAG, reused from [`fraud_pipeline.py`](airflow/dags/fraud_pipeline.py), on a GitHub Actions cron: every hour it wakes both APIs, scores a batch of 30 transactions (`/predict/batch`) and upserts them. Hourly keeps the two free Render services well under their 750 free instance-hours a month. |
| Model API | [`model-api/`](model-api/) | Serves the scikit-learn pipeline (`/predict`, `/predict/batch`, `/thresholds`). The model file is versioned as a [GitHub Release](https://github.com/cnoret/real-time-fraud-detection/releases/tag/model-v1.0.0) and downloaded at build time with its exact scikit-learn version. Live at [fraud-model-api.onrender.com](https://fraud-model-api.onrender.com/docs). |
| Database | Neon (PostgreSQL) | Table `fraud_transactions`: amount, merchant, fraud probability, prediction, ground truth, event time. |
| Dashboard | [`dashboard/src/`](dashboard/src/) | Dark multipage monitoring console (Streamlit + Altair): pipeline freshness and status, live precision, recall, F1 and ROC AUC against `is_fraud` with a threshold explorer, ingestion batches, SQL data quality checks ([`quality.py`](dashboard/src/quality.py)), and a filterable transaction table with CSV export. |

Both Render services sleep after 15 minutes without traffic and take about a minute to wake up.

## Project structure

```
.
├── .github/workflows/tests.yml          # CI: ruff + pytest on every push and pull request
├── .github/workflows/pipeline.yml       # Hourly scheduled run (GitHub Actions)
├── airflow/
│   ├── dags/fraud_detection_dag_v2.py   # Orchestration (Airflow 3, TaskFlow)
│   ├── dags/fraud_pipeline.py           # Scoring logic shared by the DAG and the scheduled run
│   └── docker-compose.yaml              # Local Airflow stack
├── scripts/score_batch.py               # Batch runner used by the scheduled workflow
├── transactions-api/                    # Transaction stream (FastAPI, Docker)
├── model-api/                           # Model serving (FastAPI, Docker)
├── dashboard/src/                       # Streamlit console: pages, metrics, SQL quality checks
├── .streamlit/config.toml               # Dashboard theme
├── tests/                               # pytest suite (APIs, DAG logic, model download, dashboard)
├── EDA_training.ipynb                   # EDA, model training and selection
├── client_realtime.py                   # Minimal client that calls both APIs
├── render.yaml                          # Render Blueprint for both APIs
├── requirements-dev.txt                 # Pinned test and lint dependencies
├── ruff.toml                            # Lint rules
└── pytest.ini
```

## How to run

**Tests** (90 tests, a few seconds; the same checks run in GitHub Actions on every push and pull request):

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest
```

**Airflow** (local):

```bash
cd airflow
docker compose up
```

Then open `http://localhost:8080` and set the Airflow Variable `NEON_DB_URI` (`TRANSACTIONS_API` and `MODEL_API` default to the Render URLs).

**Scheduled run** (any machine, no Airflow):

```bash
pip install -r scripts/requirements.txt
python scripts/score_batch.py --n 5 --dry-run          # score only, writes nothing
NEON_DB_URI="postgresql://..." python scripts/score_batch.py --n 30
```

On GitHub, the workflow reads the repository secret `NEON_DB_URI` (and optional repository variables `TRANSACTIONS_API`, `MODEL_API`). It can also be started by hand from the Actions tab.

**Dashboard** (local):

```bash
pip install -r dashboard/src/requirements.txt
export NEON_DB_URI="postgresql://user:password@host/db?sslmode=require"
streamlit run dashboard/src/streamlit_app.py
```

**APIs** (local): see [`transactions-api/README.md`](transactions-api/README.md) and [`model-api/README.md`](model-api/README.md).

## Limitations and next steps

- **The hosted pipeline runs hourly, not in real time.** Free tiers set the pace: a GitHub Actions cron (which can start a few minutes late) and Render services that sleep when idle. GitHub also pauses scheduled workflows after 60 days without repository activity.
- **Live metrics depend on volume.** At 30 transactions an hour and a fraud rate under 1 %, a week holds only a handful of frauds: the dashboard warns when precision and recall rest on fewer than 10 labelled frauds.
- **The offline metrics are optimistic.** The model uses identifier-like columns (`cc_num`, `trans_num`, names, street) as features, and the replayed holdout comes from the same file it was trained on. Next: retrain without identifiers and exclude the holdout.
- **No SQL modelling layer yet.** Data quality checks run as SQL from the dashboard; next is moving them, and the metrics, into versioned SQL models (dbt).

## Author

**Christophe Noret** · [GitHub](https://github.com/cnoret) · [LinkedIn](https://www.linkedin.com/in/christophenoret)
