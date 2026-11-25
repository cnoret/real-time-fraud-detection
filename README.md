# Real-Time Fraud Detection: an orchestrated scoring pipeline for card payments

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://cnoret-fraud-monitor.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cnoret/real-time-fraud-detection/blob/main/EDA_training.ipynb)
[![Tests](https://github.com/cnoret/real-time-fraud-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/cnoret/real-time-fraud-detection/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-3.0-017CEE?logo=apacheairflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?logo=postgresql&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An orchestrated, tested pipeline scores a live stream of card payments, stores every decision next to its ground truth in PostgreSQL, and monitors it: on six months of transactions it never saw, the model catches 80% of frauds and 85% of its alerts are real (F1 0.82).**

**Data:** 1.85 M simulated card transactions (Sparkov) · trained on 1,296,675 (Jan 2019 to Jun 2020, 7,506 frauds) · evaluated on the 555,719 that follow (Jun to Dec 2020, 2,145 frauds, 0.39%) · 50,000 of those replayed as the live stream

**Live app:** [cnoret-fraud-monitor.streamlit.app](https://cnoret-fraud-monitor.streamlit.app/)

---

## Business question

A card issuer has to decide, for each payment, whether to let it through or flag it for review. Missing a fraud costs money; flagging too many genuine payments costs analysts' time and customer trust. This project asks: **how many frauds can a model catch in near real time on payments it has never seen, and at what false-alarm cost?**

## Results (offline evaluation)

The model is trained by a reproducible script, [`training/train.py`](training/train.py) (about 40 s on a laptop), with a protocol that keeps every test transaction out of training:

1. **Train on the past, test on the future.** It learns from `fraudTrain.csv` only (January 2019 to June 2020) and is scored once on `fraudTest.csv` (June to December 2020). The transactions the live stream replays come from `fraudTest.csv`; the script checks that none of them is in the training data.
2. **No identifiers.** Card number, transaction id, names, street, zip, merchant name and raw timestamps are excluded. The model reads ten raw fields and derives its own features ([`fraud_features.py`](model-api/fraud_features.py)): amount, category, hour, day of week, cardholder age and gender, city population, and the distance between cardholder and merchant.
3. **Threshold chosen before the test.** The last months of `fraudTrain.csv` (March to June 2020) are held out to choose the decision threshold: 0.36, the one that maximises F1 there. The test set does not influence it.

Results on `fraudTest.csv` (555,719 transactions, 2,145 frauds), from [`model_card.json`](model-api/model_card.json):

| Model | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|
| **Gradient boosting (served, threshold 0.36)** | **85.3%** | **79.5%** | **0.82** | **0.997** | **0.87** |
| Logistic regression, same features (baseline) | 16.8% | 38.7% | 0.23 | 0.851 | 0.13 |

At the chosen threshold the model raises 1,999 alerts over six months: 1,706 real frauds and 293 false alarms, about 1 false alarm for 6 frauds caught. It misses 439 frauds (20%). The model API exposes other operating points, also chosen on the validation months:

| Threshold name | Value | Precision | Recall | Alerts |
|---|---|---|---|---|
| `conservative` (precision ≥ 95% on validation) | 0.60 | 90.6% | 75.0% | 1,775 |
| **`balanced` (best F1, default)** | **0.36** | **85.3%** | **79.5%** | **1,999** |
| `sensitive` (recall ≥ 90% on validation) | 0.07 | 54.5% | 89.6% | 3,527 |
| `very_sensitive` (recall ≥ 95% on validation) | 0.02 | 35.9% | 93.8% | 5,606 |

On the 50,000 replayed transactions alone (195 frauds), the same model reaches 85.4% precision and 84.1% recall: the live dashboard compares its figures with these offline ones at the same threshold.

> An earlier model (v1, in [`EDA_training.ipynb`](EDA_training.ipynb)) was trained and tested on `fraudTest.csv` with identifiers as features, and about 80% of the replayed stream had been in its training data. Its figures are retired; the dashboard keeps its rows apart, labelled as such.

## Architecture

```mermaid
flowchart LR
    A[Transactions API<br/>FastAPI on Render] -->|transactions| B[Airflow DAG, local<br/>or GitHub Actions, hourly]
    A -->|ground-truth label,<br/>served separately| B
    B -->|model inputs only,<br/>no identifiers| C[Model API<br/>FastAPI on Render]
    C -->|probability + decision| B
    T[Training script<br/>fraudTrain.csv] -->|model + model card,<br/>GitHub Release| C
    B -->|upsert| D[(Neon PostgreSQL)]
    D --> E[Streamlit dashboard<br/>Streamlit Community Cloud]
```

All hosting runs on free tiers.

| Component | Folder | Role |
|---|---|---|
| Transactions API | [`transactions-api/`](transactions-api/) | Replays a fixed 50,000-transaction holdout (195 frauds) as if payments were arriving now. The `is_fraud` label is served by a separate endpoint so it can never leak into the model's features. Live at [fraud-transactions-api.onrender.com](https://fraud-transactions-api.onrender.com/docs). |
| Airflow DAG | [`airflow/dags/fraud_detection_dag_v2.py`](airflow/dags/fraud_detection_dag_v2.py) | Runs every minute: `ingestion` → `labeling` → `inference` → `persistence`. Rejects a payload that contains the label, sends the model only its ten input fields, and upserts the result on `transaction_id`. Meant for local runs: on the free Render services, use the hourly scheduled run instead. |
| Scheduled run | [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml), [`scripts/score_batch.py`](scripts/score_batch.py) | Same steps as the DAG, reused from [`fraud_pipeline.py`](airflow/dags/fraud_pipeline.py), on a GitHub Actions cron: every hour it wakes both APIs, scores a batch of 30 transactions (`/predict/batch`) and upserts them. Hourly keeps the two free Render services well under their 750 free instance-hours a month. |
| Training | [`training/train.py`](training/train.py) | Reproducible training and evaluation (protocol above). Writes the model and its [model card](model-api/model_card.json): data fingerprints, thresholds, metrics. |
| Model API | [`model-api/`](model-api/) | Serves the scikit-learn pipeline (`/predict`, `/predict/batch`, `/thresholds`, `/model`). The model file (0.5 MB) is versioned as a [GitHub Release](https://github.com/cnoret/real-time-fraud-detection/releases/tag/model-v2.0.0) and downloaded at build time; the build fails if its hash does not match the model card. Live at [fraud-model-api.onrender.com](https://fraud-model-api.onrender.com/docs). |
| Database | Neon (PostgreSQL) | Table `fraud_transactions`: amount, merchant, fraud probability, prediction, ground truth, event time, and the model version and threshold behind each decision. |
| Dashboard | [`dashboard/src/`](dashboard/src/) | Dark multipage monitoring console (Streamlit + Altair): pipeline freshness and status, live precision, recall, F1 and ROC AUC against `is_fraud`, compared with the offline figures at the same threshold, with a threshold explorer and a model version selector, ingestion batches, SQL data quality checks ([`quality.py`](dashboard/src/quality.py)), and a filterable transaction table with CSV export. |

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
├── transactions-api/                    # Transaction stream and ground-truth labels (FastAPI, Docker)
├── training/train.py                    # Reproducible training and evaluation
├── model-api/                           # Model serving (FastAPI, Docker), features, model card
├── dashboard/src/                       # Streamlit console: pages, metrics, SQL quality checks
├── .streamlit/config.toml               # Dashboard theme
├── tests/                               # pytest suite (APIs, features, DAG logic, model download, dashboard)
├── EDA_training.ipynb                   # Exploratory analysis (its v1 model is retired)
├── client_realtime.py                   # Minimal client: fetch one transaction, score it, print the verdict
├── render.yaml                          # Render Blueprint for both APIs
├── requirements-dev.txt                 # Pinned test and lint dependencies
├── ruff.toml                            # Lint rules
└── pytest.ini
```

## How to run

**Tests** (about 150 tests, a few seconds; the same checks run in GitHub Actions on every push and pull request):

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest
```

**Training** (download [the Kaggle dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) into `data/`, ignored by git):

```bash
pip install -r training/requirements.txt
python training/train.py --train data/fraudTrain.csv --test data/fraudTest.csv
```

It writes `training/artifacts/fraud_model.pkl` and `model_card.json`. Publishing a new model: attach the pickle to a new GitHub Release, copy the card to `model-api/`, and bump `MODEL_VERSION` in [`model-api/Dockerfile`](model-api/Dockerfile).

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
- **Simulated data.** Sparkov transactions are generated, and fraud in them is easier to separate than in real card data (ROC AUC 0.997). The figures show the method works, not what a bank would get.
- **No per-card history.** The strongest real-world signals (how far a payment is from the card's usual amount, time since its previous payment, spending velocity) need a feature store; the API scores each payment on its own, so they are left out.
- **Labels arrive instantly.** The ground truth is served right after the prediction, while real chargebacks take weeks; live recall is therefore known far sooner than it would be.
- **A threshold, not a cost model.** The threshold maximises F1; with real costs per missed fraud and per review, it would be chosen to minimise expected cost.
- **No SQL modelling layer yet.** Data quality checks run as SQL from the dashboard; next is moving them, and the metrics, into versioned SQL models (dbt).

## Author

**Christophe Noret** · [GitHub](https://github.com/cnoret) · [LinkedIn](https://www.linkedin.com/in/christophenoret)
