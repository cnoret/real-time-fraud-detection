# 🕵️ Real-Time Fraud Detection Project

End-to-end fraud detection pipeline with **Airflow**, **Hugging Face Spaces**, **Neon (Postgres)**, and a **Streamlit dashboard**.  

[![Monitoring](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/cnoret/fraud-monitor-streamlit)

---
## ✨ Features

- **Airflow DAGs**:
  - Fetch real-time transactions (Jedha API based on Kaggle dataset.)
  - Preprocess + send to Hugging Face model
  - Store results in NeonDB
- **Streamlit Dashboard**:
  - KPIs, fraud probability trends, interactive filters
  - Auto-refresh, CSV export
  - Deployed on Hugging Face Spaces
- **EDA Notebook** for dataset exploration
- **API scripts** for serving and testing the fraud detection model

---

## 📂 Project Structure

```
.
├── airflow/               # Airflow environment (docker-compose, dags, plugins, logs)
│   ├── dags/              # DAGs (MVP, graph, etc.)
│   ├── config/            # configs
│   └── docker-compose.yml # Airflow local stack
│
├── api-deploy/            # Deployment configs for fraud API
├── client_realtime.py     # Local client to test API
├── dashboard/             # (Optional) dashboard assets
├── src/
│   └── streamlit_app.py   # Streamlit dashboard (deployed on HF)
│
├── notebooks/
│   └── EDA_fraud_detection.ipynb # Exploratory Data Analysis
│
├── fraud_api.py           # Fraud detection API (Hugging Face Space)
├── fraud_model.pkl        # Trained model (local storage, can be ignored in git)
├── fraudTest.csv          # Dataset (large → consider .gitignore)
│
├── requirements.txt       # Project dependencies
├── .env                   # Secrets (ignored in git)
└── README.md              # Project documentation
```

> ⚠️ Large dataset (EDA, Training) `fraudTest.csv` is **not pushed** to the repo.  
> You can download it from [this link](https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv).  

---

## 🚀 How to Run

### 1. Airflow (local)
```bash
cd airflow
docker-compose up
```
- DAGs will appear in Airflow UI (`http://localhost:8080`)
- Runs every 5 minutes → fetch → predict → store in Neon

### 2. Streamlit Dashboard (local)
```bash
cd src
streamlit run streamlit_app.py
```
→ App available at `http://localhost:8501`

### 3. Streamlit on Hugging Face
- Push `streamlit_app.py` as `app.py` and `requirements.txt` to your Space
- Add NeonDB URI as a **secret**:
  ```
  NEON_DB_URI=postgresql://user:password@host/db?sslmode=require
  ```

---

## ⚙️ Requirements

Main dependencies (see `requirements.txt`):
```
streamlit
psycopg2-binary
pandas
streamlit
plotly
apache-airflow
```

---

## 📊 Architecture

```
Jedha API (transactions) 
   → Airflow DAG 
      → Hugging Face Space (fraud model API) 
         → NeonDB (Postgres) 
            → Streamlit dashboard (HF Space)
```

---

## 🔮 Future Improvements

- **Cloud Storage Archive (S3)**:  
  Store raw + enriched transactions in S3 (JSON/Parquet) for auditing & retraining.
- **Alerts**: Slack/Email notifications when fraud probability > threshold.
- **ML Pipeline**: Automated retraining from archived data.
- **Scalability**: Deploy Airflow on managed service.
- **Database connection handling**:  
  Current Streamlit app uses a direct `psycopg2` connection.  
  In production, it would be better to:
  - Add automatic reconnection with retries (to avoid `connection already closed` errors when Neon suspends idle sessions).
  - Enable TCP keepalives to reduce silent timeouts.
  - Or switch to **SQLAlchemy** with connection pooling (`pool_pre_ping=True`) for a robust, production-ready setup.
---

## 👤 Author

**Christophe Noret**  
- [GitHub](https://github.com/cnoret)  
- [LinkedIn](https://www.linkedin.com/in/christophenoret)
