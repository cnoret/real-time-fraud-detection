# Fraud Transactions API

Data source of [real-time-fraud-detection](https://github.com/cnoret/real-time-fraud-detection). It replaces the API provided during the training course, so the pipeline no longer depends on a third-party service.

The API replays a fixed sample of **50,000 transactions** (**195 frauds**, 0.39%) drawn from the simulated *Sparkov* dataset (`fraudTest.csv`), as if the payments were arriving now. The model is trained on `fraudTrain.csv` only, which ends before `fraudTest.csv` starts, so every replayed transaction is unseen data; [`training/train.py`](../training/train.py) checks that none of these `trans_num` is in its training data.

| Endpoint | Purpose |
|---|---|
| `GET /health` | service status, volume and fraud rate of the replay set |
| `GET /v1/transactions?n=1&seed=` | `n` transactions (1 to 100) **without** their label, stamped with `event_time` |
| `GET /v1/labels?trans_num=…` | ground truth `is_fraud` for given ids (simulates chargeback feedback) |
| `GET /current-transactions` | legacy format of the former course API (pandas `orient=split`) |

The label is served apart on purpose: it can never slip into the fields sent to the model, and it lets the pipeline measure precision and recall in production.

## Hosting

Deployed on **Render** (free tier) from the `Dockerfile`, as described in `render.yaml` at the repository root. The service sleeps after 15 minutes without traffic and wakes up on the first call (about a minute).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
# http://localhost:7860/docs
```

Rebuild the replay set: `python scripts/build_holdout.py path/to/fraudTest.csv` (seed 42, reproducible).

Rate limit: 60 calls per minute per client (free hosting).
