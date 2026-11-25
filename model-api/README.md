# Fraud Detection Model API

FastAPI service that serves the fraud detection model, a scikit-learn pipeline trained by
[`training/train.py`](../training/train.py). The scheduled pipeline and the Airflow DAG call it
for every transaction.

| Endpoint | Purpose |
|---|---|
| `GET /health` | service status, model version, raw input columns and default decision threshold |
| `POST /predict` | fraud probability, decision and risk level for one transaction |
| `POST /predict/batch` | the same for a list of transactions, in a single model call |
| `GET /thresholds` | named decision thresholds (`balanced`, the default, maximises F1 on validation) |
| `GET /model` | the model card: training data, protocol, thresholds and offline metrics |

The pipeline computes its own features ([`fraud_features.py`](fraud_features.py)), so a request
sends raw transaction fields. Extra fields (ids, names) are ignored and missing ones are scored as
missing values and listed in `details.missing_fields`.

```bash
curl -X POST https://fraud-model-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"trans_date_trans_time": "2020-08-02 23:15:00", "amt": 950.0,
                "category": "shopping_net", "gender": "F", "dob": "1964-11-17",
                "city_pop": 5438, "lat": 41.47, "long": -72.58,
                "merch_lat": 41.24, "merch_long": -71.84},
       "threshold": "balanced"}'
```

## Model

`fraud_model.pkl` (0.5 MB) is not stored in git: it is attached to a
[GitHub Release](https://github.com/cnoret/real-time-fraud-detection/releases) (tag `model-v2.0.0`)
and downloaded when the image is built (`MODEL_VERSION` build argument in the `Dockerfile`). Its
model card, [`model_card.json`](model_card.json), is versioned here and records the file's SHA-256.

At build time:

- `scripts/fetch_model.py` downloads the pickle and reads the scikit-learn version that saved it;
  the image installs exactly that version, since a pickled estimator only reloads reliably with it.
- `scripts/check_model.py` fails the build if the card's version or hash does not match the
  downloaded file, or if the pipeline cannot score a sample transaction.

Publishing a new model: run the training script, attach the pickle to a new release, copy the new
card here and bump `MODEL_VERSION`.

## Hosting

Deployed on **Render** (free plan) with the Transactions API, from `render.yaml` at the repository
root. The service sleeps after 15 minutes without traffic and wakes up on the next call (about a
minute).

Locally:

```bash
docker build -t fraud-model-api model-api
docker run -p 7860:7860 fraud-model-api   # http://localhost:7860/docs
```
