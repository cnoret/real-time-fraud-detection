"""Fraud detection model API (FastAPI).

Serves the scikit-learn pipeline trained by training/train.py. The model file is
downloaded at build time from a GitHub Release (see Dockerfile), so it is not
stored in this repository; its model card (model_card.json, versioned here)
carries the decision thresholds and the offline metrics.

The pipeline computes its own features (fraud_features.py), so a request sends
the raw transaction fields listed by /health under `input_columns`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import fraud_features

HERE = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(HERE / "fraud_model.pkl"))
MODEL_CARD_PATH = Path(os.getenv("MODEL_CARD_PATH", str(HERE / "model_card.json")))


def load_model_card(path: Path = MODEL_CARD_PATH) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


model = joblib.load(MODEL_PATH)
MODEL_CARD = load_model_card()
MODEL_VERSION: str = MODEL_CARD.get("model_version", "unknown")
INPUT_COLUMNS: list[str] = MODEL_CARD.get("input_columns") or list(fraud_features.INPUT_COLUMNS)

# Named operating points, chosen on the validation period (see training/train.py).
_DEFAULT_THRESHOLDS = {"conservative": 0.6, "balanced": 0.36, "sensitive": 0.065, "very_sensitive": 0.02}
THRESHOLDS: Dict[str, float] = {**_DEFAULT_THRESHOLDS, **MODEL_CARD.get("thresholds", {})}
DEFAULT_THRESHOLD = "balanced"

DESCRIPTIONS = {
    "conservative": "Precision of at least 95% on validation - few false alerts",
    "balanced": "Maximises F1 on validation - the default decision",
    "sensitive": "Recall of at least 90% on validation - more alerts to review",
    "very_sensitive": "Recall of at least 95% on validation - investigation mode",
}

app = FastAPI(title="Fraud Detection Model API", version="2.0.0")


def _risk_level(probability: float) -> str:
    if probability >= THRESHOLDS["conservative"]:
        return "CRITICAL"
    if probability >= THRESHOLDS["balanced"]:
        return "HIGH"
    if probability >= THRESHOLDS["sensitive"]:
        return "MEDIUM"
    return "LOW"


class PredictionRequest(BaseModel):
    """Raw transaction fields, plus an optional named threshold."""

    data: Dict[str, Any]
    threshold: Optional[str] = DEFAULT_THRESHOLD


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold_used: float
    risk_level: str
    model_version: str
    details: Dict[str, Any]


def _threshold(name: Optional[str]) -> tuple[str, float]:
    if name not in THRESHOLDS:
        name = DEFAULT_THRESHOLD
    return name, THRESHOLDS[name]


def _frame(rows: list[Dict[str, Any]]) -> pd.DataFrame:
    """Keep the model's input columns only; absent fields become missing values."""
    return pd.DataFrame(rows).reindex(columns=INPUT_COLUMNS)


def _response(probability: float, data: Dict[str, Any], threshold: Optional[str]) -> PredictionResponse:
    threshold_name, threshold_value = _threshold(threshold)
    return PredictionResponse(
        prediction=int(probability >= threshold_value),
        probability=float(probability),
        threshold_used=threshold_value,
        risk_level=_risk_level(probability),
        model_version=MODEL_VERSION,
        details={
            "threshold_name": threshold_name,
            "amount": data.get("amt", 0),
            "merchant": data.get("merchant", "unknown"),
            "missing_fields": [c for c in INPUT_COLUMNS if data.get(c) is None],
            "all_thresholds_result": {
                name: int(probability >= value) for name, value in THRESHOLDS.items()
            },
        },
    )


@app.get("/")
def root():
    return {"message": "Fraud Detection API - Ready to detect fraud!", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "input_columns": INPUT_COLUMNS,
        "decision_threshold": THRESHOLDS[DEFAULT_THRESHOLD],
        "available_thresholds": list(THRESHOLDS.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: PredictionRequest):
    probability = model.predict_proba(_frame([request.data]))[0, 1]
    return _response(probability, request.data, request.threshold)


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    """Score all rows in a single predict_proba call."""
    if not requests:
        return {"predictions": []}
    probabilities = model.predict_proba(_frame([r.data for r in requests]))[:, 1]
    return {
        "predictions": [
            _response(p, r.data, r.threshold).model_dump()
            for p, r in zip(probabilities, requests)
        ]
    }


@app.get("/thresholds")
def get_thresholds():
    return {"thresholds": THRESHOLDS, "default": DEFAULT_THRESHOLD, "descriptions": DESCRIPTIONS}


@app.get("/model")
def model_card():
    """The model card: training data, protocol, thresholds and offline metrics."""
    return MODEL_CARD


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
