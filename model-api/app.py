"""Fraud detection model API (FastAPI), with configurable decision thresholds.

Serves the scikit-learn pipeline trained in EDA_training.ipynb. The model file is
downloaded at build time (see Dockerfile), so it is not stored in this repository.
"""

import os

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI(title="Fraud Detection Model API", version="1.1.0")

# Load the pre-trained model
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

THRESHOLDS = {
    "conservative": 0.5,
    "balanced": 0.1,
    "sensitive": 0.01,
    "very_sensitive": 0.001,
}


def _risk_level(probability: float) -> str:
    if probability >= 0.5:
        return "CRITICAL"
    if probability >= 0.1:
        return "HIGH"
    if probability >= 0.01:
        return "MEDIUM"
    return "LOW"


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""

    data: Dict[str, Any]
    threshold: Optional[str] = "balanced"  # default


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""

    prediction: int
    probability: float
    threshold_used: float
    risk_level: str
    details: Dict[str, Any]


@app.get("/")
def root():
    """Welcome endpoint"""
    return {"message": "Fraud Detection API - Ready to detect fraud!", "docs": "/docs"}


@app.get("/health")
def health():
    """Health check endpoint"""
    # Extract feature names from the model's preprocessor
    preprocessor = model.named_steps["preproc"]

    numeric_cols = list(preprocessor.named_transformers_["num"].feature_names_in_)
    cat_cols = list(preprocessor.named_transformers_["cat"].feature_names_in_)

    return {
        "status": "healthy",
        "model_loaded": True,
        "expected_numeric": numeric_cols,
        "expected_categorical": cat_cols,
        "available_thresholds": list(THRESHOLDS.keys()),
    }


def _threshold(name: Optional[str]) -> tuple[str, float]:
    if name not in THRESHOLDS:
        name = "balanced"
    return name, THRESHOLDS[name]


def _response(probability: float, data: Dict[str, Any], threshold: Optional[str]) -> PredictionResponse:
    threshold_name, threshold_value = _threshold(threshold)
    return PredictionResponse(
        prediction=int(probability >= threshold_value),
        probability=float(probability),
        threshold_used=threshold_value,
        risk_level=_risk_level(probability),
        details={
            "threshold_name": threshold_name,
            "amount": data.get("amt", 0),
            "merchant": data.get("merchant", "unknown"),
            "all_thresholds_result": {
                name: int(probability >= thresh) for name, thresh in THRESHOLDS.items()
            },
        },
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: PredictionRequest):
    """Predict fraud with configurable threshold"""
    probability = model.predict_proba(pd.DataFrame([request.data]))[0, 1]
    return _response(probability, request.data, request.threshold)


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction endpoint.

    Scores all rows in a single predict_proba call: the pipeline's fixed cost
    (about 1 s per call, mostly the one-hot encoder) is paid once per batch,
    not once per transaction.
    """
    if not requests:
        return {"predictions": []}
    probabilities = model.predict_proba(pd.DataFrame([r.data for r in requests]))[:, 1]
    return {
        "predictions": [
            _response(p, r.data, r.threshold).model_dump()
            for p, r in zip(probabilities, requests)
        ]
    }


@app.get("/thresholds")
def get_thresholds():
    """Get available thresholds and their meanings"""
    return {
        "thresholds": THRESHOLDS,
        "descriptions": {
            "conservative": "Default ML threshold (0.5) - Low false positives",
            "balanced": "Balanced precision/recall - Recommended for production",
            "sensitive": "High recall - Catches more fraud, more false positives",
            "very_sensitive": "Very high recall - Investigation mode",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
