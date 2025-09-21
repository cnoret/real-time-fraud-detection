"""A FastAPI application for fraud detection with configurable thresholds."""

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI()

# Load the pre-trained model
model = joblib.load("fraud_model.pkl")

THRESHOLDS = {
    "conservative": 0.5,
    "balanced": 0.1,
    "sensitive": 0.01,
    "very_sensitive": 0.001,
}


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


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: PredictionRequest):
    """Predict fraud with configurable threshold"""

    # Determine threshold
    threshold_name = request.threshold
    if threshold_name not in THRESHOLDS:
        threshold_name = "balanced"
    threshold_value = THRESHOLDS[threshold_name]

    # Prepare input data
    df_input = pd.DataFrame([request.data])

    # Make prediction
    probability = model.predict_proba(df_input)[0, 1]
    prediction = int(probability >= threshold_value)

    # Determine risk level
    if probability >= 0.5:
        risk_level = "CRITICAL"
    elif probability >= 0.1:
        risk_level = "HIGH"
    elif probability >= 0.01:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionResponse(
        prediction=prediction,
        probability=float(probability),
        threshold_used=threshold_value,
        risk_level=risk_level,
        details={
            "threshold_name": threshold_name,
            "amount": request.data.get("amt", 0),
            "merchant": request.data.get("merchant", "unknown"),
            "all_thresholds_result": {
                name: int(probability >= thresh) for name, thresh in THRESHOLDS.items()
            },
        },
    )


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction endpoint"""
    results = []
    for req in requests:
        result = predict_fraud(req)
        results.append(result.dict())
    return {"predictions": results}


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

    uvicorn.run(app, host="0.0.0.0", port=8000)
