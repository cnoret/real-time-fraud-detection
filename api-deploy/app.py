from fastapi import FastAPI
import joblib
import pandas as pd

# Create FastAPI app
app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Load the trained model
model = joblib.load("fraud_model.pkl")

# Available detection thresholds
THRESHOLDS = {
    "conservative": 0.5,  # Default ML threshold - fewer false alarms
    "balanced": 0.1,  # Good balance - recommended
    "sensitive": 0.01,  # Catches more fraud - more false alarms
    "very_sensitive": 0.001,  # Maximum sensitivity
}


@app.get("/")
def root():
    """Welcome endpoint"""
    return {"message": "Fraud Detection API - Ready to detect fraud!"}


@app.get("/health")
def health_check():
    """Check if API and model are working"""
    # Get expected columns from the model
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


@app.post("/predict")
def predict_fraud(request: dict):
    """
    Predict if a transaction is fraud

    Expected format:
    {
        "data": {transaction_data},
        "threshold": "balanced" (optional)
    }
    """
    # Get transaction data
    transaction_data = request.get("data", {})
    threshold_name = request.get("threshold", "balanced")

    # Validate threshold
    if threshold_name not in THRESHOLDS:
        threshold_name = "balanced"
    threshold_value = THRESHOLDS[threshold_name]

    # Prepare data for model
    df_input = pd.DataFrame([transaction_data])

    # Make prediction
    probability = float(model.predict_proba(df_input)[0, 1])
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

    # Return results
    return {
        "prediction": prediction,  # 0 = Normal, 1 = Fraud
        "probability": probability,  # Fraud probability (0-1)
        "threshold_used": threshold_value,  # Threshold used for decision
        "risk_level": risk_level,  # Risk category
        "details": {
            "threshold_name": threshold_name,
            "amount": transaction_data.get("amt", 0),
            "merchant": transaction_data.get("merchant", "unknown"),
            # Show what other thresholds would predict
            "all_thresholds_result": {
                name: int(probability >= thresh) for name, thresh in THRESHOLDS.items()
            },
        },
    }


@app.get("/thresholds")
def get_thresholds():
    """Get available detection thresholds and their meanings"""
    return {
        "thresholds": THRESHOLDS,
        "descriptions": {
            "conservative": "Safe mode - Low false alarms, may miss some fraud",
            "balanced": "Recommended - Good balance of detection and false alarms",
            "sensitive": "High detection - Catches more fraud, more false alarms",
            "very_sensitive": "Maximum detection - Many alerts, review all manually",
        },
    }


# Run the API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
