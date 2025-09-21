import requests
import json
import pandas as pd
import time
from datetime import datetime

# API URLs
JEDHA_URL = "https://charlestng-real-time-fraud-detection.hf.space/current-transactions"
PREDICT_URL = "https://cnoret-fraud-detection-api.hf.space/predict"
HEALTH_URL = "https://cnoret-fraud-detection-api.hf.space/health"


def get_expected_columns():
    """Get expected columns from model"""
    try:
        resp = requests.get(HEALTH_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["expected_numeric"] + data["expected_categorical"]
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return None


def fetch_transaction():
    """Fetch transaction from Jedha API"""
    try:
        resp = requests.get(JEDHA_URL, timeout=10)
        resp.raise_for_status()
        raw = resp.text.strip()

        # Parse JSON
        try:
            obj = json.loads(raw)
            if isinstance(obj, str):
                obj = json.loads(obj)
        except:
            obj = json.loads(raw.strip('"').replace('\\"', '"'))

        df = pd.DataFrame(obj["data"], columns=obj["columns"])
        return df.iloc[0].to_dict()

    except Exception as e:
        print(f"❌ Error fetching transaction: {e}")
        return None


def prepare_data(transaction, expected_columns):
    """Prepare data for prediction"""

    # Fix timestamp
    if "current_time" in transaction and "unix_time" in expected_columns:
        raw_timestamp = transaction["current_time"]
        if raw_timestamp > 1640995200000:  # If future timestamp
            training_range_start = 1577836800  # Jan 1, 2020
            training_range_size = 31536000  # 1 year
            scaled_timestamp = training_range_start + (
                raw_timestamp % training_range_size
            )
            transaction["unix_time"] = scaled_timestamp
        else:
            transaction["unix_time"] = raw_timestamp // 1000

    # Generate datetime string
    if "unix_time" in transaction and "trans_date_trans_time" in expected_columns:
        try:
            dt = datetime.fromtimestamp(transaction["unix_time"])
            transaction["trans_date_trans_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            transaction["trans_date_trans_time"] = "2020-06-15 12:00:00"

    # Build payload with defaults for missing fields
    payload = {}
    defaults = {
        "amt": 100.0,
        "cc_num": 4000000000000002,
        "zip": 12345,
        "city_pop": 50000,
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 40.7128,
        "merch_long": -74.0060,
        "first": "John",
        "last": "Doe",
        "gender": "M",
        "street": "123 Main St",
        "city": "Anytown",
        "state": "NY",
        "job": "Engineer",
        "dob": "1990-01-01",
        "merchant": "Generic Store",
        "category": "misc_pos",
        "trans_date_trans_time": "2020-06-15 12:00:00",
        "unix_time": 1592222400,
    }

    for col in expected_columns:
        if col in transaction and transaction[col] is not None:
            payload[col] = transaction[col]
        elif col in defaults:
            payload[col] = defaults[col]
        else:
            payload[col] = f"default_{int(time.time())}" if col == "trans_num" else 0

    return payload


def predict_fraud(payload):
    """Make fraud prediction"""
    try:
        request_data = {"data": payload}
        resp = requests.post(PREDICT_URL, json=request_data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


def display_result(result):
    """Display prediction result"""
    if not result:
        return

    prediction = result.get("prediction", 0)
    probability = result.get("probability", 0)
    amount = result.get("amount", 0)
    merchant = result.get("merchant", "unknown")

    # Simple classification
    if prediction == 1 or probability > 0.5:
        status = "🚨 FRAUD DETECTED"
        color = "🔴"
    elif probability > 0.01:  # 1%
        status = "⚠️  HIGH RISK"
        color = "🟠"
    elif probability > 0.001:  # 0.1%
        status = "🟡 MEDIUM RISK"
        color = "🟡"
    else:
        status = "✅ LOW RISK"
        color = "🟢"

    print(f"\n{color} {status}")
    print(f"💰 Amount: ${amount}")
    print(f"🏪 Merchant: {merchant}")
    print(f"📊 Fraud Probability: {probability:.6f} ({probability:.4%})")

    # Simple recommendation
    if probability > 0.01:
        print(f"🚫 RECOMMENDATION: Block transaction")
    elif probability > 0.001:
        print(f"👀 RECOMMENDATION: Manual review")
    else:
        print(f"✅ RECOMMENDATION: Approve")


def single_prediction(expected_cols):
    """Run single prediction"""
    print("📡 Fetching transaction...")
    transaction = fetch_transaction()
    if not transaction:
        return None

    print("🔧 Preparing data...")
    payload = prepare_data(transaction, expected_cols)

    print("🤖 Making prediction...")
    result = predict_fraud(payload)

    if result:
        result["amount"] = transaction.get("amt", payload.get("amt", 0))
        result["merchant"] = transaction.get(
            "merchant", payload.get("merchant", "unknown")
        )
        return result

    return None


def main():
    print("🚨 Automatic Fraud Detection 🚨")
    print("=" * 30)

    # Health check
    print("🔄 Checking model...")
    expected_cols = get_expected_columns()
    if not expected_cols:
        print("❌ Model unavailable")
        return

    print("✅ Model ready!")

    # Main loop
    while True:
        print(f"\n" + "=" * 30)
        print("1. Single prediction")
        print("2. Monitor (5 predictions)")
        print("0. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("👋 Goodbye!")
            break

        elif choice == "1":
            result = single_prediction(expected_cols)
            if result:
                display_result(result)

        elif choice == "2":
            print("🚀 Monitoring 5 transactions...")
            for i in range(5):
                print(f"\n--- Transaction {i+1}/5 ---")
                result = single_prediction(expected_cols)
                if result:
                    display_result(result)
                if i < 4:  # Don't wait after last one
                    time.sleep(2)
        else:
            print("❌ Invalid choice")

    print("🏁 Done!")


if __name__ == "__main__":
    main()
