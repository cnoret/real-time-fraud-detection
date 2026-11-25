"""Build-time check: the downloaded model is the one its model card describes.

    python scripts/check_model.py fraud_model.pkl model_card.json model-v2.0.0

Fails (exit 1) when the card's version or file hash does not match, or when the
pipeline cannot score a sample transaction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# The pipeline pickles a reference to fraud_features.py, which sits next to app.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(model_path: str, card_path: str, version: str) -> int:
    card = json.load(open(card_path))
    if card.get("model_version") != version:
        print(f"model_card.json is for {card.get('model_version')}, image builds {version}")
        return 1
    digest = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    if digest != card.get("model_sha256"):
        print(f"fraud_model.pkl sha256 {digest} does not match the model card")
        return 1

    import joblib
    import pandas as pd

    model = joblib.load(model_path)
    sample = pd.DataFrame([{c: None for c in card["input_columns"]} | {"amt": 42.0, "category": "grocery_pos"}])
    proba = float(model.predict_proba(sample)[0, 1])
    print(f"Model OK: {version}, steps {list(model.named_steps)}, sample probability {proba:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
