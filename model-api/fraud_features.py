"""Feature engineering shared by training (training/train.py) and serving (app.py).

The served model is a scikit-learn Pipeline whose first step calls
`build_features`, so the API accepts a raw transaction (the fields of the
Sparkov dataset) and the exact same transformation runs at training and at
inference time. The pickle references this module by name: it must be
importable as `fraud_features` wherever the model is loaded.

Identifiers and personal data are never used as features: no card number,
transaction id, name, street, zip, merchant name or raw timestamp. A model
that sees identifiers can memorise them instead of learning fraud patterns,
and its offline score then says nothing about new transactions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw fields the model reads. Anything else in the payload is ignored.
INPUT_COLUMNS: list[str] = [
    "trans_date_trans_time",  # local time of the payment, "YYYY-MM-DD HH:MM:SS"
    "amt",
    "category",
    "gender",
    "dob",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]

CATEGORICAL_FEATURES: list[str] = ["category", "gender"]
NUMERIC_FEATURES: list[str] = [
    "log_amt",
    "hour",
    "day_of_week",
    "age",
    "log_city_pop",
    "distance_km",
]
FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

_EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """The column if present, else all-missing (the model handles NaN natively)."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw transactions -> model features, in FEATURES order.

    Missing or unparsable fields become NaN rather than errors, so a partial
    payload is still scored (HistGradientBoosting routes missing values).
    """
    df = pd.DataFrame(df)
    when = pd.to_datetime(_column(df, "trans_date_trans_time"), errors="coerce")
    dob = pd.to_datetime(_column(df, "dob"), errors="coerce")
    amt = pd.to_numeric(_column(df, "amt"), errors="coerce")
    city_pop = pd.to_numeric(_column(df, "city_pop"), errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["category"] = _column(df, "category").astype("object")
    out["gender"] = _column(df, "gender").astype("object")
    out["log_amt"] = np.log1p(amt.clip(lower=0))
    out["hour"] = when.dt.hour.astype(float)
    out["day_of_week"] = when.dt.dayofweek.astype(float)
    out["age"] = (when - dob).dt.days / 365.25
    out["log_city_pop"] = np.log1p(city_pop.clip(lower=0))
    out["distance_km"] = _haversine_km(
        pd.to_numeric(_column(df, "lat"), errors="coerce"),
        pd.to_numeric(_column(df, "long"), errors="coerce"),
        pd.to_numeric(_column(df, "merch_lat"), errors="coerce"),
        pd.to_numeric(_column(df, "merch_long"), errors="coerce"),
    )
    return out[FEATURES]


def feature_names_out(transformer, input_features) -> np.ndarray:
    """Output names of the `build_features` step (for FunctionTransformer)."""
    return np.asarray(FEATURES, dtype=object)
