"""
Shared fixtures and Airflow mocking for the test suite.

The Airflow mock must be installed into sys.modules before any DAG file
is imported, so it lives at the top of conftest.py (executed first by pytest).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Mock Airflow (not installed locally) ──────────────────────────────────────
_airflow_stub = MagicMock()
_airflow_stub.models.Variable.get.return_value = "postgresql://mock/mock"

for _mod in [
    "airflow",
    "airflow.decorators",
    "airflow.models",
    "airflow.utils",
    "airflow.utils.task_group",
    "airflow.operators",
    "airflow.operators.empty",
]:
    sys.modules.setdefault(_mod, _airflow_stub)

# ── Feature lists (mirrors the model's expected schema) ───────────────────────
NUMERIC_FEATURES = [
    "cc_num", "amt", "zip", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long",
]
CATEGORICAL_FEATURES = [
    "merchant", "category", "first", "last", "gender",
    "street", "city", "state", "job", "dob",
    "trans_date_trans_time", "trans_num",
]


# ── Mock model factory ────────────────────────────────────────────────────────
def _make_mock_model(fraud_proba: float = 0.2):
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - fraud_proba, fraud_proba]])
    num_t, cat_t, prep = MagicMock(), MagicMock(), MagicMock()
    num_t.feature_names_in_ = NUMERIC_FEATURES
    cat_t.feature_names_in_ = CATEGORICAL_FEATURES
    prep.named_transformers_ = {"num": num_t, "cat": cat_t}
    m.named_steps = {"preproc": prep}
    return m


# ── API fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def api_client():
    """TestClient backed by a low-risk mock model (fraud_proba=0.0005)."""
    with patch("joblib.load", return_value=_make_mock_model(fraud_proba=0.0005)):
        sys.modules.pop("fraud_api", None)
        import fraud_api as fa
        from fastapi.testclient import TestClient
        yield TestClient(fa.app)


@pytest.fixture(scope="session")
def fraud_api_module(api_client):  # noqa: ARG001 — ensures api_client initialised first
    import fraud_api
    return fraud_api


# ── Data fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture()
def sample_transaction():
    return {
        "amt": 150.75,
        "cc_num": 4000000000000002,
        "zip": 12345,
        "lat": 40.7128,
        "long": -74.006,
        "city_pop": 50000,
        "unix_time": 1_592_222_400,
        "merch_lat": 40.7128,
        "merch_long": -74.006,
        "merchant": "Test Store",
        "category": "misc_pos",
        "first": "Jane",
        "last": "Doe",
        "gender": "F",
        "street": "456 Oak Ave",
        "city": "Testville",
        "state": "CA",
        "job": "Teacher",
        "dob": "1985-05-15",
        "trans_date_trans_time": "2020-06-15 12:00:00",
        "trans_num": "abc123",
    }
