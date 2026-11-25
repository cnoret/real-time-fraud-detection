"""Tests for the model's feature engineering, its model card and the build-time check."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fraud_features as ff

ROOT = Path(__file__).resolve().parents[1]
CARD = json.loads((ROOT / "model-api" / "model_card.json").read_text())
IDENTIFIERS = {"trans_num", "cc_num", "first", "last", "street", "zip", "merchant", "unix_time", "is_fraud"}

RAW = {
    "trans_date_trans_time": "2020-08-02 07:55:28",
    "amt": 1.07,
    "category": "shopping_pos",
    "gender": "F",
    "dob": "1964-11-17",
    "city_pop": 5438,
    "lat": 41.4682,
    "long": -72.5751,
    "merch_lat": 41.240083,
    "merch_long": -71.837788,
    "trans_num": "cf581d75ccc9ba838a05dec8bfa78b5b",
    "cc_num": 377895991033232,
}


# ── Feature engineering ───────────────────────────────────────────────────────
class TestBuildFeatures:
    def test_columns_in_order(self):
        assert list(ff.build_features(pd.DataFrame([RAW])).columns) == ff.FEATURES

    def test_no_identifier_is_a_feature_or_an_input(self):
        assert not IDENTIFIERS & set(ff.FEATURES)
        assert not IDENTIFIERS & set(ff.INPUT_COLUMNS)

    def test_values(self):
        row = ff.build_features(pd.DataFrame([RAW])).iloc[0]
        assert row["hour"] == 7 and row["day_of_week"] == 6  # Sunday
        assert row["age"] == pytest.approx(55.7, abs=0.1)
        assert row["log_amt"] == pytest.approx(np.log1p(1.07))
        assert 60 < row["distance_km"] < 70

    def test_identifiers_do_not_change_features(self):
        a = ff.build_features(pd.DataFrame([RAW]))
        b = ff.build_features(pd.DataFrame([RAW | {"trans_num": "other", "cc_num": 1}]))
        pd.testing.assert_frame_equal(a, b)

    def test_missing_and_bad_fields_become_nan(self):
        row = ff.build_features(pd.DataFrame([{"amt": "oops", "dob": None}])).iloc[0]
        assert row[ff.NUMERIC_FEATURES].isna().all()


# ── Model card ────────────────────────────────────────────────────────────────
class TestModelCard:
    def test_version_matches_dockerfile(self):
        dockerfile = (ROOT / "model-api" / "Dockerfile").read_text()
        assert re.search(r"ARG MODEL_VERSION=(\S+)", dockerfile).group(1) == CARD["model_version"]

    def test_inputs_match_feature_module(self):
        assert CARD["input_columns"] == ff.INPUT_COLUMNS
        assert CARD["features"] == ff.FEATURES

    def test_trained_on_train_file_only(self):
        assert CARD["data"]["train"]["file"] == "fraudTrain.csv"
        assert CARD["data"]["test"]["file"] == "fraudTest.csv"
        assert CARD["data"]["replayed_holdout_rows_in_train"] == 0
        assert CARD["data"]["train"]["period"][1] <= CARD["data"]["test"]["period"][0]

    def test_threshold_is_the_balanced_operating_point(self):
        assert CARD["threshold"] == CARD["thresholds"]["balanced"]
        assert CARD["metrics"]["test"]["threshold"] == CARD["threshold"]

    def test_dashboard_reads_the_same_card(self):
        import metrics

        assert metrics.CURRENT_MODEL == CARD["model_version"]
        assert metrics.SERVED_THRESHOLD == CARD["thresholds"]["balanced"]
        assert metrics.OFFLINE_METRICS["precision"] == CARD["metrics"]["test"]["precision"]


# ── Build-time check (model-api/scripts/check_model.py) ───────────────────────
@pytest.fixture(scope="module")
def check_model():
    spec = importlib.util.spec_from_file_location("check_model", ROOT / "model-api" / "scripts" / "check_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_model_rejects_wrong_version(check_model, tmp_path):
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"model_version": "model-v9"}))
    assert check_model.main(str(card), str(card), "model-v2.0.0") == 1


def test_check_model_rejects_wrong_hash(check_model, tmp_path):
    model = tmp_path / "m.pkl"
    model.write_bytes(b"not the model")
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"model_version": "v", "model_sha256": hashlib.sha256(b"x").hexdigest()}))
    assert check_model.main(str(model), str(card), "v") == 1
