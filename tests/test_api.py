"""Tests for fraud_api FastAPI endpoints."""

import numpy as np
import pytest


# ── /health ───────────────────────────────────────────────────────────────────
class TestHealth:
    def test_status_200(self, api_client):
        assert api_client.get("/health").status_code == 200

    def test_required_keys(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == "model-v2.0.0"
        assert "trans_num" not in data["input_columns"]
        assert "amt" in data["input_columns"]
        assert "available_thresholds" in data

    def test_decision_threshold_is_the_balanced_one(self, api_client, fraud_api_module):
        data = api_client.get("/health").json()
        assert data["decision_threshold"] == fraud_api_module.THRESHOLDS["balanced"]

    def test_all_thresholds_listed(self, api_client):
        data = api_client.get("/health").json()
        assert set(data["available_thresholds"]) == {
            "conservative", "balanced", "sensitive", "very_sensitive"
        }


# ── /thresholds ───────────────────────────────────────────────────────────────
class TestThresholds:
    def test_status_200(self, api_client):
        assert api_client.get("/thresholds").status_code == 200

    def test_all_names_present(self, api_client):
        data = api_client.get("/thresholds").json()
        assert set(data["thresholds"].keys()) == {
            "conservative", "balanced", "sensitive", "very_sensitive"
        }

    def test_threshold_values_come_from_the_model_card(self, api_client, fraud_api_module):
        t = api_client.get("/thresholds").json()["thresholds"]
        assert t == fraud_api_module.MODEL_CARD["thresholds"]

    def test_thresholds_are_ordered(self, api_client):
        t = api_client.get("/thresholds").json()["thresholds"]
        assert t["conservative"] > t["balanced"] > t["sensitive"] > t["very_sensitive"] > 0


# ── /predict ──────────────────────────────────────────────────────────────────
class TestPredict:
    _PAYLOAD = {"data": {"amt": 50.0, "merchant": "Test"}, "threshold": "balanced"}

    def test_status_200(self, api_client):
        assert api_client.post("/predict", json=self._PAYLOAD).status_code == 200

    def test_response_shape(self, api_client):
        data = api_client.post("/predict", json=self._PAYLOAD).json()
        assert {"prediction", "probability", "threshold_used", "risk_level", "details"} \
            <= set(data.keys())

    def test_probability_in_range(self, api_client):
        prob = api_client.post("/predict", json=self._PAYLOAD).json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_low_prob_legit_with_balanced(self, api_client):
        # api_client uses fraud_proba=0.0005, far below every threshold
        data = api_client.post("/predict", json=self._PAYLOAD).json()
        assert data["prediction"] == 0
        assert data["risk_level"] == "LOW"

    @pytest.mark.parametrize("name", ["conservative", "balanced", "sensitive", "very_sensitive"])
    def test_threshold_used_matches_name(self, api_client, fraud_api_module, name):
        payload = {"data": {"amt": 10.0}, "threshold": name}
        data = api_client.post("/predict", json=payload).json()
        assert data["threshold_used"] == fraud_api_module.THRESHOLDS[name]

    def test_unknown_threshold_falls_back_to_balanced(self, api_client, fraud_api_module):
        payload = {"data": {}, "threshold": "nonexistent"}
        data = api_client.post("/predict", json=payload).json()
        assert data["threshold_used"] == fraud_api_module.THRESHOLDS["balanced"]

    def test_model_sees_only_its_input_columns(self, api_client, fraud_api_module, sample_transaction):
        api_client.post("/predict", json={"data": sample_transaction})
        frame = fraud_api_module.model.predict_proba.call_args.args[0]
        assert list(frame.columns) == fraud_api_module.INPUT_COLUMNS

    def test_missing_fields_are_reported(self, api_client):
        details = api_client.post("/predict", json={"data": {"amt": 5.0}}).json()["details"]
        assert "amt" not in details["missing_fields"]
        assert "category" in details["missing_fields"]

    def test_response_names_the_model(self, api_client):
        assert api_client.post("/predict", json=self._PAYLOAD).json()["model_version"] == "model-v2.0.0"

    def test_details_carry_merchant_and_amount(self, api_client):
        payload = {"data": {"amt": 99.99, "merchant": "ACME"}, "threshold": "balanced"}
        details = api_client.post("/predict", json=payload).json()["details"]
        assert details["merchant"] == "ACME"
        assert details["amount"] == 99.99

    def test_all_thresholds_result_in_details(self, api_client):
        data = api_client.post("/predict", json=self._PAYLOAD).json()
        assert set(data["details"]["all_thresholds_result"].keys()) == {
            "conservative", "balanced", "sensitive", "very_sensitive"
        }


# ── /predict/batch ────────────────────────────────────────────────────────────
class TestBatch:
    def test_status_200(self, api_client):
        batch = [{"data": {"amt": 10.0}, "threshold": "balanced"}]
        assert api_client.post("/predict/batch", json=batch).status_code == 200

    def test_count_matches_input(self, api_client):
        batch = [{"data": {"amt": i * 10.0}, "threshold": "balanced"} for i in range(4)]
        data = api_client.post("/predict/batch", json=batch).json()
        assert len(data["predictions"]) == 4

    def test_scores_whole_batch_in_one_call(self, api_client, fraud_api_module):
        mock = fraud_api_module.model.predict_proba
        before = mock.call_count
        batch = [{"data": {"amt": i * 10.0}} for i in range(5)]
        api_client.post("/predict/batch", json=batch)
        assert mock.call_count == before + 1
        assert len(mock.call_args.args[0]) == 5

    def test_empty_batch(self, api_client):
        data = api_client.post("/predict/batch", json=[]).json()
        assert data["predictions"] == []


# ── /model ───────────────────────────────────────────────────────────────────
def test_model_card_endpoint(api_client):
    card = api_client.get("/model").json()
    assert card["model_version"] == "model-v2.0.0"
    assert "test" in card["metrics"]


# ── _risk_level helper ────────────────────────────────────────────────────────
class TestRiskLevel:
    @pytest.mark.parametrize("name,expected", [
        ("conservative", "CRITICAL"),
        ("balanced", "HIGH"),
        ("sensitive", "MEDIUM"),
    ])
    def test_boundaries(self, name, expected, fraud_api_module):
        t = fraud_api_module.THRESHOLDS[name]
        assert fraud_api_module._risk_level(t) == expected
        assert fraud_api_module._risk_level(t - 1e-6) != expected

    def test_low(self, fraud_api_module):
        assert fraud_api_module._risk_level(0.0) == "LOW"

    def test_api_reflects_risk_level(self, api_client, fraud_api_module):
        """Changing mock proba updates risk_level returned by /predict."""
        original = fraud_api_module.model.predict_proba.side_effect
        fraud_api_module.model.predict_proba.side_effect = lambda df: np.array([[0.05, 0.95]])
        try:
            data = api_client.post(
                "/predict", json={"data": {}, "threshold": "balanced"}
            ).json()
            assert data["risk_level"] == "CRITICAL"
            assert data["prediction"] == 1
            assert data["probability"] == pytest.approx(0.95)
        finally:
            fraud_api_module.model.predict_proba.side_effect = original
