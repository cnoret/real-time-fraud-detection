"""Tests for DAG module-level helper functions (pure Python, no Airflow needed)."""

import pytest

# Airflow is mocked in conftest.py before this import runs.
from fraud_detection_dag_v2 import (
    DEFAULTS,
    FALLBACK_SCHEMA,
    _build_payload,
    _fix_timestamps,
    _parse_label_response,
    _parse_transactions_response,
)


# ── _parse_transactions_response / _parse_label_response ──────────────────────
class TestParseTransactionsResponse:
    def test_returns_first_transaction(self):
        body = {"count": 2, "transactions": [
            {"trans_num": "a", "amt": 10.0}, {"trans_num": "b", "amt": 20.0}]}
        assert _parse_transactions_response(body) == {"trans_num": "a", "amt": 10.0}

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no transaction"):
            _parse_transactions_response({"count": 0, "transactions": []})

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            _parse_transactions_response({"foo": "bar"})

    def test_missing_trans_num_raises(self):
        with pytest.raises(ValueError, match="trans_num"):
            _parse_transactions_response({"transactions": [{"amt": 1.0}]})

    def test_label_in_payload_is_rejected(self):
        with pytest.raises(ValueError, match="Label leaked"):
            _parse_transactions_response(
                {"transactions": [{"trans_num": "a", "is_fraud": 1}]})


class TestParseLabelResponse:
    def test_known_label(self):
        body = {"labels": [{"trans_num": "a", "is_fraud": 1}], "unknown": []}
        assert _parse_label_response(body, "a") == 1

    def test_unknown_returns_none(self):
        assert _parse_label_response({"labels": [], "unknown": ["a"]}, "a") is None


# ── _fix_timestamps ───────────────────────────────────────────────────────────
class TestFixTimestamps:
    def test_rescales_ms_after_2022(self):
        ts_ms = 1_700_000_000_000  # Nov 2023 in ms
        result = _fix_timestamps(
            {"current_time": ts_ms}, ["unix_time", "trans_date_trans_time"]
        )
        training_start = 1_577_836_800
        year_seconds = 31_536_000
        assert training_start <= result["unix_time"] < training_start + year_seconds

    def test_passes_through_ms_before_2022(self):
        ts_ms = 1_600_000_000_000  # Sep 2020 in ms (before 2022-01-01 threshold)
        result = _fix_timestamps({"current_time": ts_ms}, ["unix_time"])
        assert result["unix_time"] == ts_ms // 1000

    def test_converts_unix_to_date_string(self):
        result = _fix_timestamps(
            {"unix_time": 1_592_222_400}, ["unix_time", "trans_date_trans_time"]
        )
        assert "trans_date_trans_time" in result
        assert "2020" in result["trans_date_trans_time"]

    def test_date_string_format(self):
        result = _fix_timestamps(
            {"unix_time": 1_592_222_400}, ["unix_time", "trans_date_trans_time"]
        )
        # Expect YYYY-MM-DD HH:MM:SS
        parts = result["trans_date_trans_time"].split(" ")
        assert len(parts) == 2
        assert len(parts[0].split("-")) == 3

    def test_no_timestamp_fields_unchanged(self):
        tx = {"amt": 42.0, "merchant": "Store"}
        result = _fix_timestamps(tx, ["amt", "merchant"])
        assert result == tx

    def test_returns_copy_not_mutation(self):
        original = {"current_time": 1_700_000_000_000}
        result = _fix_timestamps(original, ["unix_time"])
        assert result is not original
        assert "current_time" not in original or original["current_time"] == 1_700_000_000_000


# ── _build_payload ────────────────────────────────────────────────────────────
class TestBuildPayload:
    def test_uses_transaction_values(self, sample_transaction):
        payload = _build_payload(sample_transaction, list(sample_transaction.keys()))
        assert payload["amt"] == sample_transaction["amt"]
        assert payload["merchant"] == sample_transaction["merchant"]

    def test_filters_to_expected_columns(self, sample_transaction):
        payload = _build_payload(sample_transaction, ["amt", "merchant"])
        assert set(payload.keys()) == {"amt", "merchant"}

    def test_fills_defaults_for_missing_fields(self):
        payload = _build_payload({}, ["amt", "merchant", "gender"])
        assert payload["amt"] == DEFAULTS["amt"]
        assert payload["merchant"] == DEFAULTS["merchant"]
        assert payload["gender"] == DEFAULTS["gender"]

    def test_none_value_falls_back_to_default(self):
        payload = _build_payload({"amt": None}, ["amt"])
        assert payload["amt"] == DEFAULTS["amt"]

    def test_unknown_column_gets_zero(self):
        payload = _build_payload({}, ["unknown_col"])
        assert payload["unknown_col"] == 0

    def test_trans_num_unknown_gets_default_string(self):
        payload = _build_payload({}, ["trans_num"])
        assert isinstance(payload["trans_num"], str)
        assert payload["trans_num"].startswith("default_")

    def test_all_fallback_schema_columns_covered(self):
        payload = _build_payload({}, FALLBACK_SCHEMA)
        assert set(payload.keys()) == set(FALLBACK_SCHEMA)
