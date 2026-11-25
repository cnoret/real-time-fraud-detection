"""Tests for DAG module-level helper functions (pure Python, no Airflow needed)."""

import pytest

# Airflow is mocked in conftest.py before this import runs.
from fraud_detection_dag_v2 import (
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
    def test_source_value_wins(self):
        tx = {"trans_date_trans_time": "2020-08-02 07:55:28", "unix_time": 1_375_430_128}
        assert _fix_timestamps(tx)["trans_date_trans_time"] == "2020-08-02 07:55:28"

    def test_derived_from_current_time_ms_in_utc(self):
        result = _fix_timestamps({"current_time": 1_700_000_000_000})
        assert result["trans_date_trans_time"] == "2023-11-14 22:13:20"

    def test_derived_from_unix_time_in_utc(self):
        result = _fix_timestamps({"unix_time": 1_592_222_400})
        assert result["trans_date_trans_time"] == "2020-06-15 12:00:00"

    def test_no_timestamp_fields_unchanged(self):
        tx = {"amt": 42.0, "merchant": "Store"}
        assert _fix_timestamps(tx) == tx

    def test_bad_timestamp_is_left_missing(self):
        assert "trans_date_trans_time" not in _fix_timestamps({"current_time": "soon"})

    def test_returns_copy_not_mutation(self):
        original = {"current_time": 1_700_000_000_000}
        result = _fix_timestamps(original)
        assert result is not original
        assert original == {"current_time": 1_700_000_000_000}


# ── _build_payload ────────────────────────────────────────────────────────────
class TestBuildPayload:
    def test_uses_transaction_values(self, sample_transaction):
        payload = _build_payload(sample_transaction, list(sample_transaction.keys()))
        assert payload["amt"] == sample_transaction["amt"]
        assert payload["merchant"] == sample_transaction["merchant"]

    def test_filters_to_expected_columns(self, sample_transaction):
        payload = _build_payload(sample_transaction, ["amt", "category"])
        assert payload == {"amt": 150.75, "category": "misc_pos"}

    def test_missing_fields_are_null_not_invented(self):
        assert _build_payload({}, ["amt", "gender"]) == {"amt": None, "gender": None}

    def test_fallback_schema_has_no_identifier(self, sample_transaction):
        payload = _build_payload(sample_transaction, FALLBACK_SCHEMA)
        assert set(payload) == set(FALLBACK_SCHEMA)
        for identifier in ("trans_num", "cc_num", "first", "last", "street", "zip", "merchant", "unix_time"):
            assert identifier not in payload
