"""Tests for the shared pipeline module and the batch runner used by GitHub Actions."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import fraud_pipeline as fp

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_batch.py"


def _load_score_batch():
    spec = importlib.util.spec_from_file_location("score_batch", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_batch"] = module
    spec.loader.exec_module(module)
    return module


def _response(body, status=200):
    r = MagicMock()
    r.ok = status < 400
    r.status_code = status
    r.json.return_value = body
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    return r


TXS = [
    {"trans_num": "a", "amt": 10.0, "merchant": "shop_a", "event_time": "2026-11-25T18:00:00+00:00"},
    {"trans_num": "b", "amt": 900.0, "merchant": "shop_b", "event_time": "2026-11-25T18:00:00+00:00"},
]


# ── Parsing ───────────────────────────────────────────────────────────────────
class TestParsing:
    def test_batch_returns_all(self):
        assert fp.parse_transactions_batch({"transactions": TXS}) == TXS

    def test_batch_rejects_label_on_any_item(self):
        with pytest.raises(ValueError, match="Label leaked"):
            fp.parse_transactions_batch(
                {"transactions": [TXS[0], {"trans_num": "c", "is_fraud": 0}]})

    def test_labels_map(self):
        body = {"labels": [{"trans_num": "a", "is_fraud": 0}, {"trans_num": "b", "is_fraud": 1}],
                "unknown": ["c"]}
        assert fp.parse_labels_response(body) == {"a": 0, "b": 1}

    def test_expected_columns_from_health(self):
        health = {"input_columns": ["amt", "category"]}
        assert fp.expected_columns_from_health(health) == ["amt", "category"]

    def test_expected_columns_empty_health_falls_back(self):
        assert fp.expected_columns_from_health({}) == fp.FALLBACK_SCHEMA

    def test_fallback_schema_matches_model_inputs(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model-api"))
        import fraud_features

        assert fp.FALLBACK_SCHEMA == fraud_features.INPUT_COLUMNS

    def test_parse_prediction_keeps_version_and_threshold(self):
        pred = {"probability": 0.7, "prediction": 1, "threshold_used": 0.36, "model_version": "model-v2.0.0"}
        assert fp.parse_prediction(pred) == {
            "probability": 0.7, "prediction": 1, "model_version": "model-v2.0.0", "threshold": 0.36}

    def test_build_row_column_order(self):
        pred = {"probability": 0.7, "prediction": 1, "model_version": "model-v2.0.0", "threshold": 0.36}
        row = fp.build_row(TXS[1], pred, 1)
        assert row == ("b", 900.0, "shop_b", 0.7, 1, 1, "2026-11-25T18:00:00+00:00", "model-v2.0.0", 0.36)
        assert row.__len__() == fp.UPSERT_SQL.count("%s")

    def test_build_row_unknown_label_is_null(self):
        assert fp.build_row(TXS[0], {"probability": 0.0, "prediction": 0}, None)[5] is None


# ── API calls ─────────────────────────────────────────────────────────────────
class TestApiCalls:
    def test_fetch_transactions_rejects_oversized_batch(self):
        with pytest.raises(ValueError):
            fp.fetch_transactions("http://tx", n=fp.MAX_BATCH + 1)

    def test_fetch_transactions(self):
        with patch.object(fp.requests, "get", return_value=_response({"transactions": TXS})) as get:
            assert fp.fetch_transactions("http://tx/", n=2) == TXS
        assert get.call_args.args[0] == "http://tx/v1/transactions"
        assert get.call_args.kwargs["params"] == {"n": 2}

    def test_fetch_labels_sends_all_ids(self):
        body = {"labels": [{"trans_num": "a", "is_fraud": 0}], "unknown": ["b"]}
        with patch.object(fp.requests, "get", return_value=_response(body)) as get:
            assert fp.fetch_labels("http://tx", ["a", "b"]) == {"a": 0}
        assert get.call_args.kwargs["params"] == {"trans_num": ["a", "b"]}

    def test_fetch_labels_failure_is_not_fatal(self):
        with patch.object(fp.requests, "get", side_effect=requests.ConnectionError("down")):
            assert fp.fetch_labels("http://tx", ["a"]) == {}

    def test_fetch_labels_empty_skips_call(self):
        with patch.object(fp.requests, "get") as get:
            assert fp.fetch_labels("http://tx", []) == {}
        get.assert_not_called()

    def test_fetch_expected_columns_fallback(self):
        with patch.object(fp.requests, "get", return_value=_response({}, status=503)):
            assert fp.fetch_expected_columns("http://model") == fp.FALLBACK_SCHEMA

    def test_predict_batch(self):
        body = {"predictions": [{"probability": 0.01, "prediction": 0, "risk_level": "LOW"},
                                {"probability": 0.9, "prediction": 1, "risk_level": "CRITICAL"}]}
        with patch.object(fp.requests, "post", return_value=_response(body)) as post:
            preds = fp.predict_batch("http://model", [{"amt": 1}, {"amt": 2}])
        assert [(p["probability"], p["prediction"]) for p in preds] == [(0.01, 0), (0.9, 1)]
        assert post.call_args.kwargs["json"] == [{"data": {"amt": 1}}, {"data": {"amt": 2}}]

    def test_predict_batch_splits_into_chunks(self):
        def fake_post(url, json, timeout):
            return _response({"predictions": [{"probability": 0.0, "prediction": 0}] * len(json)})

        with patch.object(fp.requests, "post", side_effect=fake_post) as post:
            preds = fp.predict_batch("http://model", [{"amt": i} for i in range(45)], chunk_size=20)
        assert len(preds) == 45
        assert [len(c.kwargs["json"]) for c in post.call_args_list] == [20, 20, 5]

    def test_predict_batch_retries_gateway_errors(self):
        ok = _response({"predictions": [{"probability": 0.2, "prediction": 1}]})
        with patch.object(fp.requests, "post",
                          side_effect=[_response({}, status=502), requests.Timeout("slow"), ok]) as post, \
             patch.object(fp.time, "sleep"):
            assert fp.predict_batch("http://model", [{"amt": 1}])[0]["probability"] == 0.2
        assert post.call_count == 3

    def test_predict_batch_gives_up_after_retries(self):
        with patch.object(fp.requests, "post", return_value=_response({}, status=502)), \
             patch.object(fp.time, "sleep"):
            with pytest.raises(requests.HTTPError):
                fp.predict_batch("http://model", [{"amt": 1}])

    def test_predict_batch_does_not_retry_client_errors(self):
        with patch.object(fp.requests, "post", return_value=_response({}, status=422)) as post:
            with pytest.raises(requests.HTTPError):
                fp.predict_batch("http://model", [{"amt": 1}])
        assert post.call_count == 1

    def test_predict_batch_length_mismatch(self):
        with patch.object(fp.requests, "post", return_value=_response({"predictions": []})):
            with pytest.raises(ValueError, match="0 predictions for 1"):
                fp.predict_batch("http://model", [{"amt": 1}])

    def test_wait_until_healthy_retries_then_succeeds(self):
        responses = [requests.ConnectionError("cold"), _response({}, status=502), _response({})]
        with patch.object(fp.requests, "get", side_effect=responses), \
             patch.object(fp.time, "sleep") as sleep:
            assert fp.wait_until_healthy("http://svc", attempts=3, delay=1) is True
        assert sleep.call_count == 2

    def test_wait_until_healthy_gives_up(self):
        with patch.object(fp.requests, "get", side_effect=requests.Timeout("slow")), \
             patch.object(fp.time, "sleep"):
            assert fp.wait_until_healthy("http://svc", attempts=2, delay=0) is False


# ── Database ──────────────────────────────────────────────────────────────────
class TestStoreRows:
    def test_schema_then_upsert_then_commit(self):
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        rows = [fp.build_row(t, {"probability": 0.1, "prediction": 0}, 0) for t in TXS]

        assert fp.store_rows(conn, rows) == 2

        assert cur.execute.call_count == len(fp.SCHEMA_SQL)
        sql, params = cur.executemany.call_args.args
        assert "ON CONFLICT (transaction_id) DO UPDATE" in sql
        assert params == rows
        conn.commit.assert_called_once()


# ── Batch runner ──────────────────────────────────────────────────────────────
class TestScoreBatch:
    @pytest.fixture()
    def sb(self):
        return _load_score_batch()

    def test_run_end_to_end(self, sb):
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        health = {"input_columns": ["amt", "category"]}
        labels = {"labels": [{"trans_num": "a", "is_fraud": 0}, {"trans_num": "b", "is_fraud": 1}]}

        def fake_get(url, **kwargs):
            if url.endswith("/health"):
                return _response(health)
            if url.endswith("/v1/transactions"):
                return _response({"transactions": TXS})
            if url.endswith("/v1/labels"):
                return _response(labels)
            raise AssertionError(url)

        preds = {"predictions": [{"probability": 0.0, "prediction": 0},
                                 {"probability": 0.8, "prediction": 1}]}
        with patch.object(fp.requests, "get", side_effect=fake_get), \
             patch.object(fp.requests, "post", return_value=_response(preds)) as post:
            written = sb.run(2, "http://tx", "http://model", "postgresql://x",
                             connect=lambda uri: conn)

        assert written == 2
        # Payloads are aligned to the model schema (no id, no event_time, no label).
        assert post.call_args.kwargs["json"][1] == {"data": {"amt": 900.0, "category": None}}
        rows = cur.executemany.call_args.args[1]
        assert [(r[0], r[4], r[5]) for r in rows] == [("a", 0, 0), ("b", 1, 1)]
        conn.close.assert_called_once()

    def test_run_fails_when_api_never_wakes(self, sb):
        with patch.object(fp, "wait_until_healthy", return_value=False):
            with pytest.raises(RuntimeError, match="did not wake up"):
                sb.run(1, "http://tx", "http://model", "postgresql://x", connect=MagicMock())

    def test_main_requires_db_uri(self, sb, monkeypatch):
        monkeypatch.delenv("NEON_DB_URI", raising=False)
        assert sb.main(["--n", "1"]) == 2

    def test_dry_run_writes_nothing(self, sb, monkeypatch):
        monkeypatch.delenv("NEON_DB_URI", raising=False)
        with patch.object(sb, "run", return_value=0) as run:
            assert sb.main(["--n", "3", "--dry-run"]) == 0
        assert run.call_args.kwargs["db_uri"] is None
