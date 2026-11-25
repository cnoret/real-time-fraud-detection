"""Tests for the self-hosted Transactions API (transactions-api/app.py)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
from fastapi.testclient import TestClient  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "transactions-api" / "app.py"


@pytest.fixture(scope="module")
def tx_api():
    spec = importlib.util.spec_from_file_location("transactions_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["transactions_app"] = module
    spec.loader.exec_module(module)
    module.RATE_LIMIT_PER_MINUTE = 10_000
    return module


@pytest.fixture(scope="module")
def client(tx_api):
    return TestClient(tx_api.app)


class TestDataset:
    def test_holdout_size_and_fraud_rate(self, tx_api):
        df = tx_api.TRANSACTIONS
        assert len(df) == 50_000
        assert df["trans_num"].is_unique
        assert 0.002 < df["is_fraud"].mean() < 0.006

    def test_features_exclude_label(self, tx_api):
        assert "is_fraud" not in tx_api.FEATURES.columns


class TestHealth:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["transactions"] == 50_000
        assert "is_fraud" not in body["columns"]


class TestTransactions:
    def test_single_transaction_has_model_fields(self, client):
        body = client.get("/v1/transactions").json()
        assert body["count"] == 1
        tx = body["transactions"][0]
        for col in ["trans_num", "amt", "merchant", "unix_time", "trans_date_trans_time",
                    "event_time"]:
            assert col in tx
        assert "is_fraud" not in tx

    def test_batch(self, client):
        assert client.get("/v1/transactions", params={"n": 25}).json()["count"] == 25

    @pytest.mark.parametrize("n", [0, 101])
    def test_batch_bounds(self, client, n):
        assert client.get("/v1/transactions", params={"n": n}).status_code == 422

    def test_seed_is_reproducible(self, client):
        a = client.get("/v1/transactions", params={"n": 3, "seed": 7}).json()
        b = client.get("/v1/transactions", params={"n": 3, "seed": 7}).json()
        ids = lambda r: [t["trans_num"] for t in r["transactions"]]  # noqa: E731
        assert ids(a) == ids(b)


class TestLabels:
    def test_label_matches_dataset(self, client, tx_api):
        row = tx_api.TRANSACTIONS.iloc[0]
        body = client.get("/v1/labels", params={"trans_num": row["trans_num"]}).json()
        assert body["labels"] == [{"trans_num": row["trans_num"], "is_fraud": int(row["is_fraud"])}]
        assert body["unknown"] == []

    def test_unknown_id(self, client):
        body = client.get("/v1/labels", params={"trans_num": "nope"}).json()
        assert body == {"labels": [], "unknown": ["nope"]}

    def test_requires_id(self, client):
        assert client.get("/v1/labels").status_code == 422

    def test_fraud_labels_exist(self, client, tx_api):
        frauds = tx_api.TRANSACTIONS.loc[tx_api.TRANSACTIONS["is_fraud"] == 1, "trans_num"]
        body = client.get("/v1/labels", params={"trans_num": list(frauds[:5])}).json()
        assert all(item["is_fraud"] == 1 for item in body["labels"])


class TestLegacyEndpoint:
    def test_orient_split_contract(self, client):
        raw = client.get("/current-transactions").json()  # JSON-encoded string
        data = json.loads(raw)
        assert set(data) >= {"columns", "data"}
        assert len(data["data"]) == 1
        cols = data["columns"]
        assert "current_time" in cols
        for dropped in ["is_fraud", "unix_time", "trans_date_trans_time"]:
            assert dropped not in cols


class TestRateLimit:
    def test_returns_429_when_exceeded(self, tx_api):
        tx_api.RATE_LIMIT_PER_MINUTE = 2
        tx_api._hits.clear()
        c = TestClient(tx_api.app)
        codes = [c.get("/v1/transactions").status_code for _ in range(3)]
        tx_api.RATE_LIMIT_PER_MINUTE = 10_000
        tx_api._hits.clear()
        assert codes == [200, 200, 429]
