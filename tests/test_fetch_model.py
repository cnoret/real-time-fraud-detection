"""Tests for model-api/scripts/fetch_model.py (scikit-learn version detection)."""

import gzip
import importlib.util
import pickle
import zlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "model-api" / "scripts" / "fetch_model.py"


@pytest.fixture(scope="module")
def fetch_model():
    spec = importlib.util.spec_from_file_location("fetch_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Estimator:
    def __init__(self, version):
        self._sklearn_version = version
        self.coef_ = [0.1, 0.2]


def _pickled(*versions):
    return pickle.dumps({"steps": [_Estimator(v) for v in versions]})


class TestDetectSklearnVersion:
    def test_plain_pickle(self, fetch_model):
        assert fetch_model.detect_sklearn_version(_pickled("1.7.1")) == "1.7.1"

    def test_pipeline_of_several_estimators(self, fetch_model):
        assert fetch_model.detect_sklearn_version(_pickled("1.7.2", "1.7.2")) == "1.7.2"

    @pytest.mark.parametrize("compress", [zlib.compress, gzip.compress])
    def test_compressed_pickle(self, fetch_model, compress):
        assert fetch_model.detect_sklearn_version(compress(_pickled("1.5.2"))) == "1.5.2"

    def test_no_version(self, fetch_model):
        assert fetch_model.detect_sklearn_version(pickle.dumps({"a": 1})) is None
