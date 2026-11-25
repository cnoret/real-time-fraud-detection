"""Train, evaluate and export the fraud detection model, reproducibly.

Data: the simulated "Sparkov" card transactions (Kaggle kartik2112/fraud-detection).
- fraudTrain.csv  (2019-01-01 -> 2020-06-21) is the only data the model learns from.
- fraudTest.csv   (2020-06-21 -> 2020-12-31) is never touched until the final
  evaluation. The Transactions API replays a sample of it (the holdout), so the
  live stream is unseen data too.

Protocol (no information from the test period reaches the model):
1. Split fraudTrain in time: everything before VALID_START trains a candidate,
   the last months validate it (early stopping + threshold choice).
2. The decision threshold is the one that maximises F1 on that validation period.
3. Refit on the whole of fraudTrain with the number of boosting iterations
   found in step 1, keep the threshold, and score fraudTest once.

Outputs (in --out, default training/artifacts/):
- fraud_model.pkl   scikit-learn Pipeline (features -> encoder -> classifier)
- model_card.json   version, threshold, metrics, data fingerprints

Usage:
    python training/train.py --train data/fraudTrain.csv --test data/fraudTest.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model-api"))

import fraud_features as ff  # noqa: E402

MODEL_VERSION = "model-v2.0.0"
TARGET = "is_fraud"
VALID_START = pd.Timestamp("2020-03-01")
SEED = 42
HOLDOUT_PATH = ROOT / "transactions-api" / "data" / "transactions_holdout.parquet"

# Columns a model must never see (identifiers, personal data, label).
FORBIDDEN = {"trans_num", "cc_num", "first", "last", "street", "zip", "merchant", "unix_time", TARGET}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _features_step() -> FunctionTransformer:
    return FunctionTransformer(ff.build_features, feature_names_out=ff.feature_names_out)


def gradient_boosting(max_iter: int = 500, early_stopping: bool = True) -> Pipeline:
    encode = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                ff.CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", ff.NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features=list(range(len(ff.CATEGORICAL_FEATURES))),
        early_stopping=early_stopping,
        validation_fraction=0.1 if early_stopping else None,
        n_iter_no_change=30,
        scoring="average_precision",
        random_state=SEED,
    )
    return Pipeline([("features", _features_step()), ("encode", encode), ("clf", clf)])


def logistic_baseline() -> Pipeline:
    encode = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ff.CATEGORICAL_FEATURES),
            ("num", Pipeline([("fill", FunctionTransformer(lambda x: np.nan_to_num(x))), ("scale", StandardScaler())]),
             ff.NUMERIC_FEATURES),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("features", _features_step()), ("encode", encode), ("clf", clf)])


def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def operating_points(y_true, proba) -> dict[str, float]:
    """Named decision thresholds, all chosen on the validation period.

    - conservative: lowest threshold whose precision reaches 95 % (few false alerts)
    - balanced:     threshold that maximises F1 (the default decision)
    - sensitive:    highest threshold whose recall reaches 90 %
    - very_sensitive: highest threshold whose recall reaches 95 % (investigation mode)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]

    def lowest_with_precision(target: float) -> float:
        ok = np.flatnonzero(precision >= target)
        return float(thresholds[ok[0]]) if ok.size else float(thresholds[-1])

    def highest_with_recall(target: float) -> float:
        ok = np.flatnonzero(recall >= target)
        return float(thresholds[ok[-1]]) if ok.size else float(thresholds[0])

    points = {
        "conservative": lowest_with_precision(0.95),
        "balanced": best_f1_threshold(y_true, proba),
        "sensitive": highest_with_recall(0.90),
        "very_sensitive": highest_with_recall(0.95),
    }
    return {name: round(t, 4) for name, t in points.items()}


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true)),
        "frauds": int(np.sum(y_true)),
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "alerts": int(tp + fp),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", required=True, help="path to fraudTrain.csv")
    parser.add_argument("--test", required=True, help="path to fraudTest.csv")
    parser.add_argument("--out", default=str(ROOT / "training" / "artifacts"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train, test = load(args.train), load(args.test)
    leaked = set(ff.FEATURES) & FORBIDDEN
    assert not leaked, f"forbidden columns used as features: {leaked}"
    overlap = set(train["trans_num"]) & set(test["trans_num"])
    assert not overlap, f"{len(overlap)} transactions are in both train and test"
    assert train["trans_date_trans_time"].max() <= test["trans_date_trans_time"].min(), "train must precede test"

    holdout_ids = set(pd.read_parquet(HOLDOUT_PATH, columns=["trans_num"])["trans_num"])
    holdout_in_train = len(holdout_ids & set(train["trans_num"]))
    assert holdout_in_train == 0, f"{holdout_in_train} replayed transactions are in the training data"
    print(f"train {train.shape}, test {test.shape}, holdout {len(holdout_ids)} ids, none in train")

    # 1. Temporal validation inside fraudTrain
    is_valid = train["trans_date_trans_time"] >= VALID_START
    fit, valid = train[~is_valid], train[is_valid]
    X_fit, y_fit = fit[ff.INPUT_COLUMNS], fit[TARGET].to_numpy()
    X_val, y_val = valid[ff.INPUT_COLUMNS], valid[TARGET].to_numpy()

    candidates = {"hist_gradient_boosting": gradient_boosting(), "logistic_regression": logistic_baseline()}
    validation = {}
    for name, model in candidates.items():
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_val)[:, 1]
        threshold = best_f1_threshold(y_val, proba)
        validation[name] = evaluate(y_val, proba, threshold)
        print(f"[valid] {name}: {validation[name]}")

    n_iter = int(candidates["hist_gradient_boosting"].named_steps["clf"].n_iter_)
    proba_val = candidates["hist_gradient_boosting"].predict_proba(X_val)[:, 1]
    thresholds = operating_points(y_val, proba_val)
    threshold = thresholds["balanced"]
    print(f"chosen: hist_gradient_boosting, {n_iter} iterations, threshold {threshold}")

    # 2. Refit on all of fraudTrain, evaluate once on fraudTest
    X_train, y_train = train[ff.INPUT_COLUMNS], train[TARGET].to_numpy()
    X_test, y_test = test[ff.INPUT_COLUMNS], test[TARGET].to_numpy()

    final = gradient_boosting(max_iter=n_iter, early_stopping=False).fit(X_train, y_train)
    proba_test = final.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, proba_test, threshold)
    test_metrics_default = evaluate(y_test, proba_test, 0.5)
    test_by_threshold = {name: evaluate(y_test, proba_test, t) for name, t in thresholds.items()}

    baseline = logistic_baseline().fit(X_train, y_train)
    proba_base = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate(y_test, proba_base, validation["logistic_regression"]["threshold"])

    in_holdout = test["trans_num"].isin(holdout_ids).to_numpy()
    holdout_metrics = evaluate(y_test[in_holdout], proba_test[in_holdout], threshold)

    for name, m in (("test", test_metrics), ("test@0.5", test_metrics_default),
                    ("test baseline", baseline_metrics), ("replayed holdout", holdout_metrics)):
        print(f"[{name}] {m}")

    model_path = out / "fraud_model.pkl"
    joblib.dump(final, model_path, compress=3)

    card = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "estimator": "HistGradientBoostingClassifier",
        "n_iter": n_iter,
        "input_columns": ff.INPUT_COLUMNS,
        "features": ff.FEATURES,
        "excluded_columns": sorted(FORBIDDEN - {TARGET}),
        "threshold": threshold,
        "threshold_rule": f"maximises F1 on fraudTrain transactions from {VALID_START.date()} on (temporal validation)",
        "thresholds": thresholds,
        "model_sha256": sha256(str(model_path)),
        "data": {
            "train": {"file": Path(args.train).name, "sha256": sha256(args.train), "rows": int(len(train)),
                      "frauds": int(y_train.sum()),
                      "period": [str(train.trans_date_trans_time.min()), str(train.trans_date_trans_time.max())]},
            "test": {"file": Path(args.test).name, "sha256": sha256(args.test), "rows": int(len(test)),
                     "frauds": int(y_test.sum()),
                     "period": [str(test.trans_date_trans_time.min()), str(test.trans_date_trans_time.max())]},
            "replayed_holdout_rows_in_train": holdout_in_train,
        },
        "metrics": {
            "validation": validation,
            "test": test_metrics,
            "test_at_0_5": test_metrics_default,
            "test_by_threshold": test_by_threshold,
            "test_logistic_baseline": baseline_metrics,
            "replayed_holdout": holdout_metrics,
        },
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
        "training_seconds": round(time.time() - started),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"wrote {model_path} ({card['model_size_mb']} MB) and model_card.json in {card['training_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
