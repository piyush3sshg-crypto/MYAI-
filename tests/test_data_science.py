"""Tests for the new data/, analytics/, and ml/ layer."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import Dataset
from data.sample_data import load_intent_dataset
from analytics.eda import describe, class_balance, missing_report, correlation
from ml.metrics import accuracy, precision_recall_f1, confusion_matrix, roc_auc_binary
from ml.cross_validation import cross_validate


def test_dataset_train_test_split_ratio():
    records = [{"x": i, "label": "a" if i % 2 == 0 else "b"} for i in range(50)]
    ds = Dataset(records, name="toy")
    train, test = ds.train_test_split(test_ratio=0.2, seed=1)
    assert len(test) == 10
    assert len(train) == 40
    assert len(train) + len(test) == len(ds)


def test_dataset_stratified_split_preserves_classes():
    records = [{"x": i, "label": "rare"} for i in range(3)]
    records += [{"x": i, "label": "common"} for i in range(3, 30)]
    ds = Dataset(records, name="toy")
    train, test = ds.train_test_split(test_ratio=0.2, seed=1, stratify_by="label")
    test_labels = {r["label"] for r in test}
    # the rare class must still appear in the test split, not be starved out
    assert "rare" in test_labels


def test_dataset_k_folds_cover_every_record_exactly_once_as_validation():
    records = [{"x": i, "label": "a"} for i in range(20)]
    ds = Dataset(records, name="toy")
    seen_val_x = []
    for train_fold, val_fold in ds.k_folds(k=5, seed=1):
        seen_val_x.extend(r["x"] for r in val_fold)
    assert sorted(seen_val_x) == list(range(20))


def test_intent_dataset_loads_and_has_expected_columns():
    ds = load_intent_dataset()
    assert len(ds) > 0
    assert set(ds.columns()) == {"text", "label"}


def test_class_balance_reports_imbalance_ratio():
    ds = load_intent_dataset()
    report = class_balance(ds, label_column="label")
    assert report["imbalance_ratio"] is not None
    assert sum(c["count"] for c in report["classes"].values()) == len(ds)


def test_missing_report_flags_no_missing_on_clean_dataset():
    ds = load_intent_dataset()
    report = missing_report(ds)
    assert report["text"]["missing"] == 0
    assert report["label"]["missing"] == 0


def test_describe_distinguishes_numeric_and_categorical():
    ds = Dataset([{"age": 20, "city": "delhi"}, {"age": 30, "city": "pune"}], name="toy")
    report = describe(ds)
    assert report["age"]["type"] == "numeric"
    assert report["city"]["type"] == "categorical"
    assert report["age"]["mean"] == 25.0


def test_correlation_perfect_positive():
    ds = Dataset([{"a": i, "b": i * 2} for i in range(10)], name="toy")
    r = correlation(ds, "a", "b")
    assert abs(r - 1.0) < 1e-9


def test_precision_recall_f1_perfect_predictions():
    y_true = ["cat", "dog", "cat", "dog"]
    y_pred = ["cat", "dog", "cat", "dog"]
    result = precision_recall_f1(y_true, y_pred)
    assert result["macro_avg"]["f1"] == 1.0


def test_confusion_matrix_shape():
    y_true = ["a", "b", "a"]
    y_pred = ["a", "a", "a"]
    matrix = confusion_matrix(y_true, y_pred, labels=["a", "b"])
    assert matrix["b"]["a"] == 1
    assert matrix["a"]["a"] == 2


def test_roc_auc_binary_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    auc = roc_auc_binary(y_true, y_score, positive_label=1)
    assert auc == 1.0


def test_cross_validate_runs_and_returns_bounded_metrics():
    ds = load_intent_dataset()

    def train_fn(train_ds):
        # trivial "model": majority-class baseline, keeps the test fast
        counts = {}
        for r in train_ds:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        return max(counts, key=counts.get)

    def predict_fn(model, record):
        return model

    result = cross_validate(ds, train_fn, predict_fn, k=5, seed=1, stratify_by="label")
    assert result["mean_accuracy"] is not None
    assert 0.0 <= result["mean_accuracy"] <= 1.0
