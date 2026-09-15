"""Tests for Priority-4 additions: stronger baselines, hyperparameter
search, model persistence, and dataset error handling.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.sample_data import load_spam_dataset
from data.dataset import Dataset
from ml.baseline import LogisticRegressionBaseline, KNNBaseline, MajorityClassBaseline
from ml.hyperparameter_search import grid_search, best_params
from ai.neural import NeuralIntentClassifier


def test_logistic_regression_beats_majority_baseline():
    ds = load_spam_dataset()
    train, test = ds.train_test_split(test_ratio=0.25, seed=1, stratify_by="label")

    lr = LogisticRegressionBaseline(epochs=40, learning_rate=0.5).fit(train)
    lr_correct = sum(1 for r in test if lr.predict(r) == r["label"])

    maj = MajorityClassBaseline().fit(train, label_column="label")
    maj_correct = sum(1 for r in test if maj.predict(r) == r["label"])

    assert lr_correct >= maj_correct


def test_knn_baseline_predicts_valid_labels():
    ds = load_spam_dataset()
    train, test = ds.train_test_split(test_ratio=0.2, seed=1, stratify_by="label")
    knn = KNNBaseline(k=3).fit(train)
    predictions = {knn.predict(r) for r in test}
    assert predictions.issubset({"ham", "spam"})


def test_grid_search_returns_sorted_results():
    ds = load_spam_dataset()

    def model_factory(train_ds, hidden_size):
        clf = NeuralIntentClassifier(hidden_size=hidden_size, seed=42)
        for r in train_ds:
            clf.add_example(r["label"], r["text"])
        clf.train(epochs=15, learning_rate=0.3, validation_split=0.0)
        return clf

    def predict_fn(model, record):
        return model.predict(record["text"]).intent_name

    results = grid_search(
        ds, param_grid={"hidden_size": [4, 8]},
        model_factory=model_factory, predict_fn=predict_fn,
        k=3, seed=1,
    )
    assert len(results) == 2
    # sorted best-first
    assert results[0]["mean_accuracy"] >= results[1]["mean_accuracy"]
    assert best_params(results) in ({"hidden_size": 4}, {"hidden_size": 8})


def test_model_persistence_save_and_load_roundtrip(tmp_path_str="/tmp/myai_test_model.json"):
    clf = NeuralIntentClassifier(hidden_size=8, seed=42)
    clf.add_example("greeting", "hi there")
    clf.add_example("farewell", "goodbye")
    clf.train(epochs=20, learning_rate=0.3, validation_split=0.0)

    clf.save(tmp_path_str)
    reloaded = NeuralIntentClassifier.load(tmp_path_str)

    original_pred = clf.predict("hi there").intent_name
    reloaded_pred = reloaded.predict("hi there").intent_name
    assert original_pred == reloaded_pred

    os.remove(tmp_path_str)


def test_dataset_from_csv_missing_file_raises_clear_error():
    try:
        Dataset.from_csv("/tmp/definitely_does_not_exist_myai.csv")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e)


def test_dataset_from_json_malformed_raises_clear_error():
    bad_path = "/tmp/myai_bad_json_test.json"
    with open(bad_path, "w") as f:
        f.write("{not valid json,,,")
    try:
        Dataset.from_json(bad_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "malformed" in str(e)
    finally:
        os.remove(bad_path)
