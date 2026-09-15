"""Tests for the Priority-2/3/4 additions: TF-IDF features, baselines,
statistical utilities, and SVG chart generation.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.sample_data import load_intent_dataset, load_spam_dataset
from ml.text_features import TfidfVectorizer, tokenize, feature_importance_by_class
from ml.baseline import MajorityClassBaseline, StratifiedRandomBaseline
from ml.stats import bootstrap_confidence_interval, repeated_cross_validate, compare_to_baseline
from ml.visualize import bar_chart_svg, heatmap_svg, line_chart_svg


def test_tokenize_unigrams():
    tokens = tokenize("What is 7 plus 8?")
    assert tokens == ["what", "is", "7", "plus", "8"]


def test_tokenize_bigrams_included():
    tokens = tokenize("hi there", ngram_range=(1, 2))
    assert "hi" in tokens and "there" in tokens
    assert "hi_there" in tokens


def test_tfidf_vectorizer_fit_transform():
    texts = ["the cat sat", "the dog sat", "the cat ran"]
    vec = TfidfVectorizer().fit(texts)
    weights = vec.transform("the cat sat")
    assert "cat" in weights
    # "the" appears in every doc -> lowest idf among these tokens
    assert weights["the"] < weights["cat"]


def test_tfidf_top_terms_excludes_unseen_words():
    vec = TfidfVectorizer().fit(["hello world", "goodbye world"])
    top = vec.top_terms("hello completely_unknown_token", k=5)
    terms = [t for t, _ in top]
    assert "hello" in terms
    assert "completely_unknown_token" not in terms


def test_feature_importance_by_class_returns_all_labels():
    ds = load_intent_dataset()
    fi = feature_importance_by_class(ds, top_k=5)
    assert set(fi.keys()) == {"greeting", "farewell", "math_query", "family_query", "weather_query"}
    for label, terms in fi.items():
        assert len(terms) <= 5


def test_majority_baseline_predicts_most_common_label():
    ds = load_spam_dataset()
    baseline = MajorityClassBaseline().fit(ds, label_column="label")
    assert baseline.majority_label == "ham"  # 90 ham vs 70 spam
    assert baseline.predict({"text": "anything"}) == "ham"


def test_stratified_baseline_only_predicts_seen_labels():
    ds = load_spam_dataset()
    baseline = StratifiedRandomBaseline(seed=1).fit(ds, label_column="label")
    predictions = {baseline.predict({}) for _ in range(50)}
    assert predictions.issubset({"ham", "spam"})


def test_bootstrap_ci_contains_the_mean():
    values = [0.8, 0.82, 0.79, 0.81, 0.83]
    ci = bootstrap_confidence_interval(values, n_resamples=500, seed=1)
    assert ci["lower"] <= ci["mean"] <= ci["upper"]


def test_bootstrap_ci_single_value_is_degenerate():
    ci = bootstrap_confidence_interval([0.9])
    assert ci["mean"] == ci["lower"] == ci["upper"] == 0.9


def test_compare_to_baseline_detects_clear_improvement():
    model_accs = [0.95, 0.96, 0.94, 0.97, 0.95]
    baseline_accs = [0.55, 0.56, 0.54, 0.55, 0.57]
    result = compare_to_baseline(model_accs, baseline_accs, seed=1)
    assert result["significant"] is True
    assert result["mean_diff"] > 0.3


def test_compare_to_baseline_no_difference_is_not_significant():
    model_accs = [0.70, 0.71, 0.69, 0.70, 0.72]
    baseline_accs = [0.70, 0.69, 0.71, 0.70, 0.68]
    result = compare_to_baseline(model_accs, baseline_accs, seed=1)
    assert result["significant"] is False


def test_bar_chart_svg_is_well_formed():
    svg = bar_chart_svg(["a", "b"], [3, 7], title="t")
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "<rect" in svg


def test_heatmap_svg_is_well_formed():
    matrix = {"a": {"a": 5, "b": 1}, "b": {"a": 0, "b": 6}}
    svg = heatmap_svg(matrix, ["a", "b"], ["a", "b"], title="t")
    assert svg.startswith("<svg")
    assert svg.count("<rect") >= 4  # background + 4 cells (at least 4)


def test_line_chart_svg_is_well_formed():
    svg = line_chart_svg([5, 4, 3, 2, 1], title="loss")
    assert svg.startswith("<svg")
    assert "<polyline" in svg


def test_repeated_cross_validate_returns_confidence_interval():
    ds = load_spam_dataset()

    def train_fn(train_ds):
        return MajorityClassBaseline().fit(train_ds, label_column="label")

    def predict_fn(model, record):
        return model.predict(record)

    result = repeated_cross_validate(ds, train_fn, predict_fn, k=5, n_repeats=3, base_seed=1, stratify_by="label")
    assert result["accuracy_ci"]["lower"] <= result["accuracy_ci"]["mean"] <= result["accuracy_ci"]["upper"]
