"""End-to-end data science pipeline demo.

Loads the spam_dataset.csv (160 rows, 2 well-separated classes — chosen
over the 5-class intent set specifically because it's large/clean
enough to make cross-validation, baseline comparison, and confidence
intervals actually meaningful), then runs:

    load -> EDA -> TF-IDF feature importance -> cross-validation
    -> repeated CV with confidence intervals -> baseline comparison
    -> confusion matrix -> markdown + SVG report

Run with: python3 data_science_report.py
Output:   reports/spam_classifier/report.md (+ 3 .svg charts)
"""
from data.sample_data import load_spam_dataset
from analytics.eda import describe, class_balance, missing_report
from ml.text_features import feature_importance_by_class
from ml.cross_validation import cross_validate
from ml.stats import repeated_cross_validate, compare_to_baseline
from ml.baseline import MajorityClassBaseline
from ml.metrics import accuracy
from ml.experiment_tracker import ExperimentTracker
from ml.report import generate_report
from ai.neural import NeuralIntentClassifier


def train_fn(train_dataset):
    classifier = NeuralIntentClassifier(hidden_size=8, seed=42)
    for record in train_dataset:
        classifier.add_example(record["label"], record["text"])
    classifier.train(epochs=40, learning_rate=0.2, validation_split=0.0)
    return classifier


def predict_fn(model, record):
    return model.predict(record["text"]).intent_name


def baseline_train_fn(train_dataset):
    return MajorityClassBaseline().fit(train_dataset, label_column="label")


def baseline_predict_fn(model, record):
    return model.predict(record)


def per_repeat_accuracies(dataset, train_fn, predict_fn, k, n_repeats, base_seed, stratify_by):
    """Same repeats/seeds as ml.stats.repeated_cross_validate, but
    returns the raw per-repeat mean accuracy list (needed to pair up
    with the baseline's per-repeat accuracies for compare_to_baseline).
    """
    accs = []
    for i in range(n_repeats):
        seed = base_seed + i
        result = cross_validate(dataset, train_fn, predict_fn, k=k, seed=seed, stratify_by=stratify_by)
        accs.append(result["mean_accuracy"])
    return accs


def main():
    print("=== Data Science Pipeline: spam_dataset ===\n")

    dataset = load_spam_dataset()
    print(f"Loaded {dataset.name} version {dataset.version}: {len(dataset)} rows")

    print("\n--- EDA ---")
    balance = class_balance(dataset, label_column="label")
    missing = missing_report(dataset)
    print("class balance:", balance)
    print("missing report:", missing)

    print("\n--- Feature importance (TF-IDF) ---")
    fi = feature_importance_by_class(dataset, ngram_range=(1, 1), top_k=6)
    for label, terms in fi.items():
        print(f"  {label}: {[t for t, _ in terms]}")

    print("\n--- Cross-validation (single run, k=5) ---")
    cv_result = cross_validate(dataset, train_fn, predict_fn, k=5, seed=42, stratify_by="label")
    print(f"  mean_accuracy={cv_result['mean_accuracy']:.3f}  mean_macro_f1={cv_result['mean_macro_f1']:.3f}")

    print("\n--- Repeated cross-validation (5 repeats, with 95% CI) ---")
    repeated = repeated_cross_validate(
        dataset, train_fn, predict_fn, k=5, n_repeats=3, base_seed=42, stratify_by="label",
    )
    acc_ci = repeated["accuracy_ci"]
    print(f"  accuracy = {acc_ci['mean']:.3f}  95% CI [{acc_ci['lower']:.3f}, {acc_ci['upper']:.3f}]")

    print("\n--- Baseline comparison (vs majority-class) ---")
    model_accs = per_repeat_accuracies(dataset, train_fn, predict_fn, k=5, n_repeats=3, base_seed=42, stratify_by="label")
    baseline_accs = per_repeat_accuracies(dataset, baseline_train_fn, baseline_predict_fn, k=5, n_repeats=3, base_seed=42, stratify_by="label")
    comparison = compare_to_baseline(model_accs, baseline_accs)
    print(f"  model - baseline = {comparison['mean_diff']:.3f}  "
          f"95% CI [{comparison['ci_lower']:.3f}, {comparison['ci_upper']:.3f}]  "
          f"significant={comparison['significant']}")

    print("\n--- Confusion matrix (fold 0 of a fresh 5-fold split) ---")
    train_fold, val_fold = next(dataset.k_folds(k=5, seed=99, stratify_by="label"))
    model = train_fn(train_fold)
    y_true = [r["label"] for r in val_fold]
    y_pred = [predict_fn(model, r) for r in val_fold]

    tracker = ExperimentTracker(log_path="ml/experiments.json")
    tracker.log_run(
        experiment_name="spam_classifier_full_pipeline",
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        params={"hidden_size": 8, "epochs": 40, "learning_rate": 0.2, "k": 5, "n_repeats": 3},
        metrics={
            "cv_mean_accuracy": cv_result["mean_accuracy"],
            "repeated_cv_accuracy_mean": acc_ci["mean"],
            "repeated_cv_accuracy_ci_lower": acc_ci["lower"],
            "repeated_cv_accuracy_ci_upper": acc_ci["upper"],
            "baseline_diff": comparison["mean_diff"],
            "baseline_diff_significant": comparison["significant"],
        },
        notes="full pipeline run via data_science_report.py",
    )

    print("\n--- Generating report ---")
    result = generate_report(
        dataset=dataset,
        label_column="label",
        cv_result=cv_result,
        repeated_cv_result=repeated,
        baseline_comparison=comparison,
        feature_importance=fi,
        y_true_for_confusion=y_true,
        y_pred_for_confusion=y_pred,
        output_dir="reports/spam_classifier",
        report_title="Spam Classifier — Model Report",
    )
    print(f"  report written to {result['report_path']}")
    print(f"  charts: {result['charts']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys
        print(f"\n[FATAL] data science pipeline crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
