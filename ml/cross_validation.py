"""K-fold cross-validation — generic over any model, so it can wrap
NeuralIntentClassifier, DecisionStumpLearner, or anything future models
are added later.
"""
from data.dataset import Dataset
from ml.metrics import accuracy, precision_recall_f1


def cross_validate(dataset, train_fn, predict_fn, k=5, seed=42, stratify_by="label"):
    """Run k-fold cross-validation.

    train_fn(train_dataset) -> model
    predict_fn(model, record) -> predicted_label

    Returns per-fold metrics plus an overall summary, so results can be
    logged directly via ml.experiment_tracker.
    """
    fold_results = []

    for fold_index, (train_fold, val_fold) in enumerate(
        dataset.k_folds(k=k, seed=seed, stratify_by=stratify_by)
    ):
        model = train_fn(train_fold)

        y_true = [r[stratify_by] for r in val_fold]
        y_pred = [predict_fn(model, r) for r in val_fold]

        fold_accuracy = accuracy(y_true, y_pred)
        fold_prf1 = precision_recall_f1(y_true, y_pred)

        fold_results.append({
            "fold": fold_index,
            "train_size": len(train_fold),
            "val_size": len(val_fold),
            "accuracy": fold_accuracy,
            "macro_f1": fold_prf1["macro_avg"]["f1"],
        })

    accuracies = [f["accuracy"] for f in fold_results if f["accuracy"] is not None]
    f1_scores = [f["macro_f1"] for f in fold_results]

    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None

    return {
        "k": k,
        "folds": fold_results,
        "mean_accuracy": mean_accuracy,
        "mean_macro_f1": mean_f1,
    }
