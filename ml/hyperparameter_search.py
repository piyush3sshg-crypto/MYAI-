"""Hyperparameter tuning via grid search — the project only ever used
hardcoded values (hidden_size=16, epochs=400, ...) with no systematic
search for what actually works best. This fills that gap.
"""
import itertools

from ml.cross_validation import cross_validate


def grid_search(dataset, param_grid, model_factory, predict_fn, k=5, seed=42, stratify_by="label"):
    """Try every combination of hyperparameters in param_grid, score
    each with k-fold cross-validation, and return results sorted best
    to worst.

    param_grid: dict of {param_name: [values_to_try]}
    model_factory: fn(train_dataset, **params) -> trained model
                    (i.e. your train_fn, but accepting the swept params)
    predict_fn: fn(model, record) -> predicted_label

    Example:
        grid_search(
            dataset,
            param_grid={"hidden_size": [8, 16, 32], "epochs": [40, 100]},
            model_factory=lambda train_ds, hidden_size, epochs: train_intent_classifier(
                train_ds, hidden_size=hidden_size, epochs=epochs,
            ),
            predict_fn=predict_intent,
        )
    """
    keys = list(param_grid.keys())
    value_combinations = list(itertools.product(*param_grid.values()))

    results = []
    for combo in value_combinations:
        params = dict(zip(keys, combo))

        def train_fn(train_dataset, _params=params):
            return model_factory(train_dataset, **_params)

        cv_result = cross_validate(dataset, train_fn, predict_fn, k=k, seed=seed, stratify_by=stratify_by)
        results.append({
            "params": params,
            "mean_accuracy": cv_result["mean_accuracy"],
            "mean_macro_f1": cv_result["mean_macro_f1"],
        })

    results.sort(key=lambda r: (r["mean_accuracy"] is not None, r["mean_accuracy"]), reverse=True)
    return results


def best_params(grid_search_results):
    return grid_search_results[0]["params"] if grid_search_results else None
