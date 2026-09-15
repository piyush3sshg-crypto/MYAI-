"""Statistical rigor utilities — confidence intervals and repeated CV.

A single cross-validation run with one seed gives a point estimate that
can swing a lot on small datasets (see the ~38% vs ~100% training gap
found earlier). Repeating CV across seeds and reporting a confidence
interval, instead of one number, is standard practice this project was
missing entirely.
"""
import math
import random

from ml.cross_validation import cross_validate


def bootstrap_confidence_interval(values, n_resamples=1000, ci=0.95, seed=42):
    """Bootstrap CI for the mean of `values` — works for any metric
    (accuracy per fold, F1 per fold, etc.) without assuming normality,
    which matters when you only have a handful of fold scores.
    """
    if not values:
        return {"mean": None, "lower": None, "upper": None}
    if len(values) == 1:
        return {"mean": values[0], "lower": values[0], "upper": values[0]}

    rng = random.Random(seed)
    n = len(values)
    resample_means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(sample) / n)

    resample_means.sort()
    alpha = (1.0 - ci) / 2.0
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1.0 - alpha) * n_resamples) - 1
    upper_idx = min(upper_idx, n_resamples - 1)

    return {
        "mean": sum(values) / n,
        "lower": resample_means[lower_idx],
        "upper": resample_means[upper_idx],
        "std": math.sqrt(sum((v - sum(values) / n) ** 2 for v in values) / n),
    }


def repeated_cross_validate(dataset, train_fn, predict_fn, k=5, n_repeats=5,
                             base_seed=42, stratify_by="label", ci=0.95):
    """Run k-fold CV n_repeats times with different shuffles and report
    a confidence interval on the mean accuracy/F1, instead of a single
    fragile point estimate from one split.
    """
    all_accuracies = []
    all_f1s = []
    per_repeat = []

    for repeat_index in range(n_repeats):
        seed = base_seed + repeat_index
        result = cross_validate(
            dataset, train_fn, predict_fn, k=k, seed=seed, stratify_by=stratify_by,
        )
        if result["mean_accuracy"] is not None:
            all_accuracies.append(result["mean_accuracy"])
        all_f1s.append(result["mean_macro_f1"])
        per_repeat.append({"seed": seed, **{
            k2: v for k2, v in result.items() if k2 in ("mean_accuracy", "mean_macro_f1")
        }})

    return {
        "n_repeats": n_repeats,
        "k": k,
        "per_repeat": per_repeat,
        "accuracy_ci": bootstrap_confidence_interval(all_accuracies, ci=ci, seed=base_seed),
        "macro_f1_ci": bootstrap_confidence_interval(all_f1s, ci=ci, seed=base_seed),
    }


def compare_to_baseline(model_accuracies, baseline_accuracies, seed=42, n_resamples=1000):
    """Bootstrap estimate of (model - baseline) accuracy difference,
    with a CI. If the CI for the difference includes 0, the model's
    apparent edge over the baseline isn't statistically convincing at
    this sample size.
    """
    if len(model_accuracies) != len(baseline_accuracies) or not model_accuracies:
        return None

    rng = random.Random(seed)
    n = len(model_accuracies)
    diffs = [m - b for m, b in zip(model_accuracies, baseline_accuracies)]

    resample_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(sample) / n)
    resample_means.sort()

    mean_diff = sum(diffs) / n
    lower = resample_means[int(0.025 * n_resamples)]
    upper = resample_means[min(int(0.975 * n_resamples), n_resamples - 1)]

    return {
        "mean_diff": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "significant": lower > 0 or upper < 0,
    }
