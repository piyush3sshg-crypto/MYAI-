"""Exploratory data analysis utilities — the layer the project docs
flagged as completely missing (data science maturity 4.5/10).

All functions take a data.dataset.Dataset (or plain list of dicts) and
return plain dicts, so results can be printed, logged, or fed straight
into ml/experiment_tracker.py.
"""
import math


def _is_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _as_float(value):
    return float(value)


def describe_column(records, column):
    """Descriptive stats for one column: count, missing, mean, std,
    min, max for numeric columns; value counts for categorical ones.
    """
    values = [r.get(column) for r in records]
    missing = sum(1 for v in values if v is None or v == "")
    present = [v for v in values if v is not None and v != ""]

    if present and all(_is_number(v) for v in present):
        numeric = [_as_float(v) for v in present]
        n = len(numeric)
        mean = sum(numeric) / n
        variance = sum((x - mean) ** 2 for x in numeric) / n if n > 0 else 0.0
        return {
            "column": column,
            "type": "numeric",
            "count": n,
            "missing": missing,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(numeric),
            "max": max(numeric),
        }

    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "column": column,
        "type": "categorical",
        "count": len(present),
        "missing": missing,
        "unique": len(counts),
        "top_values": ranked[:5],
    }


def describe(dataset):
    """describe() for every column — the pandas-style df.describe() the
    project was missing entirely."""
    records = list(dataset)
    columns = dataset.columns() if hasattr(dataset, "columns") else (
        sorted({k for r in records for k in r.keys()})
    )
    return {col: describe_column(records, col) for col in columns}


def missing_report(dataset):
    """Percentage of missing values per column — run this before
    trusting any downstream model."""
    records = list(dataset)
    total = len(records) or 1
    columns = dataset.columns() if hasattr(dataset, "columns") else (
        sorted({k for r in records for k in r.keys()})
    )
    report = {}
    for col in columns:
        missing = sum(1 for r in records if r.get(col) in (None, ""))
        report[col] = {"missing": missing, "pct": round(100 * missing / total, 2)}
    return report


def class_balance(dataset, label_column="label"):
    """Class distribution for a categorical target — flags imbalance
    that would otherwise silently skew training/evaluation."""
    records = list(dataset)
    counts = {}
    for r in records:
        label = r.get(label_column)
        counts[label] = counts.get(label, 0) + 1
    total = len(records) or 1
    balance = {
        label: {"count": count, "pct": round(100 * count / total, 2)}
        for label, count in counts.items()
    }
    if counts:
        largest = max(counts.values())
        smallest = min(counts.values())
        imbalance_ratio = round(largest / smallest, 2) if smallest else None
    else:
        imbalance_ratio = None
    return {"classes": balance, "imbalance_ratio": imbalance_ratio}


def correlation(dataset, column_a, column_b):
    """Pearson correlation between two numeric columns."""
    records = list(dataset)
    pairs = [
        (r.get(column_a), r.get(column_b)) for r in records
        if _is_number(r.get(column_a)) and _is_number(r.get(column_b))
    ]
    if len(pairs) < 2:
        return None

    xs = [_as_float(p[0]) for p in pairs]
    ys = [_as_float(p[1]) for p in pairs]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def correlation_matrix(dataset, numeric_columns):
    """Pairwise Pearson correlation for a list of numeric columns."""
    return {
        (a, b): correlation(dataset, a, b)
        for a in numeric_columns for b in numeric_columns
    }
