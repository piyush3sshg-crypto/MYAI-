"""Evaluation metrics — the project docs specifically flagged that
nothing beyond raw accuracy was ever computed. These fill that gap
without pulling in scikit-learn.
"""


def accuracy(y_true, y_pred):
    if not y_true:
        return None
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    """Returns {actual_label: {predicted_label: count}}."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def precision_recall_f1(y_true, y_pred, labels=None):
    """Per-class precision/recall/F1, plus macro averages — the metrics
    that were entirely absent before (only training_accuracy existed)."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": tp + fn,
        }

    n = len(labels) or 1
    macro = {
        "precision": sum(v["precision"] for v in per_class.values()) / n,
        "recall": sum(v["recall"] for v in per_class.values()) / n,
        "f1": sum(v["f1"] for v in per_class.values()) / n,
    }
    return {"per_class": per_class, "macro_avg": macro}


def roc_auc_binary(y_true, y_score, positive_label=1):
    """ROC-AUC for binary classification via the rank-sum (Mann-Whitney)
    method — no external dependency needed.

    y_true: list of true labels (two distinct values)
    y_score: list of predicted scores/probabilities for the positive class
    """
    pairs = list(zip(y_true, y_score))
    positives = [s for t, s in pairs if t == positive_label]
    negatives = [s for t, s in pairs if t != positive_label]

    if not positives or not negatives:
        return None

    ranked = sorted(pairs, key=lambda ts: ts[1])
    ranks = {}
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    rank_sum_positive = sum(
        ranks[idx] for idx, (t, _) in enumerate(ranked) if t == positive_label
    )
    n_pos, n_neg = len(positives), len(negatives)
    auc = (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc
