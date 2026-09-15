"""Generates a single markdown report + SVG charts from a dataset and a
trained/evaluated model — the "proper data scientist deliverable" that
was entirely absent before (the project only ever printed numbers to
the terminal).
"""
import os

from analytics.eda import describe, class_balance, missing_report
from ml.metrics import confusion_matrix
from ml.visualize import bar_chart_svg, heatmap_svg, save_svg


def generate_report(dataset, label_column, cv_result, repeated_cv_result,
                     baseline_comparison, feature_importance, y_true_for_confusion,
                     y_pred_for_confusion, output_dir, report_title="Model Report"):
    os.makedirs(output_dir, exist_ok=True)

    balance = class_balance(dataset, label_column=label_column)
    missing = missing_report(dataset)

    # --- charts ---
    labels = list(balance["classes"].keys())
    counts = [balance["classes"][l]["count"] for l in labels]
    balance_svg_path = os.path.join(output_dir, "class_balance.svg")
    save_svg(bar_chart_svg(labels, counts, title="Class balance", value_format="{:.0f}"), balance_svg_path)

    conf_labels = sorted(set(y_true_for_confusion) | set(y_pred_for_confusion))
    matrix = confusion_matrix(y_true_for_confusion, y_pred_for_confusion, labels=conf_labels)
    confusion_svg_path = os.path.join(output_dir, "confusion_matrix.svg")
    save_svg(
        heatmap_svg(matrix, conf_labels, conf_labels, title="Confusion matrix (held-out fold)"),
        confusion_svg_path,
    )

    fold_accs = [f["accuracy"] for f in cv_result["folds"] if f["accuracy"] is not None]
    fold_labels = [f"fold {f['fold']}" for f in cv_result["folds"]]
    fold_svg_path = os.path.join(output_dir, "fold_accuracy.svg")
    save_svg(bar_chart_svg(fold_labels, fold_accs, title="Accuracy per fold"), fold_svg_path)

    # --- markdown ---
    lines = []
    lines.append(f"# {report_title}\n")
    lines.append(f"**Dataset:** `{dataset.name}` (version `{dataset.version}`), {len(dataset)} rows\n")

    lines.append("## 1. Data overview\n")
    lines.append(f"- Rows: {len(dataset)}")
    lines.append(f"- Columns: {', '.join(dataset.columns())}")
    lines.append(f"- Class imbalance ratio (largest/smallest): {balance['imbalance_ratio']}")
    for col, info in missing.items():
        lines.append(f"- Missing in `{col}`: {info['missing']} ({info['pct']}%)")
    lines.append("\n![Class balance](class_balance.svg)\n")

    lines.append("## 2. Class distribution\n")
    lines.append("| Class | Count | % |")
    lines.append("|---|---|---|")
    for label, info in balance["classes"].items():
        lines.append(f"| {label} | {info['count']} | {info['pct']}% |")
    lines.append("")

    lines.append("## 3. Top distinguishing terms per class (TF-IDF)\n")
    for label, terms in feature_importance.items():
        term_str = ", ".join(f"`{t}` ({w:.2f})" for t, w in terms)
        lines.append(f"- **{label}**: {term_str}")
    lines.append("")

    lines.append("## 4. Cross-validation results\n")
    lines.append(f"- Single run (k={cv_result['k']}): "
                  f"mean accuracy = {cv_result['mean_accuracy']:.3f}, "
                  f"mean macro-F1 = {cv_result['mean_macro_f1']:.3f}")
    acc_ci = repeated_cv_result["accuracy_ci"]
    f1_ci = repeated_cv_result["macro_f1_ci"]
    lines.append(
        f"- Repeated CV ({repeated_cv_result['n_repeats']} repeats, k={repeated_cv_result['k']}): "
        f"accuracy = {acc_ci['mean']:.3f} "
        f"(95% CI [{acc_ci['lower']:.3f}, {acc_ci['upper']:.3f}])"
    )
    lines.append(
        f"- Repeated CV macro-F1 = {f1_ci['mean']:.3f} "
        f"(95% CI [{f1_ci['lower']:.3f}, {f1_ci['upper']:.3f}])"
    )
    lines.append("\n![Accuracy per fold](fold_accuracy.svg)\n")

    lines.append("## 5. Comparison against baseline\n")
    if baseline_comparison:
        verdict = "statistically meaningful" if baseline_comparison["significant"] else "NOT statistically distinguishable from chance/baseline at this sample size"
        lines.append(
            f"- Model accuracy − baseline accuracy = {baseline_comparison['mean_diff']:.3f} "
            f"(95% CI [{baseline_comparison['ci_lower']:.3f}, {baseline_comparison['ci_upper']:.3f}])"
        )
        lines.append(f"- **Verdict:** the improvement over the baseline is {verdict}.")
    else:
        lines.append("- No baseline comparison available.")
    lines.append("")

    lines.append("## 6. Confusion matrix (held-out fold)\n")
    lines.append("![Confusion matrix](confusion_matrix.svg)\n")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    return {
        "report_path": report_path,
        "charts": [balance_svg_path, confusion_svg_path, fold_svg_path],
    }
