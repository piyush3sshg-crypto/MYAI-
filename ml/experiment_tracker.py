"""Minimal experiment tracker — logs dataset version, hyperparameters,
and metrics for every training run to a local JSON file.

This is deliberately dependency-free (no MLflow/W&B) so it works inside
Termux with zero setup, while still answering the question the project
docs raised: "which dataset/params produced this model?"
"""
import json
import os
import time


class ExperimentTracker:
    def __init__(self, log_path="ml/experiments.json"):
        self.log_path = log_path
        self._entries = self._load()

    def _load(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, encoding="utf-8") as handle:
                try:
                    return json.load(handle)
                except json.JSONDecodeError:
                    return []
        return []

    def _save(self):
        directory = os.path.dirname(self.log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as handle:
            json.dump(self._entries, handle, indent=2)

    def log_run(self, experiment_name, dataset_name, dataset_version, params, metrics, notes=""):
        entry = {
            "experiment": experiment_name,
            "dataset": dataset_name,
            "dataset_version": dataset_version,
            "params": params,
            "metrics": metrics,
            "notes": notes,
            "timestamp": time.time(),
            "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._entries.append(entry)
        self._save()
        return entry

    def history(self, experiment_name=None):
        if experiment_name is None:
            return list(self._entries)
        return [e for e in self._entries if e["experiment"] == experiment_name]

    def best_run(self, experiment_name, metric_key, higher_is_better=True):
        runs = self.history(experiment_name)
        scored = [r for r in runs if r["metrics"].get(metric_key) is not None]
        if not scored:
            return None
        return max(
            scored,
            key=lambda r: r["metrics"][metric_key] if higher_is_better else -r["metrics"][metric_key],
        )
