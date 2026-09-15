"""Lightweight dataset container for MYAI's data science layer.

No external dependencies (no pandas/numpy) — everything is plain Python
lists/dicts, matching the rest of the MYAI codebase.
"""
import csv
import json
import os
import random


class Dataset:
    """A list of records (dicts) with a name and optional version tag.

    A "record" is just {"column_name": value, ...}. This mirrors how the
    rest of MYAI represents facts (predicate/args tuples) — simple,
    inspectable, serializable.
    """

    def __init__(self, records, name="unnamed", version="v1"):
        self.records = list(records)
        self.name = name
        self.version = version

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @classmethod
    def from_csv(cls, path, name=None, version="v1"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset CSV not found: {path}")
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                records = [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            raise ValueError(f"could not decode {path} as UTF-8 CSV: {exc}") from exc
        if not records:
            raise ValueError(f"dataset CSV is empty (no data rows): {path}")
        return cls(records, name=name or path, version=version)

    @classmethod
    def from_json(cls, path, name=None, version="v1"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset JSON not found: {path}")
        try:
            with open(path, encoding="utf-8") as handle:
                records = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"dataset JSON is malformed: {path} ({exc})") from exc
        if isinstance(records, dict):
            records = [records]
        if not records:
            raise ValueError(f"dataset JSON is empty: {path}")
        return cls(records, name=name or path, version=version)

    def to_csv(self, path):
        if not self.records:
            raise ValueError("cannot write an empty dataset to csv")
        fieldnames = list(self.records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2)

    def columns(self):
        if not self.records:
            return []
        keys = []
        seen = set()
        for record in self.records:
            for key in record.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def column(self, name):
        return [record.get(name) for record in self.records]

    def filter(self, predicate):
        return Dataset(
            [r for r in self.records if predicate(r)],
            name=self.name, version=self.version,
        )

    def map(self, fn):
        return Dataset([fn(r) for r in self.records], name=self.name, version=self.version)

    def train_test_split(self, test_ratio=0.2, seed=42, stratify_by=None):
        """Split into (train, test) Datasets.

        If stratify_by is given (a column name), the split preserves the
        proportion of each class in that column — important for small,
        imbalanced datasets like intent-classification examples, where a
        plain random split can accidentally starve a class of test rows.
        """
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("test_ratio must be between 0 and 1")

        rng = random.Random(seed)

        if stratify_by is None:
            shuffled = list(self.records)
            rng.shuffle(shuffled)
            cut = int(len(shuffled) * test_ratio)
            test_records = shuffled[:cut]
            train_records = shuffled[cut:]
        else:
            buckets = {}
            for record in self.records:
                key = record.get(stratify_by)
                buckets.setdefault(key, []).append(record)
            train_records, test_records = [], []
            for key, bucket in buckets.items():
                rng.shuffle(bucket)
                cut = max(1, int(len(bucket) * test_ratio)) if len(bucket) > 1 else 0
                test_records.extend(bucket[:cut])
                train_records.extend(bucket[cut:])
            rng.shuffle(train_records)
            rng.shuffle(test_records)

        return (
            Dataset(train_records, name=f"{self.name}_train", version=self.version),
            Dataset(test_records, name=f"{self.name}_test", version=self.version),
        )

    def k_folds(self, k=5, seed=42, stratify_by=None):
        """Yield (train_fold, val_fold) Dataset pairs for k-fold cross-validation."""
        if k < 2:
            raise ValueError("k must be >= 2")

        rng = random.Random(seed)

        if stratify_by is None:
            indices = list(range(len(self.records)))
            rng.shuffle(indices)
            folds = [indices[i::k] for i in range(k)]
        else:
            buckets = {}
            for i, record in enumerate(self.records):
                buckets.setdefault(record.get(stratify_by), []).append(i)
            folds = [[] for _ in range(k)]
            for key, idxs in buckets.items():
                rng.shuffle(idxs)
                for i, idx in enumerate(idxs):
                    folds[i % k].append(idx)

        for i in range(k):
            val_idx = set(folds[i])
            train_records = [r for j, r in enumerate(self.records) if j not in val_idx]
            val_records = [self.records[j] for j in folds[i]]
            yield (
                Dataset(train_records, name=f"{self.name}_fold{i}_train", version=self.version),
                Dataset(val_records, name=f"{self.name}_fold{i}_val", version=self.version),
            )

    def summary(self):
        return {
            "name": self.name,
            "version": self.version,
            "rows": len(self.records),
            "columns": self.columns(),
        }

    def __repr__(self):
        return f"Dataset(name={self.name!r}, version={self.version!r}, rows={len(self.records)})"
