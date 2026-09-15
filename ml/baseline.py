"""Baseline models — the sanity check every real evaluation needs.

If your fancy neural classifier doesn't clearly beat a majority-class
guess, its accuracy number is not evidence of anything. This was
missing entirely before: main.py only ever reported the neural model's
own accuracy, with nothing to compare it against.
"""
import math
import random

from ml.text_features import TfidfVectorizer


class LogisticRegressionBaseline:
    """One-vs-rest logistic regression over TF-IDF features, trained by
    plain gradient descent. Pure Python — no numpy/sklearn — but a real
    linear model, so it's a meaningfully stronger baseline than a
    constant/random guess.
    """

    def __init__(self, epochs=60, learning_rate=0.5, ngram_range=(1, 1)):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1)
        self.weights = {}   # label -> list of weights, aligned to vocabulary
        self.bias = {}      # label -> float
        self.labels = []

    def fit(self, dataset, label_column="label", text_column="text"):
        texts = [r[text_column] for r in dataset]
        labels = [r[label_column] for r in dataset]
        self.vectorizer.fit(texts)
        self.labels = sorted(set(labels))
        vocab_size = len(self.vectorizer.vocabulary)

        vectors = [self.vectorizer.transform_dense(t) for t in texts]

        for label in self.labels:
            weights = [0.0] * vocab_size
            bias = 0.0
            targets = [1.0 if l == label else 0.0 for l in labels]

            for _ in range(self.epochs):
                for vec, target in zip(vectors, targets):
                    z = bias + sum(w * x for w, x in zip(weights, vec))
                    pred = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
                    error = pred - target
                    for i, x in enumerate(vec):
                        if x != 0.0:
                            weights[i] -= self.learning_rate * error * x
                    bias -= self.learning_rate * error

            self.weights[label] = weights
            self.bias[label] = bias
        return self

    def predict(self, record, text_column="text"):
        vec = self.vectorizer.transform_dense(record[text_column])
        best_label, best_score = None, float("-inf")
        for label in self.labels:
            z = self.bias[label] + sum(w * x for w, x in zip(self.weights[label], vec))
            if z > best_score:
                best_score, best_label = z, label
        return best_label


class KNNBaseline:
    """k-nearest-neighbors over TF-IDF cosine similarity — a simple,
    non-parametric baseline that needs no training loop at all."""

    def __init__(self, k=3, ngram_range=(1, 1)):
        self.k = k
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1)
        self.train_vectors = []
        self.train_labels = []

    def fit(self, dataset, label_column="label", text_column="text"):
        texts = [r[text_column] for r in dataset]
        self.train_labels = [r[label_column] for r in dataset]
        self.vectorizer.fit(texts)
        self.train_vectors = [self.vectorizer.transform(t) for t in texts]
        return self

    @staticmethod
    def _cosine(a, b):
        common = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (norm_a * norm_b)

    def predict(self, record, text_column="text"):
        query = self.vectorizer.transform(record[text_column])
        scored = [
            (self._cosine(query, vec), label)
            for vec, label in zip(self.train_vectors, self.train_labels)
        ]
        scored.sort(key=lambda sl: sl[0], reverse=True)
        top_k = scored[:self.k] if scored else []
        if not top_k:
            return None
        votes = {}
        for _, label in top_k:
            votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get)





class MajorityClassBaseline:
    """Always predicts the most frequent class seen during training."""

    def __init__(self):
        self.majority_label = None

    def fit(self, dataset, label_column="label"):
        counts = {}
        for record in dataset:
            label = record[label_column]
            counts[label] = counts.get(label, 0) + 1
        self.majority_label = max(counts, key=counts.get) if counts else None
        return self

    def predict(self, record):
        return self.majority_label


class StratifiedRandomBaseline:
    """Predicts a random class, weighted by training-set class frequency.

    Slightly harder to beat than the majority baseline, and the metric
    scikit-learn's DummyClassifier(strategy='stratified') reports."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.labels = []
        self.weights = []

    def fit(self, dataset, label_column="label"):
        counts = {}
        for record in dataset:
            label = record[label_column]
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values()) or 1
        self.labels = list(counts.keys())
        self.weights = [counts[l] / total for l in self.labels]
        return self

    def predict(self, record):
        if not self.labels:
            return None
        return self.rng.choices(self.labels, weights=self.weights, k=1)[0]
