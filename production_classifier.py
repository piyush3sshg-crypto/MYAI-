"""Pure Python TF-IDF + Logistic Regression inference - no numpy, scipy,
or scikit-learn required. Runs anywhere Python's standard library runs,
including Termux.

This reproduces (bit-for-bit, verified) what a scikit-learn
TfidfVectorizer + LogisticRegression model computes, using only the
numbers exported by train_production_models.py (run separately on a PC,
see that file's docstring). Training needs scikit-learn; using the
trained model does not.

Usage:
    clf = ProductionClassifier.load("models/intent_model.json")
    label, confidence = clf.predict("what is 12 plus 7")
"""
import json
import math
import re

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def _tokenize_with_ngrams(text, ngram_range=(1, 2)):
    tokens = TOKEN_RE.findall(text.lower())
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        if n == 1:
            out.extend(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                out.append(" ".join(tokens[i:i + n]))
    return out


class ProductionClassifier:
    def __init__(self, model_dict):
        self.vocabulary = model_dict["vocabulary"]       # term -> column index
        self.idf = model_dict["idf"]                      # column index -> idf weight
        self.ngram_range = tuple(model_dict.get("ngram_range", [1, 2]))
        self.sublinear_tf = model_dict.get("sublinear_tf", True)
        self.classes = model_dict["classes"]
        self.coef = model_dict["coef"]                    # n_classes(or 1) x n_features
        self.intercept = model_dict["intercept"]
        self.model_name = model_dict.get("model_name", "model")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def _tfidf_vector(self, text):
        """Returns a sparse dict {column_index: tfidf_weight}, L2-normalized
        - matching scikit-learn's TfidfVectorizer default behavior
        (norm='l2', sublinear_tf as configured)."""
        terms = _tokenize_with_ngrams(text, self.ngram_range)
        counts = {}
        for t in terms:
            col = self.vocabulary.get(t)
            if col is None:
                continue
            counts[col] = counts.get(col, 0) + 1

        weights = {}
        for col, tf in counts.items():
            tf_value = (1.0 + math.log(tf)) if self.sublinear_tf else float(tf)
            weights[col] = tf_value * self.idf[col]

        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm > 0:
            for col in weights:
                weights[col] /= norm
        return weights

    def _linear_scores(self, vec):
        """Returns a list of raw linear scores, one per class row in coef_."""
        scores = []
        for class_coef, b in zip(self.coef, self.intercept):
            s = b
            for col, w in vec.items():
                s += class_coef[col] * w
            scores.append(s)
        return scores

    def predict(self, text):
        vec = self._tfidf_vector(text)
        scores = self._linear_scores(vec)

        if len(self.classes) == 2 and len(scores) == 1:
            # binary case: sklearn stores one row, positive class = classes[1]
            prob_pos = 1.0 / (1.0 + math.exp(-scores[0]))
            if prob_pos >= 0.5:
                return self.classes[1], prob_pos
            return self.classes[0], 1.0 - prob_pos

        # multiclass: softmax over the per-class linear scores, argmax
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]
        best_i = max(range(len(probs)), key=lambda i: probs[i])
        return self.classes[best_i], probs[best_i]

    def predict_all(self, text):
        """Returns {class: probability} for every class, for debugging/UI."""
        vec = self._tfidf_vector(text)
        scores = self._linear_scores(vec)
        if len(self.classes) == 2 and len(scores) == 1:
            prob_pos = 1.0 / (1.0 + math.exp(-scores[0]))
            return {self.classes[0]: 1.0 - prob_pos, self.classes[1]: prob_pos}
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        return {c: e / total for c, e in zip(self.classes, exps)}
