"""Text feature engineering — TF-IDF and n-grams.

The classifier so far only used raw bag-of-words counts (see
ai/neural.py's _vectorize). TF-IDF down-weights common words (like
"the", "is") and up-weights words that are distinctive for a class,
which is standard practice and was completely missing before.

Pure Python (no numpy) so this stays portable to Termux.
"""
import math
import re


def tokenize(text, ngram_range=(1, 1)):
    """Lowercase word tokens, optionally extended with n-grams.

    ngram_range=(1, 2) returns unigrams + bigrams, e.g. "what is" ->
    ["what", "is", "what_is"].
    """
    words = re.findall(r"[a-zA-Z']+|\d+", text.lower())
    tokens = []
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        if n == 1:
            tokens.extend(words)
        else:
            for i in range(len(words) - n + 1):
                tokens.append("_".join(words[i:i + n]))
    return tokens


class TfidfVectorizer:
    """Fit on a list of texts, transform texts into TF-IDF weight dicts.

    Usage:
        vec = TfidfVectorizer(ngram_range=(1, 2))
        vec.fit(list_of_texts)
        weights = vec.transform("some new text")   # {token: tfidf_weight}
        vector = vec.transform_dense("some new text")  # list aligned to vec.vocabulary
    """

    def __init__(self, ngram_range=(1, 1), min_df=1):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vocabulary = []
        self.vocab_index = {}
        self.idf = {}
        self.fitted = False

    def fit(self, texts):
        doc_token_sets = [set(tokenize(t, self.ngram_range)) for t in texts]
        doc_freq = {}
        for tokens in doc_token_sets:
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        n_docs = len(texts)
        kept = sorted(t for t, df in doc_freq.items() if df >= self.min_df)
        self.vocabulary = kept
        self.vocab_index = {t: i for i, t in enumerate(kept)}
        # smoothed idf, same formula scikit-learn uses by default
        self.idf = {
            t: math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0
            for t in kept
        }
        self.fitted = True
        return self

    def _term_frequencies(self, text):
        tokens = tokenize(text, self.ngram_range)
        if not tokens:
            return {}
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        total = len(tokens)
        return {token: count / total for token, count in counts.items()}

    def transform(self, text):
        """Sparse representation: {token: tfidf_weight}, vocabulary-only tokens."""
        if not self.fitted:
            raise RuntimeError("call fit() before transform()")
        tf = self._term_frequencies(text)
        return {
            token: freq * self.idf[token]
            for token, freq in tf.items()
            if token in self.idf
        }

    def transform_dense(self, text):
        """Dense vector aligned to self.vocabulary — feed this straight
        into ai.neural.NeuralIntentClassifier-style models."""
        weights = self.transform(text)
        return [weights.get(token, 0.0) for token in self.vocabulary]

    def top_terms(self, text, k=5):
        """The k highest-weighted terms for a piece of text — useful
        for explaining why a document got classified a certain way."""
        weights = self.transform(text)
        return sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]


def feature_importance_by_class(dataset, label_column="label", text_column="text", ngram_range=(1, 1), top_k=10):
    """For each class, the terms with the highest mean TF-IDF weight —
    a lightweight stand-in for proper feature-importance analysis
    (chi-square / mutual information) that needs no external library.
    """
    texts = [r[text_column] for r in dataset]
    labels = [r[label_column] for r in dataset]

    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=2)
    vectorizer.fit(texts)

    sums_by_class = {}
    counts_by_class = {}
    for text, label in zip(texts, labels):
        weights = vectorizer.transform(text)
        bucket = sums_by_class.setdefault(label, {})
        for token, w in weights.items():
            bucket[token] = bucket.get(token, 0.0) + w
        counts_by_class[label] = counts_by_class.get(label, 0) + 1

    result = {}
    for label, sums in sums_by_class.items():
        n = counts_by_class[label]
        means = {token: total / n for token, total in sums.items()}
        result[label] = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return result
