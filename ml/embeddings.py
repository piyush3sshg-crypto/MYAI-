"""Word embeddings — a real skip-gram model with negative sampling,
trained from scratch (gradient descent, sigmoid, no numpy), matching
the project's existing "hand-built neural nets" style in ai/neural.py.

Before this, the only text representation was TF-IDF/bag-of-words —
words with no semantic relationship to each other. Embeddings learn a
dense vector per word such that words used in similar contexts end up
with similar vectors (the classic "you shall know a word by the
company it keeps" idea).
"""
import math
import random

from ml.text_features import tokenize


def _sigmoid(x):
    x = max(-30, min(30, x))
    return 1.0 / (1.0 + math.exp(-x))


class WordEmbeddings:
    """Skip-gram with negative sampling — the same algorithm behind
    word2vec, implemented from scratch in pure Python.

    Usage:
        emb = WordEmbeddings(dim=16, window=2, negative_samples=5, seed=42)
        emb.fit(list_of_texts, epochs=20)
        emb.most_similar("free")   # -> [(word, similarity), ...]
        emb.vector("free")         # -> list[float]
    """

    def __init__(self, dim=16, window=2, negative_samples=5, learning_rate=0.05, seed=42):
        self.dim = dim
        self.window = window
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)
        self.vocabulary = []
        self.vocab_index = {}
        self.word_vectors = []    # "input" embeddings
        self.context_vectors = [] # "output" embeddings (used only during training)
        self.word_freq = {}
        self.fitted = False

    def _build_vocab(self, tokenized_docs):
        freq = {}
        for tokens in tokenized_docs:
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
        self.vocabulary = sorted(freq.keys())
        self.vocab_index = {w: i for i, w in enumerate(self.vocabulary)}
        self.word_freq = freq

    def _init_vectors(self):
        n = len(self.vocabulary)
        scale = 0.5 / self.dim
        self.word_vectors = [
            [self.rng.uniform(-scale, scale) for _ in range(self.dim)] for _ in range(n)
        ]
        self.context_vectors = [
            [self.rng.uniform(-scale, scale) for _ in range(self.dim)] for _ in range(n)
        ]

    def _negative_sample(self, exclude_idx):
        # frequency-weighted negative sampling (common words sampled more,
        # same idea as word2vec's unigram^0.75 table, simplified)
        idx = self.rng.randrange(len(self.vocabulary))
        tries = 0
        while idx == exclude_idx and tries < 5:
            idx = self.rng.randrange(len(self.vocabulary))
            tries += 1
        return idx

    def fit(self, texts, epochs=20, min_count=1):
        tokenized_docs = [tokenize(t) for t in texts]
        self._build_vocab(tokenized_docs)

        if min_count > 1:
            kept = {w for w, c in self.word_freq.items() if c >= min_count}
            self.vocabulary = sorted(kept)
            self.vocab_index = {w: i for i, w in enumerate(self.vocabulary)}

        self._init_vectors()
        if not self.vocabulary:
            self.fitted = True
            return self

        pairs = []
        for tokens in tokenized_docs:
            idxs = [self.vocab_index[t] for t in tokens if t in self.vocab_index]
            for center_pos, center_idx in enumerate(idxs):
                start = max(0, center_pos - self.window)
                end = min(len(idxs), center_pos + self.window + 1)
                for context_pos in range(start, end):
                    if context_pos == center_pos:
                        continue
                    pairs.append((center_idx, idxs[context_pos]))

        for _ in range(epochs):
            self.rng.shuffle(pairs)
            for center_idx, context_idx in pairs:
                center_vec = self.word_vectors[center_idx]

                # positive example: center should predict this real context word
                self._sgd_step(center_vec, self.context_vectors[context_idx], label=1.0)

                # negative examples: center should NOT predict random words
                for _ in range(self.negative_samples):
                    neg_idx = self._negative_sample(context_idx)
                    self._sgd_step(center_vec, self.context_vectors[neg_idx], label=0.0)

        self.fitted = True
        return self

    def _sgd_step(self, center_vec, context_vec, label):
        dot = sum(c * o for c, o in zip(center_vec, context_vec))
        pred = _sigmoid(dot)
        error = (pred - label) * self.learning_rate
        for i in range(self.dim):
            grad_center = error * context_vec[i]
            grad_context = error * center_vec[i]
            center_vec[i] -= grad_center
            context_vec[i] -= grad_context

    def vector(self, word):
        if not self.fitted:
            raise RuntimeError("call fit() before vector()")
        idx = self.vocab_index.get(word)
        return self.word_vectors[idx] if idx is not None else None

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
        norm_b = math.sqrt(sum(y * y for y in b)) or 1e-9
        return dot / (norm_a * norm_b)

    def most_similar(self, word, top_k=5):
        query_vec = self.vector(word)
        if query_vec is None:
            return []
        scored = []
        for other_word, idx in self.vocab_index.items():
            if other_word == word:
                continue
            sim = self._cosine(query_vec, self.word_vectors[idx])
            scored.append((other_word, sim))
        scored.sort(key=lambda ws: ws[1], reverse=True)
        return scored[:top_k]

    def sentence_vector(self, text):
        """Average word vectors — a simple but real way to turn a whole
        sentence into a single fixed-size embedding for use as model
        input, instead of raw TF-IDF counts."""
        tokens = tokenize(text)
        vectors = [self.vector(t) for t in tokens]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return [0.0] * self.dim
        return [sum(vs) / len(vectors) for vs in zip(*vectors)]

    def to_dict(self):
        return {
            "dim": self.dim,
            "vocabulary": self.vocabulary,
            "word_vectors": self.word_vectors,
        }

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            data = json.load(f)
        emb = cls(dim=data["dim"])
        emb.vocabulary = data["vocabulary"]
        emb.vocab_index = {w: i for i, w in enumerate(emb.vocabulary)}
        emb.word_vectors = data["word_vectors"]
        emb.fitted = True
        return emb
