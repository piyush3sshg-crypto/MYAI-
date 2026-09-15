import math
from collections import Counter


def levenshtein_distance(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_levenshtein(a, b):
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


def jaccard_similarity(set_a, set_b):
    set_a, set_b = set(set_a), set(set_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def ngrams(sequence, n):
    sequence = list(sequence)
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def ngram_overlap(a, b, n=2):
    ngrams_a = ngrams(a, n)
    ngrams_b = ngrams(b, n)
    return jaccard_similarity(ngrams_a, ngrams_b)


def cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    keys = set(vec_a.keys()) | set(vec_b.keys())
    dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product_dense(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))


def cosine_similarity_dense(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = dot_product_dense(vec_a, vec_b)
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TfidfVectorizer:
    def __init__(self):
        self.document_frequency = Counter()
        self.total_documents = 0
        self.vocabulary = set()
        self.fitted = False

    def fit(self, tokenized_documents):
        self.document_frequency = Counter()
        self.total_documents = len(tokenized_documents)
        for tokens in tokenized_documents:
            unique_tokens = set(tokens)
            self.vocabulary |= unique_tokens
            for token in unique_tokens:
                self.document_frequency[token] += 1
        self.fitted = True
        return self

    def partial_fit(self, tokens):
        unique_tokens = set(tokens)
        self.total_documents += 1
        self.vocabulary |= unique_tokens
        for token in unique_tokens:
            self.document_frequency[token] += 1
        self.fitted = True

    def idf(self, term):
        df = self.document_frequency.get(term, 0)
        return math.log((self.total_documents + 1) / (df + 1)) + 1.0

    def transform(self, tokens):
        term_freq = Counter(tokens)
        total_terms = sum(term_freq.values()) or 1
        vector = {}
        for term, count in term_freq.items():
            tf = count / total_terms
            vector[term] = tf * self.idf(term)
        return vector

    def fit_transform(self, tokenized_documents):
        self.fit(tokenized_documents)
        return [self.transform(tokens) for tokens in tokenized_documents]


def weighted_score(components, weights):
    total_weight = sum(weights) or 1.0
    return sum(c * w for c, w in zip(components, weights)) / total_weight


def hybrid_text_similarity(tokens_a, tokens_b, vectorizer=None):
    scores = []
    weights = []
    scores.append(jaccard_similarity(tokens_a, tokens_b))
    weights.append(1.0)
    scores.append(ngram_overlap(tokens_a, tokens_b, 2))
    weights.append(1.0)
    if vectorizer is not None and vectorizer.fitted:
        vec_a = vectorizer.transform(tokens_a)
        vec_b = vectorizer.transform(tokens_b)
        scores.append(cosine_similarity(vec_a, vec_b))
        weights.append(2.0)
    return weighted_score(scores, weights)

