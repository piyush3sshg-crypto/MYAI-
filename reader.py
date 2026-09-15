from core.tokenizee import Tokenizer, split_sentences
from core.similarity import TfidfVectorizer, cosine_similarity


class DocumentReader:
    def __init__(self):
        self.tokenizer = Tokenizer(strip_stopwords=True, apply_stemming=True)
        self.sentences = []
        self.sentence_tokens = []
        self.vectorizer = TfidfVectorizer()
        self.sentence_vectors = []
        self.loaded = False

    def load(self, text):
        self.sentences = split_sentences(text)
        self.sentence_tokens = [self.tokenizer.process(s) for s in self.sentences]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.sentence_tokens)
        self.sentence_vectors = [self.vectorizer.transform(t) for t in self.sentence_tokens]
        self.loaded = bool(self.sentences)
        return self.loaded

    def _similarity_matrix(self):
        n = len(self.sentences)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                matrix[i][j] = cosine_similarity(self.sentence_vectors[i], self.sentence_vectors[j])
        return matrix

    def _text_rank_scores(self, damping=0.85, iterations=30):
        n = len(self.sentences)
        if n == 0:
            return []
        matrix = self._similarity_matrix()
        row_sums = [sum(matrix[i]) or 1e-12 for i in range(n)]
        scores = [1.0 / n] * n
        for _ in range(iterations):
            new_scores = [0.0] * n
            for i in range(n):
                incoming = 0.0
                for j in range(n):
                    if i == j or matrix[j][i] == 0.0:
                        continue
                    incoming += (matrix[j][i] / row_sums[j]) * scores[j]
                new_scores[i] = (1 - damping) / n + damping * incoming
            scores = new_scores
        return scores

    def summarize(self, num_sentences=3):
        if not self.loaded or len(self.sentences) == 0:
            return ""
        if len(self.sentences) <= num_sentences:
            return " ".join(self.sentences)
        scores = self._text_rank_scores()
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked_indices[:num_sentences])
        return " ".join(self.sentences[i] for i in top_indices)

    def key_sentences(self, num_sentences=5):
        if not self.loaded:
            return []
        scores = self._text_rank_scores()
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.sentences[i], scores[i]) for i in ranked_indices[:num_sentences]]

    def ask(self, question, top_k=1):
        if not self.loaded:
            return []
        question_tokens = self.tokenizer.process(question)
        question_vector = self.vectorizer.transform(question_tokens)
        scored = []
        for i, sentence_vector in enumerate(self.sentence_vectors):
            score = cosine_similarity(question_vector, sentence_vector)
            scored.append((score, self.sentences[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def keyword_frequency(self, top_k=10):
        from collections import Counter
        counter = Counter()
        for tokens in self.sentence_tokens:
            counter.update(tokens)
        return counter.most_common(top_k)

    def word_count(self):
        return sum(len(tokens) for tokens in self.sentence_tokens)

    def sentence_count(self):
        return len(self.sentences)
