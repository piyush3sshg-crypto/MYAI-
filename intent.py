from core.tokenizee import Tokenizer
from core.similarity import TfidfVectorizer, cosine_similarity, hybrid_text_similarity


class IntentExample:
    def __init__(self, intent_name, text, tokens, vector):
        self.intent_name = intent_name
        self.text = text
        self.tokens = tokens
        self.vector = vector


class IntentMatch:
    def __init__(self, intent_name, score, matched_example):
        self.intent_name = intent_name
        self.score = score
        self.matched_example = matched_example

    def __repr__(self):
        return f"IntentMatch({self.intent_name!r}, score={self.score:.4f})"


class IntentClassifier:
    def __init__(self, threshold=0.18, tokenizer=None):
        self.threshold = threshold
        self.tokenizer = tokenizer or Tokenizer(strip_stopwords=True, apply_stemming=True)
        self.vectorizer = TfidfVectorizer()
        self.examples = []
        self.entity_extractors = {}
        self._dirty = True

    def add_example(self, intent_name, text):
        tokens = self.tokenizer.process(text)
        self.examples.append((intent_name, text, tokens))
        self._dirty = True

    def add_examples(self, intent_name, texts):
        for text in texts:
            self.add_example(intent_name, text)

    def register_entity_extractor(self, intent_name, extractor_fn):
        self.entity_extractors[intent_name] = extractor_fn

    def _rebuild_index(self):
        tokenized_docs = [tokens for _, _, tokens in self.examples]
        self.vectorizer.fit(tokenized_docs)
        self.indexed_examples = [
            IntentExample(intent, text, tokens, self.vectorizer.transform(tokens))
            for intent, text, tokens in self.examples
        ]
        self._dirty = False

    def classify(self, text, top_k=3):
        if self._dirty:
            self._rebuild_index()
        if not self.indexed_examples:
            return []
        tokens = self.tokenizer.process(text)
        query_vector = self.vectorizer.transform(tokens)
        scored = []
        for example in self.indexed_examples:
            tfidf_score = cosine_similarity(query_vector, example.vector)
            hybrid_score = hybrid_text_similarity(tokens, example.tokens, self.vectorizer)
            combined = 0.6 * tfidf_score + 0.4 * hybrid_score
            scored.append(IntentMatch(example.intent_name, combined, example))
        scored.sort(key=lambda m: m.score, reverse=True)
        best_per_intent = {}
        for match in scored:
            if match.intent_name not in best_per_intent or match.score > best_per_intent[match.intent_name].score:
                best_per_intent[match.intent_name] = match
        ranked = sorted(best_per_intent.values(), key=lambda m: m.score, reverse=True)
        return ranked[:top_k]

    def predict(self, text):
        ranked = self.classify(text, top_k=1)
        if not ranked or ranked[0].score < self.threshold:
            return IntentMatch("unknown", ranked[0].score if ranked else 0.0, None)
        best = ranked[0]
        entities = {}
        extractor = self.entity_extractors.get(best.intent_name)
        if extractor is not None:
            entities = extractor(text)
        best.entities = entities
        return best


class SlotPattern:
    def __init__(self, name, candidates):
        self.name = name
        self.candidates = [c.lower() for c in candidates]

    def match(self, tokens):
        found = []
        lowered = [t.lower() for t in tokens]
        for candidate in self.candidates:
            candidate_tokens = candidate.split()
            n = len(candidate_tokens)
            for i in range(len(lowered) - n + 1):
                if lowered[i:i + n] == candidate_tokens:
                    found.append(candidate)
        return found


class SlotFiller:
    def __init__(self):
        self.slots = {}

    def add_slot(self, name, candidates):
        self.slots[name] = SlotPattern(name, candidates)

    def extract(self, tokens):
        result = {}
        for name, pattern in self.slots.items():
            matches = pattern.match(tokens)
            if matches:
                result[name] = matches
        return result

