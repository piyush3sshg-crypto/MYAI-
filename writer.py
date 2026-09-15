import random

from core.tokenizee import Tokenizer, split_sentences

SENTENCE_START = "<START>"
SENTENCE_END = "<END>"


class MarkovChainModel:
    def __init__(self, order=2):
        self.order = order
        self.transitions = {}
        self.starters = []
        self.raw_tokenizer = Tokenizer(strip_stopwords=False, apply_stemming=False)

    def _tokenize_sentence(self, sentence):
        return self.raw_tokenizer.process(sentence)

    def train(self, text):
        sentences = split_sentences(text)
        for sentence in sentences:
            words = self._tokenize_sentence(sentence)
            if len(words) < self.order + 1:
                continue
            padded = [SENTENCE_START] * self.order + words + [SENTENCE_END]
            starter_key = tuple(padded[0:self.order])
            self.starters.append(starter_key)
            for i in range(len(padded) - self.order):
                key = tuple(padded[i:i + self.order])
                next_word = padded[i + self.order]
                self.transitions.setdefault(key, []).append(next_word)

    def _weighted_choice(self, candidates):
        return random.choice(candidates)

    def generate(self, max_words=40, seed=None):
        if not self.transitions:
            return ""
        rng = random.Random(seed)
        if self.starters:
            state = rng.choice(self.starters)
        else:
            state = rng.choice(list(self.transitions.keys()))
        output_words = []
        for _ in range(max_words):
            candidates = self.transitions.get(state)
            if not candidates:
                break
            next_word = rng.choice(candidates)
            if next_word == SENTENCE_END:
                state = tuple(list(state[1:]) + [SENTENCE_START]) if self.order > 0 else state
                if len(output_words) > 0 and output_words[-1] not in (".", "!", "?"):
                    output_words.append(".")
                continue
            output_words.append(next_word)
            state = tuple(list(state[1:]) + [next_word])
        return self._detokenize(output_words)

    def generate_starting_with(self, prompt_words, max_words=40, seed=None):
        if not self.transitions:
            return ""
        rng = random.Random(seed)
        prompt_words = list(prompt_words)[-self.order:]
        while len(prompt_words) < self.order:
            prompt_words = [SENTENCE_START] + prompt_words
        state = tuple(prompt_words)
        output_words = list(prompt_words)
        output_words = [w for w in output_words if w != SENTENCE_START]
        for _ in range(max_words):
            candidates = self.transitions.get(state)
            if not candidates:
                fallback_keys = [k for k in self.transitions if k[0] == state[-1]]
                if not fallback_keys:
                    break
                state = rng.choice(fallback_keys)
                candidates = self.transitions.get(state)
                if not candidates:
                    break
            next_word = rng.choice(candidates)
            if next_word == SENTENCE_END:
                if output_words and output_words[-1] not in (".", "!", "?"):
                    output_words.append(".")
                break
            output_words.append(next_word)
            state = tuple(list(state[1:]) + [next_word])
        return self._detokenize(output_words)

    def _detokenize(self, words):
        if not words:
            return ""
        text = ""
        no_space_before = {".", ",", "!", "?", ";", ":"}
        for i, word in enumerate(words):
            if i == 0:
                text += word
            elif word in no_space_before:
                text += word
            else:
                text += " " + word
        if text and text[-1] not in (".", "!", "?"):
            text += "."
        return text[0].upper() + text[1:] if text else text


class TopicWriter:
    def __init__(self, order=2):
        self.order = order
        self.topic_models = {}

    def learn(self, topic_name, text):
        model = self.topic_models.setdefault(topic_name, MarkovChainModel(order=self.order))
        model.train(text)
        return model

    def known_topics(self):
        return list(self.topic_models.keys())

    def write_about(self, topic_name, max_words=40, seed=None):
        if topic_name not in self.topic_models:
            return None
        return self.topic_models[topic_name].generate(max_words=max_words, seed=seed)

    def continue_writing(self, topic_name, prompt_text, max_words=40, seed=None):
        if topic_name not in self.topic_models:
            return None
        model = self.topic_models[topic_name]
        prompt_words = model._tokenize_sentence(prompt_text)
        return model.generate_starting_with(prompt_words, max_words=max_words, seed=seed)

    def merge_topics_into(self, target_topic, source_topics):
        combined_text_parts = []
        for source in source_topics:
            if source in self.topic_models:
                pass
        return target_topic
