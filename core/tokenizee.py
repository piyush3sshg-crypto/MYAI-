import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "of", "to", "in", "on", "at", "for", "with", "by", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "and", "or", "but", "if",
    "then", "so", "than", "that", "this", "these", "those", "it", "its",
    "as", "do", "does", "did", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "not", "no", "nor",
}

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

VOWELS = set("aeiou")

STEP1A_SUFFIXES = [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")]
STEP1B_SUFFIXES = ["eed", "ed", "ing"]
DOUBLE_CONSONANT_ENDINGS = {"bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"}


def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return WORD_PATTERN.findall(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def _measure(stem):
    pattern = "".join("V" if ch in VOWELS else "C" for ch in stem)
    pattern = re.sub(r"CC+", "C", pattern)
    pattern = re.sub(r"VV+", "V", pattern)
    return pattern.count("VC")


def _contains_vowel(stem):
    return any(ch in VOWELS for ch in stem)


def _ends_double_consonant(stem):
    return len(stem) >= 2 and stem[-2:] in DOUBLE_CONSONANT_ENDINGS


def _ends_cvc(stem):
    if len(stem) < 3:
        return False
    c1, v, c2 = stem[-3], stem[-2], stem[-1]
    if c1 in VOWELS or v not in VOWELS or c2 in VOWELS:
        return False
    if c2 in ("w", "x", "y"):
        return False
    return True


def stem_word(word):
    if len(word) <= 2:
        return word
    stem = word
    for suffix, replacement in STEP1A_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)] + replacement
            break
    if stem.endswith("eed"):
        if _measure(stem[:-3]) > 0:
            stem = stem[:-1]
    else:
        step1b_applied = False
        for suffix in ("ed", "ing"):
            if stem.endswith(suffix) and _contains_vowel(stem[: -len(suffix)]):
                stem = stem[: -len(suffix)]
                step1b_applied = True
                break
        if step1b_applied:
            if stem.endswith(("at", "bl", "iz")):
                stem = stem + "e"
            elif _ends_double_consonant(stem) and not stem.endswith(("l", "s", "z")):
                stem = stem[:-1]
            elif _measure(stem) == 1 and _ends_cvc(stem):
                stem = stem + "e"
    if stem.endswith("y") and _contains_vowel(stem[:-1]):
        stem = stem[:-1] + "i"
    return stem


def stem_tokens(tokens):
    return [stem_word(t) for t in tokens]


def normalize_text(text, strip_stopwords=False, apply_stemming=False):
    tokens = tokenize(text)
    if strip_stopwords:
        tokens = remove_stopwords(tokens)
    if apply_stemming:
        tokens = stem_tokens(tokens)
    return tokens


def char_ngrams(word, n=3):
    padded = f"^{word}$"
    if len(padded) < n:
        return [padded]
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


class Tokenizer:
    def __init__(self, strip_stopwords=False, apply_stemming=False):
        self.strip_stopwords = strip_stopwords
        self.apply_stemming = apply_stemming

    def process(self, text):
        return normalize_text(
            text,
            strip_stopwords=self.strip_stopwords,
            apply_stemming=self.apply_stemming,
        )

    def process_sentences(self, text):
        return [self.process(s) for s in split_sentences(text)]

