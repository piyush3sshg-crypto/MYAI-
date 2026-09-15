import re

PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}
PREPOSITIONS = {
    "in", "on", "at", "by", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "of", "off", "over", "under", "to", "for",
}
CONJUNCTIONS = {"and", "or", "but", "nor", "so", "yet", "because", "although", "if", "when", "while"}
AUX_VERBS = {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would", "shall", "should", "can", "could", "may", "might", "must"}
WH_WORDS = {"what", "why", "how", "when", "where", "who", "whom", "which", "whose"}

VERB_SUFFIXES = ("ing", "ed", "ize", "ise", "ify", "ate")
ADVERB_SUFFIXES = ("ly",)
ADJECTIVE_SUFFIXES = ("ful", "ous", "ive", "able", "ible", "al", "ic", "less")
NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "er", "or", "ism", "ship")

NUMBER_PATTERN = re.compile(r"^\d+(\.\d+)?$")


def tag_token(token, index, tokens):
    lower = token.lower()
    if NUMBER_PATTERN.match(token):
        return "NUM"
    if lower in WH_WORDS:
        return "WH"
    if lower in PRONOUNS:
        return "PRON"
    if lower in DETERMINERS:
        return "DET"
    if lower in PREPOSITIONS:
        return "PREP"
    if lower in CONJUNCTIONS:
        return "CONJ"
    if lower in AUX_VERBS:
        return "AUX"
    if lower.endswith(ADVERB_SUFFIXES) and len(lower) > 4:
        return "ADV"
    if lower.endswith(VERB_SUFFIXES) and len(lower) > 4:
        return "VERB"
    if lower.endswith(ADJECTIVE_SUFFIXES) and len(lower) > 4:
        return "ADJ"
    if lower.endswith(NOUN_SUFFIXES) and len(lower) > 4:
        return "NOUN"
    if index > 0 and tokens[index - 1].lower() in DETERMINERS:
        return "NOUN"
    if token[:1].isupper() and index > 0:
        return "PROPN"
    return "NOUN"


def pos_tag(tokens):
    return [(tok, tag_token(tok, i, tokens)) for i, tok in enumerate(tokens)]


def extract_noun_phrases(tagged_tokens):
    phrases = []
    current = []
    for token, tag in tagged_tokens:
        if tag in ("DET", "ADJ", "NOUN", "PROPN", "NUM"):
            current.append(token)
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))
    return phrases


def extract_verb_phrases(tagged_tokens):
    phrases = []
    current = []
    for token, tag in tagged_tokens:
        if tag in ("AUX", "VERB", "ADV"):
            current.append(token)
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))
    return phrases


def classify_sentence_type(tokens):
    if not tokens:
        return "unknown"
    first = tokens[0].lower()
    stripped = tokens[-1] if tokens else ""
    if stripped == "?" or first in WH_WORDS:
        return "question"
    if first in AUX_VERBS:
        return "yes_no_question"
    if tokens[0].lower() in {"please", "let"} or (tokens and tag_token(tokens[0], 0, tokens) == "VERB"):
        return "command"
    return "statement"


class ShallowParseResult:
    def __init__(self, tokens, tagged, noun_phrases, verb_phrases, sentence_type):
        self.tokens = tokens
        self.tagged = tagged
        self.noun_phrases = noun_phrases
        self.verb_phrases = verb_phrases
        self.sentence_type = sentence_type

    def to_dict(self):
        return {
            "tokens": self.tokens,
            "tagged": self.tagged,
            "noun_phrases": self.noun_phrases,
            "verb_phrases": self.verb_phrases,
            "sentence_type": self.sentence_type,
        }


def shallow_parse(tokens):
    tagged = pos_tag(tokens)
    noun_phrases = extract_noun_phrases(tagged)
    verb_phrases = extract_verb_phrases(tagged)
    sentence_type = classify_sentence_type(tokens)
    return ShallowParseResult(tokens, tagged, noun_phrases, verb_phrases, sentence_type)


def extract_subject_predicate_object(tagged_tokens):
    subject, predicate, obj = [], [], []
    stage = 0
    for token, tag in tagged_tokens:
        if stage == 0:
            if tag in ("PRON", "NOUN", "PROPN", "DET"):
                subject.append(token)
            elif tag in ("VERB", "AUX"):
                stage = 1
                predicate.append(token)
        elif stage == 1:
            if tag in ("VERB", "AUX", "ADV"):
                predicate.append(token)
            else:
                stage = 2
                obj.append(token)
        else:
            obj.append(token)
    return " ".join(subject), " ".join(predicate), " ".join(obj)

