"""Automatic, unreviewed learning from live chat traffic.

Two things get learned automatically, with no /teach or /good command
needed:

1. INTENT LEARNING — every message that the neural intent classifier
   already predicts with high confidence gets logged as a new training
   example (text, predicted_label) to data/learned_intents.jsonl. The
   next time anything calls load_intent_dataset(), these get merged in
   alongside the hand-written examples, so the classifier that trains
   from the growing example pool improves over time.

   This is "self-training" / pseudo-labeling: a known, standard technique,
   but also a known risk — if the classifier is confidently wrong, it will
   reinforce its own mistake. The min_confidence threshold (default 0.75)
   and dedup-by-text are the two safety rails. There is no human review
   step by design (the user explicitly asked for fully automatic learning).

2. FACT LEARNING — simple relationship/identity statements ("my father is
   Arjun", "my name is Priya") are extracted with regex and:
     a) added immediately to the live FactStore via add_fact_fn, with a
        lower confidence (0.7) than hand-coded facts (1.0), so downstream
        reasoning can tell curated facts from chat-derived ones if it
        ever needs to.
     b) appended to knowledge/learned_facts.jsonl so they persist and get
        reloaded on the next run (see load_learned_facts / System usage
        in system.py).
"""
import json
import os
import re
import time

QUESTION_WORDS = {
    "how", "what", "why", "who", "when", "where", "which", "many", "much",
    "is", "are", "does", "do", "did", "was", "were", "can", "could", "would",
    "there", "any", "some",
    "kya", "kaun", "kab", "kahan", "kyun", "kaise", "kitna", "kitne", "kitni",
}

INTENT_LOG_PATH = "data/learned_intents.jsonl"
FACT_LOG_PATH = "knowledge/learned_facts.jsonl"
DEFAULT_MIN_CONFIDENCE = 0.75
FACT_CONFIDENCE = 0.7

_SYMS = r".,!?;:()\[\]{}\"'"

FACT_PATTERNS = [
    # (regex, builder) — builder(match) -> list of (predicate, arg1, arg2) triples
    (re.compile(rf"(?:my name is|i am called|i'm called|mera naam|mera nam)\s+([^\s{_SYMS}]+)", re.I),
     lambda m: [("name", "user", m.group(1).lower())]),
    (re.compile(rf"(?:my father is|my dad is|mera papa|mera pita)\s+([^\s{_SYMS}]+)", re.I),
     lambda m: [("parent", m.group(1).lower(), "user"), ("male", m.group(1).lower())]),
    (re.compile(rf"(?:my mother is|my mom is|meri mummy|meri maa)\s+([^\s{_SYMS}]+)", re.I),
     lambda m: [("parent", m.group(1).lower(), "user"), ("female", m.group(1).lower())]),
    (re.compile(rf"([^\s{_SYMS}]+)\s+is my son", re.I),
     lambda m: [("parent", "user", m.group(1).lower()), ("male", m.group(1).lower())]),
    (re.compile(rf"([^\s{_SYMS}]+)\s+is my daughter", re.I),
     lambda m: [("parent", "user", m.group(1).lower()), ("female", m.group(1).lower())]),
    (re.compile(rf"([^\s{_SYMS}]+)\s+is my brother", re.I),
     lambda m: [("sibling", "user", m.group(1).lower()), ("male", m.group(1).lower())]),
    (re.compile(rf"([^\s{_SYMS}]+)\s+is my sister", re.I),
     lambda m: [("sibling", "user", m.group(1).lower()), ("female", m.group(1).lower())]),
    (re.compile(rf"(?:i live in|i stay in|main)\s+([^\s{_SYMS}]+)", re.I),
     lambda m: [("lives_in", "user", m.group(1).lower())]),
]


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _already_logged(path, text):
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("text") == text:
                return True
    return False


def learn_intent(text, classifier, path=INTENT_LOG_PATH, min_confidence=DEFAULT_MIN_CONFIDENCE):
    """Log (text, predicted_label) if the classifier is confident and this
    exact text hasn't been logged before. Returns the label logged, or
    None if nothing was logged (low confidence / duplicate / untrained)."""
    text = (text or "").strip()
    if not text or classifier is None or not getattr(classifier, "trained", False):
        return None
    match = classifier.predict(text)
    if match.score < min_confidence or match.intent_name == "unknown":
        return None
    if _already_logged(path, text):
        return None
    _append_jsonl(path, {
        "text": text, "label": match.intent_name,
        "confidence": round(match.score, 4), "ts": time.time(), "source": "auto_chat",
    })
    return match.intent_name


def learn_fact(text, add_fact_fn, path=FACT_LOG_PATH):
    """Extract simple relationship/identity facts from free text, apply
    them immediately via add_fact_fn(predicate, *args, confidence, source),
    and persist them so they survive a restart. Returns the list of
    (predicate, arg1, arg2) triples learned this call."""
    text = (text or "").strip()
    if not text:
        return []
    learned = []
    for rx, builder in FACT_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        if any(g and g.lower() in QUESTION_WORDS for g in m.groups()):
            # e.g. "mera naam kya hai" (a QUESTION) also matches the "mera
            # naam <value>" STATEMENT pattern with "kya" captured as the
            # name. Reject any match where the captured value is itself a
            # question word, so asking a question never gets learned as a
            # fact.
            continue
        for triple in builder(m):
            predicate, args = triple[0], triple[1:]
            if _already_logged(path, json.dumps({"predicate": predicate, "args": list(args)})):
                continue
            if add_fact_fn is not None:
                add_fact_fn(predicate, *args, confidence=FACT_CONFIDENCE, source="auto_chat")
            _append_jsonl(path, {
                "predicate": predicate, "args": list(args),
                "confidence": FACT_CONFIDENCE, "ts": time.time(), "source": "auto_chat",
                "text": json.dumps({"predicate": predicate, "args": list(args)}),
            })
            learned.append(triple)
    return learned


def load_learned_intents(path=INTENT_LOG_PATH):
    """Returns a list of (label, text) tuples for merging into the
    hand-written INTENT_EXAMPLES at dataset-load time."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "label" in rec and "text" in rec:
                out.append((rec["label"], rec["text"]))
    return out


def load_learned_facts(path=FACT_LOG_PATH):
    """Returns a list of (predicate, *args) tuples for FactStore.load()."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "predicate" in rec and "args" in rec:
                out.append(tuple([rec["predicate"]] + list(rec["args"])))
    return out
