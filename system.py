import os
import json

import config

from Datastructures import (
    TrieNode, Trie, UpdatablePriorityQueue, DisjointSet,
    WeightedGraph, LRUCache, RingBuffer, MinMaxHeap,
)
from core.similarity import (
    levenshtein_distance, normalized_levenshtein, jaccard_similarity,
    ngrams, ngram_overlap, cosine_similarity, cosine_similarity_dense,
    dot_product_dense, TfidfVectorizer, weighted_score, hybrid_text_similarity,
)
from core.tokenizee import (
    Tokenizer, tokenize, split_sentences, normalize_text,
    remove_stopwords, stem_word, stem_tokens, char_ngrams,
)
from parser import (
    pos_tag, shallow_parse, ShallowParseResult, extract_noun_phrases,
    extract_verb_phrases, extract_subject_predicate_object, classify_sentence_type,
)
from intent import IntentClassifier, IntentMatch, IntentExample, SlotFiller, SlotPattern
from Facts import Fact, FactStore, is_variable, unify, substitute, cartesian_bindings
from rules import Rule, RuleEngine, RuleFiring
from graph_kb import KnowledgeGraph, ConceptHierarchy
from planning.action import ActionSchema, GroundedAction, state_from_fact_store, extract_objects
from planning.planner import PlanningProblem, AStarPlanner, GreedyBestFirstPlanner, plan as run_planner
from memory.working_memory import ActivationUnit, WorkingMemory, AttentionFocus
from memory.episodic import Episode, EpisodicMemory
from deduction import DeductionResult, DeductiveReasoner
from abduction import Hypothesis, AbductiveReasoner
from induction import TrainingExample, VersionSpaceLearner, DecisionStumpLearner, RuleInducer
from math_engine import (
    ArithmeticReasoner, evaluate_expression, parse_linear_equation,
    solve_quadratic, tokenize_expression,
)
from scheduler import ReasoningTask, TaskScheduler, PeriodicJobRunner
from Dialogue import DialogueManager, DialogueContext, ConversationTurn, render_template
from ai.neural import NeuralIntentClassifier, NeuralIntentMatch
from writer import TopicWriter, MarkovChainModel
from reader import DocumentReader
from ai.rnn import NeuralTopicWriter, CharRNN


class ThoughtTrace:
    def __init__(self):
        self.steps = []

    def log(self, stage, detail):
        import time
        self.steps.append({"stage": stage, "detail": detail, "timestamp": time.time()})

    def render(self):
        return "\n".join(f"[{s['stage']}] {s['detail']}" for s in self.steps)


class EngineResponse:
    def __init__(self, kind, payload, trace, confidence=1.0):
        self.kind = kind
        self.payload = payload
        self.trace = trace
        self.confidence = confidence

    def __repr__(self):
        return f"EngineResponse(kind={self.kind}, confidence={self.confidence:.3f})"


class _ProductionIntentAdapter:
    """Wraps ProductionClassifier (sklearn-trained, pure-Python inference)
    so it's a drop-in replacement for NeuralIntentClassifier — every
    existing caller only ever uses .predict(text, threshold=...) and reads
    .intent_name / .score / .entities off the result, so nothing else
    needs to change."""

    def __init__(self, underlying):
        self._underlying = underlying
        self.trained = True

    def predict(self, text, threshold=0.0):
        label, score = self._underlying.predict(text)
        if score < threshold:
            return NeuralIntentMatch("unknown", score)
        return NeuralIntentMatch(label, score)


class System:
    def __init__(self):
        self.tokenizer = Tokenizer(
            strip_stopwords=config.get("tokenizer.strip_stopwords", True),
            apply_stemming=config.get("tokenizer.apply_stemming", True),
        )
        self.fact_store = FactStore()
        self.rule_engine = RuleEngine(self.fact_store)
        self.knowledge_graph = KnowledgeGraph()
        self.intent_classifier = IntentClassifier(
            threshold=config.get("intent_threshold", 0.18)
        )
        self.working_memory = WorkingMemory(
            decay_rate=config.get("working_memory.decay_rate", 0.05),
            activation_threshold=config.get("working_memory.activation_threshold", 0.1),
            capacity=config.get("working_memory.capacity", 200),
        )
        self.attention = AttentionFocus(
            self.working_memory, focus_size=config.get("attention.focus_size", 5)
        )
        self.episodic_memory = EpisodicMemory(
            max_episodes=config.get("episodic_memory.max_episodes", 5000)
        )
        self.deductive_reasoner = DeductiveReasoner(
            self.fact_store, self.rule_engine, self.knowledge_graph
        )
        self.abductive_reasoner = AbductiveReasoner(self.fact_store, self.rule_engine)
        self.scheduler = TaskScheduler()
        self.global_vectorizer = TfidfVectorizer()
        self.action_schemas = []
        self.intent_handlers = {}
        self.tick_count = 0
        self.dialogue = DialogueManager(self)
        self.neural_intent_classifier = None
        self.topic_writer = TopicWriter(order=2)
        self.document_reader = None
        self.neural_topic_writer = NeuralTopicWriter(hidden_size=32, seed=42)

    def add_fact(self, predicate, *args, confidence=1.0, source=None):
        return self.fact_store.add(predicate, args, confidence, source)

    def load_learned_facts(self, path=None):
        """Load facts that were auto-learned from past chat sessions
        (see learning/auto_learn.py) into the live fact store. Safe to
        call even if the file doesn't exist yet (returns 0)."""
        from learning.auto_learn import load_learned_facts, FACT_LOG_PATH
        tuples = load_learned_facts(path or FACT_LOG_PATH)
        for t in tuples:
            self.fact_store.add(t[0], t[1:], confidence=0.7, source="auto_chat")
        return len(tuples)

    def add_rule(self, rule):
        self.rule_engine.add_rule(rule)

    def add_relation(self, source, relation, target, weight=1.0):
        self.knowledge_graph.add_edge(source, relation, target, weight)

    def register_intent_examples(self, intent_name, examples):
        self.intent_classifier.add_examples(intent_name, examples)

    def register_intent_handler(self, intent_name, handler_fn):
        self.intent_handlers[intent_name] = handler_fn

    def register_action_schema(self, schema):
        self.action_schemas.append(schema)

    def enable_neural_intents(self, hidden_size=16, epochs=400, learning_rate=0.15, verbose=False,
                               validation_split=0.2, model_path=None):
        """Train the neural intent classifier — or, if model_path is given
        and a saved model already exists there, load it instead of
        retraining from scratch every run.
        """
        if model_path and os.path.exists(model_path):
            try:
                self.neural_intent_classifier = NeuralIntentClassifier.load(model_path)
                return {"loaded_from": model_path, "retrained": False}
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                # Corrupt or incompatible save file — fall through and retrain
                # rather than crashing the whole demo over a stale cache.
                print(f"warning: could not load saved model at {model_path} ({exc}); retraining")

        classifier = NeuralIntentClassifier(hidden_size=hidden_size, seed=42)
        for intent_name, text, _tokens in self.intent_classifier.examples:
            classifier.add_example(intent_name, text)
        report = classifier.train(
            epochs=epochs, learning_rate=learning_rate, verbose=verbose,
            validation_split=validation_split,
        )
        self.neural_intent_classifier = classifier

        if model_path:
            try:
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                classifier.save(model_path)
                report["saved_to"] = model_path
            except OSError as exc:
                print(f"warning: could not save model to {model_path} ({exc})")

        report["retrained"] = True
        return report

    def enable_production_intents(self, model_path="models/intent_model.json"):
        """Use the production-grade scikit-learn-trained classifier
        (TF-IDF + Logistic Regression, trained on your PC via
        pc_training/train_production_models.py) instead of the
        hand-rolled NeuralIntentClassifier. Sets self.neural_intent_classifier
        to a thin adapter so every existing call site (api_server.py,
        learning/auto_learn.py, ai/llm.py's chat wiring) keeps working
        unchanged — they only ever call .predict(text, threshold=...)
        and read .intent_name / .score / .entities off the result.
        """
        from production_classifier import ProductionClassifier
        underlying = ProductionClassifier.load(model_path)
        self.neural_intent_classifier = _ProductionIntentAdapter(underlying)
        return {"loaded_from": model_path, "kind": "production_sklearn"}

    def learn_topic(self, topic_name, text):
        return self.topic_writer.learn(topic_name, text)

    def known_topics(self):
        return self.topic_writer.known_topics()

    def write_about(self, topic_name, max_words=40, seed=None):
        return self.topic_writer.write_about(topic_name, max_words=max_words, seed=seed)

    def continue_writing(self, topic_name, prompt_text, max_words=40, seed=None):
        return self.topic_writer.continue_writing(topic_name, prompt_text, max_words=max_words, seed=seed)

    def learn_topic_neural(self, topic_name, text, epochs=60, seq_length=20, learning_rate=0.15, verbose=False):
        return self.neural_topic_writer.learn(
            topic_name, text, seq_length=seq_length, epochs=epochs,
            learning_rate=learning_rate, verbose=verbose,
        )

    def write_about_neural(self, topic_name, seed_text="", length=120, temperature=0.7, seed=None):
        return self.neural_topic_writer.write_about(
            topic_name, seed_text=seed_text, length=length, temperature=temperature, seed=seed
        )

    def read_document(self, text):
        self.document_reader = DocumentReader()
        self.document_reader.load(text)
        return self.document_reader

    def summarize_document(self, num_sentences=3):
        if self.document_reader is None:
            raise RuntimeError("no document has been loaded — call read_document(text) first")
        return self.document_reader.summarize(num_sentences=num_sentences)

    def ask_document(self, question, top_k=1):
        if self.document_reader is None:
            raise RuntimeError("no document has been loaded — call read_document(text) first")
        return self.document_reader.ask(question, top_k=top_k)

    def document_keywords(self, top_k=10):
        if self.document_reader is None:
            raise RuntimeError("no document has been loaded — call read_document(text) first")
        return self.document_reader.keyword_frequency(top_k=top_k)

    def _vector_for_text(self, text):
        tokens = self.tokenizer.process(text)
        self.global_vectorizer.partial_fit(tokens)
        return self.global_vectorizer.transform(tokens)

    def perceive(self, text):
        trace = ThoughtTrace()
        trace.log("perceive", f"received input of length {len(text)}")
        sentences = split_sentences(text)
        parsed_sentences = []
        for sentence in sentences:
            tokens = self.tokenizer.process(sentence)
            parsed = shallow_parse(tokens)
            parsed_sentences.append(parsed)
            for np in parsed.noun_phrases:
                self.working_memory.activate(np, amount=0.6)
        trace.log("perceive", f"parsed {len(parsed_sentences)} sentences")
        return parsed_sentences, trace

    def think(self, text, use_neural=False):
        trace = ThoughtTrace()
        parsed_sentences, perceive_trace = self.perceive(text)
        trace.steps.extend(perceive_trace.steps)
        if use_neural and self.neural_intent_classifier is not None:
            intent_match = self.neural_intent_classifier.predict(text, threshold=0.5)
            trace.log("intent", f"[neural] classified as {intent_match.intent_name} score={intent_match.score:.4f}")
        else:
            intent_match = self.intent_classifier.predict(text)
            trace.log("intent", f"classified as {intent_match.intent_name} score={intent_match.score:.4f}")
        response = self._route(text, intent_match, parsed_sentences, trace)
        context_vector = self._vector_for_text(text)
        self.episodic_memory.store(
            {"input": text, "response_kind": response.kind, "response_payload": response.payload},
            context_vector,
            tags=[intent_match.intent_name],
        )
        self.working_memory.decay()
        self.attention.compute_focus()
        self.tick_count += 1
        return response

    def _route(self, text, intent_match, parsed_sentences, trace):
        if intent_match.intent_name in self.intent_handlers:
            trace.log("route", f"dispatching to registered handler for {intent_match.intent_name}")
            payload = self.intent_handlers[intent_match.intent_name](text, self, parsed_sentences)
            return EngineResponse("handled", payload, trace, intent_match.score)
        return self._attempt_deduction(text, parsed_sentences, trace)

    def _attempt_deduction(self, text, parsed_sentences, trace):
        for parsed in parsed_sentences:
            subj, pred, obj = extract_subject_predicate_object(parsed.tagged)
            if not subj or not pred:
                continue
            predicate_key = pred.split()[0] if pred else "related_to"
            query = (predicate_key, subj, obj) if obj else (predicate_key, subj, "?x")
            query = tuple(t if t else "?x" for t in query)
            result = self.deductive_reasoner.answer_query(query)
            trace.log("deduction", f"query={query} satisfied={result.is_satisfied()}")
            if result.is_satisfied():
                return EngineResponse("deduction", result.bindings_list, trace, result.proof_confidence)
        observation_guess = self._guess_observation(parsed_sentences)
        if observation_guess is not None:
            hypotheses = self.abductive_reasoner.explain_observation(observation_guess)
            trace.log("abduction", f"generated {len(hypotheses)} hypotheses for {observation_guess}")
            if hypotheses:
                return EngineResponse("abduction", hypotheses, trace, hypotheses[0].plausibility)
        trace.log("fallback", "no deduction or abduction path found")
        return EngineResponse("unknown", None, trace, 0.0)

    def _guess_observation(self, parsed_sentences):
        for parsed in parsed_sentences:
            if parsed.noun_phrases:
                return ("observed", parsed.noun_phrases[0])
        return None

    def converse(self, text):
        return self.dialogue.handle_turn(text)

    def think_neural(self, text):
        return self.think(text, use_neural=True)

    def plan_towards_goal(self, initial_state, goal_conditions, strategy=None):
        strategy = strategy or config.get("planner.strategy", "astar")
        actions, cost, expansions = run_planner(
            initial_state, goal_conditions, self.action_schemas, strategy
        )
        return actions, cost, expansions

    def run_forward_chaining(self):
        return self.rule_engine.forward_chain()

    def infer_graph_closures(self):
        return self.knowledge_graph.infer_transitive_closures()

    def recall_similar(self, text, top_k=None):
        top_k = top_k or config.get("episodic_memory.top_k", 5)
        vector = self._vector_for_text(text)
        return self.episodic_memory.retrieve_similar(vector, top_k=top_k)

    def introspect(self):
        return {
            "facts": self.fact_store.size(),
            "rules": len(self.rule_engine.rules),
            "graph_nodes": len(self.knowledge_graph.node_attrs),
            "working_memory_units": len(self.working_memory.units),
            "episodes": self.episodic_memory.size(),
            "ticks": self.tick_count,
        }
