"""Starter pytest suite for MYAI core behavior.

Run with: pytest tests/test_core.py -v
(from the MYAI/ project root, so the top-level modules are importable)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from system import System
from rules import Rule
from main import seed_family_knowledge, seed_chatbot_intents


def build_brain():
    brain = System()
    seed_family_knowledge(brain)
    seed_chatbot_intents(brain)
    return brain


def test_forward_chaining_produces_grandparent_facts():
    brain = build_brain()
    brain.run_forward_chaining()
    grandparents = brain.fact_store.all_by_predicate("grandparent")
    assert len(grandparents) > 0


def test_forward_chaining_produces_father_facts():
    brain = build_brain()
    brain.run_forward_chaining()
    fathers = [f.as_tuple() for f in brain.fact_store.all_by_predicate("father")]
    assert ("father", "arjun", "vihaan") in fathers


def test_knowledge_graph_closure_is_transitive():
    brain = build_brain()
    closures = list(brain.infer_graph_closures())
    assert ("arjun", "is_a", "living_being") in closures or any(
        c[0] == "arjun" and c[2] == "living_being" for c in closures
    )


def test_math_query_dialogue():
    brain = build_brain()
    rendered, response = brain.converse("what is 12 plus 45")
    assert response.kind == "handled"
    assert "57" in str(rendered) or "57" in str(response.payload)


def test_neural_intent_classifier_trains_and_reports_validation_accuracy():
    brain = build_brain()
    report = brain.enable_neural_intents(hidden_size=16, epochs=100, learning_rate=0.15)
    assert report["training_accuracy"] is not None
    # Regression guard: validation_split now defaults to 0.2, so this must
    # never silently come back as None (see project docs, issue #3).
    assert report["validation_accuracy"] is not None
    assert 0.0 <= report["validation_accuracy"] <= 1.0


def test_neural_intent_classifier_predicts_known_intent():
    brain = build_brain()
    brain.enable_neural_intents(hidden_size=16, epochs=200, learning_rate=0.15)
    match = brain.neural_intent_classifier.predict("what is 7 plus 8")
    assert match.intent_name == "math_query"


def test_induction_learns_consistent_hypothesis():
    from induction import TrainingExample, VersionSpaceLearner

    domains = {"sky": ["sunny", "rainy"], "temperature": ["warm", "cold"]}
    examples = [
        TrainingExample({"sky": "sunny", "temperature": "warm"}, True),
        TrainingExample({"sky": "rainy", "temperature": "cold"}, False),
    ]
    learner = VersionSpaceLearner(domains)
    specific, general = learner.train(examples)
    assert specific is not None
    assert general is not None
