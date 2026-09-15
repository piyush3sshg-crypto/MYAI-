from system import System
from rules import Rule
from planning.action import ActionSchema
from math_engine import parse_linear_equation, evaluate_expression, ArithmeticReasoner
from induction import TrainingExample, VersionSpaceLearner, DecisionStumpLearner
from weather import format_weather

from data.sample_data import load_intent_dataset
from analytics.eda import describe, class_balance, missing_report
from ml.cross_validation import cross_validate
from ml.experiment_tracker import ExperimentTracker
from ai.neural import NeuralIntentClassifier


def seed_family_knowledge(brain):
    brain.add_fact("parent", "arjun", "vihaan")
    brain.add_fact("parent", "vihaan", "kabir")
    brain.add_fact("parent", "meera", "vihaan")
    brain.add_fact("parent", "kabir", "ishaan")
    brain.add_fact("male", "arjun")
    brain.add_fact("male", "vihaan")
    brain.add_fact("male", "kabir")
    brain.add_fact("male", "ishaan")
    brain.add_fact("female", "meera")

    brain.add_rule(Rule(
        name="grandparent_rule",
        conditions=[("parent", "?a", "?b"), ("parent", "?b", "?c")],
        conclusions=[("grandparent", "?a", "?c")],
    ))
    brain.add_rule(Rule(
        name="father_rule",
        conditions=[("parent", "?a", "?b"), ("male", "?a")],
        conclusions=[("father", "?a", "?b")],
    ))
    brain.add_rule(Rule(
        name="ancestor_base_rule",
        conditions=[("parent", "?a", "?b")],
        conclusions=[("ancestor", "?a", "?b")],
    ))
    brain.add_rule(Rule(
        name="ancestor_transitive_rule",
        conditions=[("parent", "?a", "?b"), ("ancestor", "?b", "?c")],
        conclusions=[("ancestor", "?a", "?c")],
    ))

    brain.add_relation("arjun", "is_a", "person")
    brain.add_relation("vihaan", "is_a", "person")
    brain.add_relation("person", "is_a", "living_being")
    brain.knowledge_graph.mark_transitive("is_a")


def seed_blocks_world(brain):
    on_table_a = ("on_table", "block_a")
    on_b_on_a = ("on", "block_b", "block_a")
    clear_b = ("clear", "block_b")
    handempty = ("handempty",)

    move_schema = ActionSchema(
        name="move",
        parameters=("?x", "?y", "?z"),
        preconditions=[("on", "?x", "?y"), ("clear", "?x"), ("clear", "?z"), ("handempty",)],
        add_effects=[("on", "?x", "?z"), ("clear", "?y"), ("handempty",)],
        del_effects=[("on", "?x", "?y"), ("clear", "?z"), ("handempty",)],
    )
    move_to_table_schema = ActionSchema(
        name="move_to_table",
        parameters=("?x", "?y"),
        preconditions=[("on", "?x", "?y"), ("clear", "?x"), ("handempty",)],
        add_effects=[("on_table", "?x"), ("clear", "?y"), ("handempty",)],
        del_effects=[("on", "?x", "?y"), ("handempty",)],
    )
    move_from_table_schema = ActionSchema(
        name="move_from_table",
        parameters=("?x", "?z"),
        preconditions=[("on_table", "?x"), ("clear", "?x"), ("clear", "?z"), ("handempty",)],
        add_effects=[("on", "?x", "?z"), ("handempty",)],
        del_effects=[("on_table", "?x"), ("clear", "?z")],
    )
    brain.register_action_schema(move_schema)
    brain.register_action_schema(move_to_table_schema)
    brain.register_action_schema(move_from_table_schema)

    initial_state = {on_table_a, on_b_on_a, clear_b, handempty}
    goal_state = {("on_table", "block_b"), ("on", "block_a", "block_b")}
    return initial_state, goal_state


def seed_chatbot_intents(brain):
    brain.register_intent_examples("greeting", [
        "hi there", "good morning", "hey how are you", "namaste",
        "hello", "yo whats up", "good evening", "hii",
    ])
    brain.register_intent_examples("farewell", [
        "goodbye", "see you later", "bye take care",
        "catch you later", "i am leaving now", "farewell then",
    ])
    brain.register_intent_examples("math_query", [
        "what is 12 plus 45",
        "solve 2x + 4 = 10",
        "calculate the square root of 81",
        "evaluate 5 times 7 minus 3",
        "what is 9 divided by 3",
        "what is 100 minus 37",
    ])
    brain.register_intent_examples("family_query", [
        "who is the father of vihaan",
        "who are the grandparents of kabir",
        "is arjun an ancestor of ishaan",
        "tell me about the family tree",
        "who is vihaan related to",
    ])
    brain.register_intent_examples("weather_query", [
        "mausam kaisa hai",
        "kya barish ho rahi hai",
        "aaj dhoop hai kya",
        "how is the weather today",
        "is it raining outside",
        "will it be sunny tomorrow",
    ])

    def greeting_handler(text, engine_ref, parsed_sentences):
        return "Namaste, main aapka thinking engine hoon."

    def farewell_handler(text, engine_ref, parsed_sentences):
        return "Alvida, phir milenge."

    def math_handler(text, engine_ref, parsed_sentences):
        if "=" in text:
            try:
                solution = parse_linear_equation(text.split("solve", 1)[-1].strip())
                return f"solution: x = {solution}"
            except Exception as exc:
                return f"could not solve equation: {exc}"
        word_operators = {
            "plus": "+", "minus": "-", "times": "*", "multiplied by": "*",
            "divided by": "/", "over": "/", "modulo": "%",
        }
        normalized_text = text.lower()
        for word, symbol in word_operators.items():
            normalized_text = normalized_text.replace(word, f" {symbol} ")
        digits_and_ops = "".join(ch for ch in normalized_text if ch.isdigit() or ch in "+-*/(). ")
        try:
            value = evaluate_expression(digits_and_ops) if digits_and_ops.strip() else None
            return f"result: {value}" if value is not None else "could not extract expression"
        except Exception as exc:
            return f"could not evaluate expression: {exc}"

    def family_handler(text, engine_ref, parsed_sentences):
        engine_ref.run_forward_chaining()
        fathers = engine_ref.fact_store.all_by_predicate("father")
        grandparents = engine_ref.fact_store.all_by_predicate("grandparent")
        ancestors = engine_ref.fact_store.all_by_predicate("ancestor")
        return {
            "fathers": [f.as_tuple() for f in fathers],
            "grandparents": [g.as_tuple() for g in grandparents],
            "ancestors": [a.as_tuple() for a in ancestors],
        }

    def weather_handler(text, engine_ref, parsed_sentences):
        return format_weather(26.85, 80.95, "Lucknow")

    brain.register_intent_handler("greeting", greeting_handler)
    brain.register_intent_handler("farewell", farewell_handler)
    brain.register_intent_handler("math_query", math_handler)
    brain.register_intent_handler("family_query", family_handler)
    brain.register_intent_handler("weather_query", weather_handler)


def demonstrate_induction():
    domains = {
        "sky": ["sunny", "cloudy", "rainy"],
        "temperature": ["warm", "cold"],
        "humidity": ["high", "normal"],
        "wind": ["strong", "weak"],
    }
    examples = [
        TrainingExample({"sky": "sunny", "temperature": "warm", "humidity": "normal", "wind": "strong"}, True),
        TrainingExample({"sky": "sunny", "temperature": "warm", "humidity": "high", "wind": "strong"}, True),
        TrainingExample({"sky": "rainy", "temperature": "cold", "humidity": "high", "wind": "strong"}, False),
        TrainingExample({"sky": "sunny", "temperature": "warm", "humidity": "high", "wind": "weak"}, True),
    ]
    learner = VersionSpaceLearner(domains)
    specific, general = learner.train(examples)
    stump = DecisionStumpLearner()
    stump.train(examples, list(domains.keys()))
    return specific, general, stump.best_attribute


def _train_intent_classifier(train_dataset, hidden_size=8, epochs=30, learning_rate=0.2):
    classifier = NeuralIntentClassifier(hidden_size=hidden_size, seed=42)
    for record in train_dataset:
        classifier.add_example(record["label"], record["text"])
    classifier.train(epochs=epochs, learning_rate=learning_rate, validation_split=0.0)
    return classifier


def _predict_intent(classifier, record):
    match = classifier.predict(record["text"])
    return match.intent_name


def demonstrate_data_science():
    """End-to-end pass through the new data science layer: load a
    dataset, run EDA, cross-validate a real model, and log the run.
    This is what the project docs flagged as entirely missing.
    """
    dataset = load_intent_dataset()

    balance = class_balance(dataset, label_column="label")
    missing = missing_report(dataset)

    cv_results = cross_validate(
        dataset,
        train_fn=_train_intent_classifier,
        predict_fn=_predict_intent,
        k=5, seed=42, stratify_by="label",
    )

    tracker = ExperimentTracker(log_path="ml/experiments.json")
    tracker.log_run(
        experiment_name="intent_classifier_cv",
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        params={"hidden_size": 8, "epochs": 30, "learning_rate": 0.2, "k": 5},
        metrics={
            "mean_accuracy": cv_results["mean_accuracy"],
            "mean_macro_f1": cv_results["mean_macro_f1"],
        },
        notes="baseline cross-validated run via main.py demo",
    )

    return balance, missing, cv_results


def run_demo():
    brain = System()
    seed_family_knowledge(brain)
    seed_chatbot_intents(brain)
    initial_state, goal_state = seed_blocks_world(brain)

    print("=== Family Reasoning ===")
    brain.run_forward_chaining()
    for fact in brain.fact_store.all_by_predicate("grandparent"):
        print("grandparent:", fact.args)
    for fact in brain.fact_store.all_by_predicate("ancestor"):
        print("ancestor:", fact.args)

    print("\n=== Knowledge Graph Closure ===")
    for triple in brain.infer_graph_closures():
        print("inferred:", triple)

    print("\n=== Blocks World Planning ===")
    actions, cost, expansions = brain.plan_towards_goal(initial_state, goal_state)
    if actions:
        for action in actions:
            print("action:", action)
        print("total cost:", cost, "expansions:", expansions)
    else:
        print("no plan found within expansion budget:", expansions)

    print("\n=== Dialogue Simulation ===")
    sample_turns = [
        "hi there",
        "who is the father of vihaan",
        "solve 2*x + 4 = 10",
        "what is 12 plus 45",
        "kya barish ho rahi hai",
        "see you later",
    ]
    for turn in sample_turns:
        rendered, response = brain.converse(turn)
        print(f"user: {turn}")
        print(f"engine[{response.kind}]: {rendered if isinstance(rendered, str) else response.payload}")

    print("\n=== Inductive Learning Demo ===")
    specific, general, best_attribute = demonstrate_induction()
    print("specific boundary:", specific)
    print("general boundary:", general)
    print("best splitting attribute:", best_attribute)

    print("\n=== Reflection ===")
    print(brain.introspect())

    print("\n=== Neural Intent Classifier ===")
    report = brain.enable_neural_intents(
        hidden_size=10, epochs=80, learning_rate=0.2,
        model_path="models/intent_classifier.json",
    )
    if report.get("loaded_from"):
        print(f"loaded saved model from {report['loaded_from']} (skipped retraining)")
    print("training report:", report)
    neural_test_sentences = [
        "hey there",
        "talk to you later",
        "what is 7 plus 8",
        "who is vihaan father",
    ]
    for sentence in neural_test_sentences:
        match = brain.neural_intent_classifier.predict(sentence)
        engine_response = brain.think_neural(sentence)
        print(f"input: {sentence!r} -> neural intent: {match.intent_name} (confidence {match.score:.4f}) -> response kind: {engine_response.kind}")

    print("\n=== Data Science Layer: EDA + Cross-Validation ===")
    balance, missing, cv_results = demonstrate_data_science()
    print("class balance:", balance)
    print("missing value report:", missing)
    print("5-fold cross-validation:", {
        "mean_accuracy": cv_results["mean_accuracy"],
        "mean_macro_f1": cv_results["mean_macro_f1"],
    })
    print("(run logged to ml/experiments.json)")

    print("\n=== Writer: Text Generation (Markov chain) ===")
    space_text = (
        "The universe is vast and full of mysteries. Stars are born inside giant clouds "
        "of gas and dust called nebulae. A black hole is a region of space where gravity "
        "is so strong that nothing can escape, not even light. Astronauts travel to space "
        "using powerful rockets that burn fuel to escape earth gravity. The sun is a star "
        "at the center of our solar system and it provides light and heat to all planets. "
        "Mars is a red planet and scientists believe it once had liquid water on its surface."
    )
    brain.learn_topic("space", space_text)
    print("known topics:", brain.known_topics())
    for seed in range(2):
        print(f"generated (seed={seed}):", brain.write_about("space", max_words=25, seed=seed))

    print("\n=== Neural Writer: Character RNN (genuine backprop-through-time) ===")
    sky_text = (
        "the sun rises in the east and sets in the west. the moon shines at night. "
        "stars twinkle in the dark sky. the sky is blue during the day. "
        "clouds float above the mountains. rain falls from the clouds sometimes."
    )
    rnn_report = brain.learn_topic_neural("sky", sky_text, epochs=25, seq_length=18, learning_rate=0.2)
    print("training report:", rnn_report)
    print("neural generated:", brain.write_about_neural("sky", seed_text="the ", length=80, temperature=0.6, seed=2))

    print("\n=== Reader: Summarization + Q&A ===")
    climate_article = (
        "Climate change is one of the biggest challenges facing humanity today. Rising "
        "global temperatures are causing ice caps to melt at an alarming rate. This "
        "melting ice contributes directly to rising sea levels around the world. "
        "Scientists believe that reducing carbon emissions is critical to slowing down "
        "this process. Renewable energy sources like solar and wind power can help "
        "reduce our dependence on fossil fuels. Forests play a huge role in absorbing "
        "carbon dioxide from the atmosphere."
    )
    brain.read_document(climate_article)
    print("summary:", brain.summarize_document(num_sentences=2))
    qa_result = brain.ask_document("what helps reduce carbon emissions")
    print("question: what helps reduce carbon emissions")
    print("answer:", qa_result[0][1], f"(score={qa_result[0][0]:.3f})")
    print("keywords:", brain.document_keywords(top_k=5))


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as exc:
        import sys
        print(f"\n[FATAL] MYAI demo crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Run 'python3 verify.py' to check which component is broken.", file=sys.stderr)
        raise
