import time

from ai.neural import NeuralIntentClassifier
from math_engine import parse_linear_equation, evaluate_expression


class Tool:
    def __init__(self, name, description, function):
        self.name = name
        self.description = description
        self.function = function

    def call(self, text, context):
        return self.function(text, context)


class ToolAgent:
    def __init__(self, hidden_size=16, seed=42):
        self.tools = {}
        self.classifier = NeuralIntentClassifier(hidden_size=hidden_size, seed=seed)
        self.trained = False
        self.context = {}
        self.call_log = []

    def register_tool(self, name, description, function, example_phrases):
        self.tools[name] = Tool(name, description, function)
        self.classifier.add_examples(name, example_phrases)

    def train(self, epochs=400, learning_rate=0.15, verbose=False):
        report = self.classifier.train(epochs=epochs, learning_rate=learning_rate, verbose=verbose)
        self.trained = True
        return report

    def list_tools(self):
        return [{"name": t.name, "description": t.description} for t in self.tools.values()]

    def run(self, text, threshold=0.4):
        if not self.trained:
            raise RuntimeError("agent has not been trained yet — call train() first")
        match = self.classifier.predict(text, threshold=threshold)
        entry = {
            "input": text,
            "tool_selected": match.intent_name,
            "confidence": match.score,
            "timestamp": time.time(),
        }
        if match.intent_name == "unknown" or match.intent_name not in self.tools:
            entry["result"] = None
            entry["error"] = "no matching tool found for this request"
            self.call_log.append(entry)
            return entry
        tool = self.tools[match.intent_name]
        try:
            result = tool.call(text, self.context)
            entry["result"] = result
            entry["error"] = None
        except Exception as exc:
            entry["result"] = None
            entry["error"] = str(exc)
        self.call_log.append(entry)
        return entry


def calculator_tool(text, context):
    lowered = text.lower()
    if "=" in text:
        equation_part = text
        if "solve" in lowered:
            equation_part = text[lowered.index("solve") + len("solve"):].strip()
        solution = parse_linear_equation(equation_part)
        return f"x = {solution}"
    word_operators = {
        "plus": "+", "minus": "-", "times": "*", "multiplied by": "*",
        "divided by": "/", "over": "/", "modulo": "%",
    }
    normalized = lowered
    for word, symbol in word_operators.items():
        normalized = normalized.replace(word, f" {symbol} ")
    digits_and_ops = "".join(ch for ch in normalized if ch.isdigit() or ch in "+-*/(). ")
    if not digits_and_ops.strip():
        return "could not find a numeric expression in the request"
    return evaluate_expression(digits_and_ops)


def make_general_knowledge_tool(fact_store):
    def tool(text, context):
        lowered = text.lower()
        words = lowered.replace("?", "").split()

        if "capital" in words:
            for fact in fact_store.all_by_predicate("capital_of"):
                if fact.args[0] in words:
                    return f"The capital of {fact.args[0].replace('_',' ')} is {fact.args[1].replace('_',' ')}."

        if "planet" in words or "planets" in words:
            if "largest" in words:
                return "Jupiter is the largest planet in the solar system."
            if "smallest" in words:
                return "Mercury is the smallest planet in the solar system."

        for fact in fact_store.all_by_predicate("chemical_formula"):
            if fact.args[0].replace("_", " ") in lowered:
                return f"The chemical formula of {fact.args[0].replace('_',' ')} is {fact.args[1]}."

        for fact in fact_store.all_by_predicate("river_in"):
            if fact.args[1] in words:
                return f"The {fact.args[0].capitalize()} river flows through {fact.args[1].replace('_',' ')}."

        return "I could not find a matching fact for this question."
    return tool


def make_summarizer_tool(reader_factory):
    def tool(text, context):
        document = context.get("current_document")
        if document is None:
            return "no document has been loaded into context — set context['current_document'] first"
        reader = reader_factory()
        reader.load(document)
        return reader.summarize(num_sentences=2)
    return tool


def make_generator_tool(writer):
    def tool(text, context):
        topic = context.get("current_topic")
        if topic is None or topic not in writer.known_topics():
            return "no writing topic is loaded into context — set context['current_topic'] first"
        return writer.write_about(topic, max_words=25, seed=1)
    return tool


def build_default_agent(fact_store=None, reader_factory=None, writer=None):
    agent = ToolAgent()

    agent.register_tool(
        "calculator",
        "evaluates arithmetic expressions and solves simple linear equations",
        calculator_tool,
        [
            "what is 12 plus 45",
            "calculate 9 times 3",
            "solve 2x + 4 = 10",
            "what is 100 minus 37",
            "evaluate 5 times 7 minus 3",
            "what is 20 divided by 4",
        ],
    )

    if fact_store is not None:
        agent.register_tool(
            "general_knowledge",
            "answers general factual questions about geography, science, and history",
            make_general_knowledge_tool(fact_store),
            [
                "what is the capital of india",
                "what is the largest planet",
                "what is the chemical formula of water",
                "which river is in egypt",
                "what is the smallest planet",
            ],
        )

    if reader_factory is not None:
        agent.register_tool(
            "summarizer",
            "summarizes the currently loaded document into key sentences",
            make_summarizer_tool(reader_factory),
            [
                "summarize this document",
                "give me a summary",
                "what are the key points",
                "condense this text for me",
            ],
        )

    if writer is not None:
        agent.register_tool(
            "generator",
            "generates new text about the currently loaded writing topic",
            make_generator_tool(writer),
            [
                "write something about this topic",
                "generate some text",
                "continue writing about this",
                "compose a sentence about it",
            ],
        )

    return agent
