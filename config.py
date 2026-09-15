INTENT_CONFIDENCE_THRESHOLD = 0.18
WORKING_MEMORY_DECAY_RATE = 0.05
WORKING_MEMORY_ACTIVATION_THRESHOLD = 0.1
WORKING_MEMORY_CAPACITY = 200
ATTENTION_FOCUS_SIZE = 5
EPISODIC_MEMORY_MAX_EPISODES = 5000
EPISODIC_MEMORY_MIN_SCORE = 0.0
EPISODIC_MEMORY_TOP_K = 5
PLANNER_MAX_EXPANSIONS = 20000
PLANNER_DEFAULT_STRATEGY = "astar"
RULE_ENGINE_MAX_ITERATIONS = 1000
BACKWARD_CHAIN_MAX_DEPTH = 50
TASK_SCHEDULER_MAX_HISTORY = 500
DIALOGUE_MAX_HISTORY = 200
TOKENIZER_STRIP_STOPWORDS = True
TOKENIZER_APPLY_STEMMING = True

RUNTIME_SETTINGS = {
    "intent_threshold": INTENT_CONFIDENCE_THRESHOLD,
    "working_memory": {
        "decay_rate": WORKING_MEMORY_DECAY_RATE,
        "activation_threshold": WORKING_MEMORY_ACTIVATION_THRESHOLD,
        "capacity": WORKING_MEMORY_CAPACITY,
    },
    "attention": {
        "focus_size": ATTENTION_FOCUS_SIZE,
    },
    "episodic_memory": {
        "max_episodes": EPISODIC_MEMORY_MAX_EPISODES,
        "min_score": EPISODIC_MEMORY_MIN_SCORE,
        "top_k": EPISODIC_MEMORY_TOP_K,
    },
    "planner": {
        "max_expansions": PLANNER_MAX_EXPANSIONS,
        "strategy": PLANNER_DEFAULT_STRATEGY,
    },
    "rule_engine": {
        "max_iterations": RULE_ENGINE_MAX_ITERATIONS,
        "backward_chain_max_depth": BACKWARD_CHAIN_MAX_DEPTH,
    },
    "scheduler": {
        "max_history": TASK_SCHEDULER_MAX_HISTORY,
    },
    "dialogue": {
        "max_history": DIALOGUE_MAX_HISTORY,
    },
    "tokenizer": {
        "strip_stopwords": TOKENIZER_STRIP_STOPWORDS,
        "apply_stemming": TOKENIZER_APPLY_STEMMING,
    },
}


def get(path, default=None):
    node = RUNTIME_SETTINGS
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def override(path, value):
    parts = path.split(".")
    node = RUNTIME_SETTINGS
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value

