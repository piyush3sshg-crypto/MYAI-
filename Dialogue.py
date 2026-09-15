import time


class ConversationTurn:
    def __init__(self, speaker, text, engine_response=None):
        self.speaker = speaker
        self.text = text
        self.engine_response = engine_response
        self.timestamp = time.time()


class DialogueContext:
    def __init__(self):
        self.slots = {}
        self.active_topic = None
        self.pending_clarification = None

    def set_slot(self, name, value):
        self.slots[name] = value

    def get_slot(self, name, default=None):
        return self.slots.get(name, default)

    def clear_slot(self, name):
        if name in self.slots:
            del self.slots[name]

    def set_topic(self, topic):
        self.active_topic = topic

    def request_clarification(self, question):
        self.pending_clarification = question

    def resolve_clarification(self):
        self.pending_clarification = None


RESPONSE_TEMPLATES = {
    "deduction": [
        "Based on what I know, {payload}",
        "I can confirm that {payload}",
    ],
    "abduction": [
        "The most plausible explanation involves {payload}",
        "This might be because {payload}",
    ],
    "handled": [
        "{payload}",
    ],
    "unknown": [
        "I do not have enough information to answer that yet.",
        "Could you clarify what you mean?",
    ],
}


def render_template(kind, payload):
    templates = RESPONSE_TEMPLATES.get(kind, RESPONSE_TEMPLATES["unknown"])
    template = templates[0]
    if "{payload}" in template:
        return template.format(payload=payload)
    return template


class DialogueManager:
    def __init__(self, engine):
        self.engine = engine
        self.context = DialogueContext()
        self.history = []
        self.max_history = 200

    def handle_turn(self, user_text):
        self.history.append(ConversationTurn("user", user_text))
        response = self.engine.think(user_text)
        rendered = render_template(response.kind, response.payload)
        self.history.append(ConversationTurn("engine", rendered, response))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return rendered, response

    def recent_turns(self, n=10):
        return self.history[-n:]

    def summarize_context(self):
        topics = [t.text for t in self.history if t.speaker == "user"]
        return {
            "topic": self.context.active_topic,
            "slots": dict(self.context.slots),
            "recent_user_turns": topics[-5:],
            "pending_clarification": self.context.pending_clarification,
        }

    def reset(self):
        self.context = DialogueContext()
        self.history = []

