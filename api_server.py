"""A real HTTP API for MYAI — no Flask/FastAPI, just the standard
library's http.server, so it needs zero extra installs on Termux.

Endpoints:
    GET  /health                    -> {"status": "ok"}
    POST /predict/intent            -> {"text": "..."}  => {"intent": ..., "confidence": ...}
    POST /predict/spam              -> {"text": "..."}  => {"label": ...}

Security measures (the project had none of this before):
    - Input length capped (MAX_INPUT_LENGTH) to prevent abuse
    - Control characters stripped from input
    - Simple in-memory rate limiter per client IP
    - JSON parse errors return 400, not a stack trace
    - Every request is logged (path, status, duration)

Run with:  python3 api_server.py [port]
Then:      curl -X POST http://localhost:8000/predict/intent \
                -H "Content-Type: application/json" \
                -d '{"text": "hi there"}'
"""
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict, deque

from core.logging_setup import get_logger, log_prediction, log_error
from system import System
from main import seed_family_knowledge, seed_chatbot_intents
from data.sample_data import load_spam_dataset
from ai.neural import NeuralIntentClassifier

MAX_INPUT_LENGTH = 1000
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60

logger = get_logger("myai.api")


def sanitize_text(text):
    """Strip control characters and cap length — basic input hygiene
    before anything touches a model."""
    if not isinstance(text, str):
        raise ValueError("'text' must be a string")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    if len(cleaned) > MAX_INPUT_LENGTH:
        raise ValueError(f"'text' exceeds max length of {MAX_INPUT_LENGTH} characters")
    if not cleaned.strip():
        raise ValueError("'text' must not be empty")
    return cleaned


class RateLimiter:
    """Fixed-window-ish in-memory rate limiter, keyed by client IP.
    Good enough for a single-process local API; not meant to replace a
    real reverse-proxy rate limiter in production."""

    def __init__(self, max_requests=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)

    def allow(self, client_id):
        now = time.time()
        hits = self._hits[client_id]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


def build_models():
    """Train (or load) the models the API serves. Called once at
    startup, not per-request."""
    brain = System()
    seed_family_knowledge(brain)
    seed_chatbot_intents(brain)
    brain.enable_neural_intents(
        hidden_size=10, epochs=80, learning_rate=0.2,
        model_path="models/intent_classifier.json",
    )

    spam_dataset = load_spam_dataset()
    spam_classifier = NeuralIntentClassifier(hidden_size=8, seed=42)
    for record in spam_dataset:
        spam_classifier.add_example(record["label"], record["text"])
    spam_classifier.train(epochs=40, learning_rate=0.2, validation_split=0.0)

    return brain, spam_classifier


class MYAIRequestHandler(BaseHTTPRequestHandler):
    brain = None
    spam_classifier = None
    rate_limiter = RateLimiter()

    def log_message(self, format, *args):
        pass  # suppress default stderr access logs; we log via `logger` instead

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            raise ValueError("empty request body")
        if length > 10_000:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON: {exc}") from exc

    def _check_rate_limit(self):
        client_ip = self.client_address[0]
        if not self.rate_limiter.allow(client_ip):
            self._send_json(429, {"error": "rate limit exceeded, try again later"})
            return False
        return True

    def do_GET(self):
        start = time.time()
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})
        logger.info(f"GET {self.path} -> {(time.time()-start)*1000:.1f}ms")

    def do_POST(self):
        start = time.time()
        if not self._check_rate_limit():
            return

        try:
            if self.path == "/predict/intent":
                self._handle_intent()
            elif self.path == "/predict/spam":
                self._handle_spam()
            else:
                self._send_json(404, {"error": "not found"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            log_error(logger, "api", exc, context=self.path)
        except Exception as exc:
            self._send_json(500, {"error": "internal server error"})
            log_error(logger, "api", exc, context=self.path)
        finally:
            logger.info(f"POST {self.path} -> {(time.time()-start)*1000:.1f}ms")

    def _handle_intent(self):
        body = self._read_json_body()
        text = sanitize_text(body.get("text", ""))
        start = time.time()
        match = self.brain.neural_intent_classifier.predict(text)
        duration_ms = (time.time() - start) * 1000
        log_prediction(logger, "intent_classifier", text, match.intent_name,
                        confidence=match.score, duration_ms=duration_ms)
        try:
            from learning.auto_learn import learn_intent, learn_fact
            learn_intent(text, self.brain.neural_intent_classifier)
            learn_fact(text, self.brain.add_fact)
        except ImportError:
            pass
        self._send_json(200, {"intent": match.intent_name, "confidence": match.score})

    def _handle_spam(self):
        body = self._read_json_body()
        text = sanitize_text(body.get("text", ""))
        start = time.time()
        match = self.spam_classifier.predict(text)
        duration_ms = (time.time() - start) * 1000
        log_prediction(logger, "spam_classifier", text, match.intent_name,
                        confidence=match.score, duration_ms=duration_ms)
        self._send_json(200, {"label": match.intent_name, "confidence": match.score})


def run_server(port=8000):
    logger.info(f"training/loading models before starting server on port {port}")
    brain, spam_classifier = build_models()
    MYAIRequestHandler.brain = brain
    MYAIRequestHandler.spam_classifier = spam_classifier

    server = HTTPServer(("0.0.0.0", port), MYAIRequestHandler)
    logger.info(f"MYAI API listening on http://0.0.0.0:{port}")
    print(f"MYAI API running at http://localhost:{port}  (Ctrl+C to stop)")
    print("Try: curl -X POST http://localhost:%d/predict/intent -H 'Content-Type: application/json' -d '{\"text\": \"hi there\"}'" % port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
