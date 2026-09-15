"""Tests for the final round: word embeddings, logging, and the API's
input validation / rate limiting logic (not the live HTTP server —
that's covered by the CI smoke test instead).
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.embeddings import WordEmbeddings
from core.logging_setup import get_logger, log_prediction, log_error
from api_server import sanitize_text, RateLimiter, MAX_INPUT_LENGTH


def test_word_embeddings_train_and_vector_shape():
    texts = ["hi there", "hello friend", "goodbye now", "bye friend"]
    emb = WordEmbeddings(dim=8, window=2, negative_samples=2, seed=1)
    emb.fit(texts, epochs=5)
    vec = emb.vector("hi")
    assert vec is not None
    assert len(vec) == 8


def test_word_embeddings_unknown_word_returns_none():
    emb = WordEmbeddings(dim=8, seed=1).fit(["hi there"], epochs=3)
    assert emb.vector("totally_unseen_word_xyz") is None


def test_word_embeddings_most_similar_excludes_self():
    emb = WordEmbeddings(dim=8, seed=1).fit(["hi there hello friend"], epochs=5)
    results = emb.most_similar("hi", top_k=5)
    words = [w for w, _ in results]
    assert "hi" not in words


def test_word_embeddings_sentence_vector_has_correct_dim():
    emb = WordEmbeddings(dim=10, seed=1).fit(["hi there", "bye now"], epochs=5)
    vec = emb.sentence_vector("hi there")
    assert len(vec) == 10


def test_word_embeddings_save_load_roundtrip():
    path = "/tmp/myai_test_embeddings.json"
    emb = WordEmbeddings(dim=6, seed=1).fit(["hi there", "hello friend"], epochs=5)
    emb.save(path)
    reloaded = WordEmbeddings.load(path)
    assert reloaded.vector("hi") == emb.vector("hi")
    os.remove(path)


def test_logger_returns_usable_logger():
    logger = get_logger("myai.test")
    assert logger is not None
    logger.info("test message from test suite")  # should not raise


def test_sanitize_text_strips_control_chars():
    result = sanitize_text("hello\x00world")
    assert "\x00" not in result


def test_sanitize_text_rejects_empty():
    try:
        sanitize_text("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sanitize_text_rejects_oversized_input():
    try:
        sanitize_text("x" * (MAX_INPUT_LENGTH + 1))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "exceeds max length" in str(e)


def test_sanitize_text_rejects_non_string():
    try:
        sanitize_text(12345)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rate_limiter_allows_up_to_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    results = [limiter.allow("test_client") for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_rate_limiter_tracks_clients_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("client_a") is True
    assert limiter.allow("client_b") is True
    assert limiter.allow("client_a") is False
