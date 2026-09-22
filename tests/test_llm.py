import os
import sys
import json
import shutil
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.llm import GPT, BPE, Index, Memory, LLMCore, PRESETS, no_grad, wire, wiki, grow, synth


def tiny_cfg(vocab=40):
    c = dict(PRESETS["tiny"])
    c.update(dim=16, layers=2, heads=2, ctx=12, drop=0.0, vocab_total=vocab)
    return c


class GradientTests(unittest.TestCase):
    def test_backward_matches_numeric(self):
        net = GPT(tiny_cfg(), seed=3, dtype=np.float64)
        rng = np.random.default_rng(1)
        x = rng.integers(0, 40, size=(2, 8))
        y = rng.integers(0, 40, size=(2, 8))
        w = np.ones((2, 8))
        net.loss(x, y, w).backward()
        for name in ("emb", "0.wq", "1.f2", "0.g1", "gf"):
            p = net.P[name]
            for _ in range(4):
                idx = tuple(rng.integers(0, s) for s in p.d.shape)
                old = p.d[idx]
                e = 1e-5
                p.d[idx] = old + e
                with no_grad():
                    up = float(net.loss(x, y, w).d)
                p.d[idx] = old - e
                with no_grad():
                    dn = float(net.loss(x, y, w).d)
                p.d[idx] = old
                num = (up - dn) / (2 * e)
                self.assertAlmostEqual(num, p.g[idx], delta=1e-5 + 1e-3 * abs(num))

    def test_causality(self):
        net = GPT(tiny_cfg(), seed=5, dtype=np.float64)
        a = np.array([[1, 2, 3, 4, 5, 6]])
        b = np.array([[1, 2, 3, 9, 9, 9]])
        with no_grad():
            la, lb = net.forward(a).d, net.forward(b).d
        np.testing.assert_allclose(la[0, :3], lb[0, :3], atol=1e-9)


class TokenizerTests(unittest.TestCase):
    def test_roundtrip_multilingual(self):
        text = "space rockets travel fast. अंतरिक्ष में रॉकेट तेज़ चलते हैं। les fusées voyagent vite. 火箭飞得很快。" * 6
        t = BPE().fit(text, 420)
        for s in ("space rockets", "अंतरिक्ष में रॉकेट", "fusées 火箭 xyz 123 ¿?"):
            self.assertEqual(t.dec(t.enc(s)), s)
        self.assertLess(len(t.enc("space rockets travel fast.")), len("space rockets travel fast.".encode()))


class MemoryTests(unittest.TestCase):
    def test_recall_and_profile(self):
        d = tempfile.mkdtemp()
        try:
            m = Memory(os.path.join(d, "m.json"))
            m.learn("my name is Arjun")
            m.learn("mujhe cricket pasand hai")
            m.add("u", "the eiffel tower is in paris france")
            for i in range(10):
                m.add("u", f"filler talk number {i}")
                m.add("b", f"filler reply {i}")
            self.assertIn("paris", m.recall("where is the eiffel tower")[0])
            self.assertEqual(m.profile["name"], "Arjun")
            again = Memory(os.path.join(d, "m.json"))
            self.assertEqual(len(again.turns), len(m.turns))
        finally:
            shutil.rmtree(d)


class IndexTests(unittest.TestCase):
    def test_bm25_ranks_relevant_first(self):
        ix = Index()
        for s in ("rockets burn fuel to reach orbit", "cats sleep all day", "the ocean is deep and cold"):
            ix.add(s)
        self.assertEqual(ix.query("how do rockets reach orbit")[0][0], "rockets burn fuel to reach orbit")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "knowledge"))
        with open(os.path.join(self.root, "knowledge", "space.txt"), "w", encoding="utf-8") as fh:
            fh.write(
                "Rockets burn fuel to escape earth gravity. The moon orbits the earth every twenty seven days. "
                "Astronauts train for years before they travel to space. The sun is a star at the center of our solar system. "
                "Mars is called the red planet because of iron oxide on its surface. Jupiter is the largest planet in the solar system. "
            ) * 3

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_train_reduces_loss_and_persists(self):
        core = LLMCore("tiny", root=self.root)
        self.assertFalse(core.ready)
        h = core.train(steps=40, batch=4, lr=4e-3, every=20, log=lambda *_: None)
        self.assertLess(h[-1][1], h[0][1] + 1e-9)
        self.assertTrue(core.ready)
        again = LLMCore("tiny", root=self.root)
        self.assertTrue(again.ready)
        self.assertEqual(again.steps, core.steps)
        out = again.respond("tell me about rockets")
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)
        self.assertTrue(os.path.exists(os.path.join(self.root, "ml", "llm_runs.json")))

    def test_wire_routes_unhandled_to_llm(self):
        core = LLMCore("tiny", root=self.root)

        class Sys:
            def respond(self, text):
                if "plus" in text:
                    return {"kind": "handled", "text": "result: 57.0"}
                return {"kind": "unhandled", "text": ""}

        s = Sys()
        wire(s, core)
        self.assertEqual(s.respond("what is 12 plus 45")["text"], "result: 57.0")
        r = s.respond("what is the moon")
        self.assertEqual(r["kind"], "handled")
        self.assertTrue(r["text"])


class GrowTests(unittest.TestCase):
    def test_wiki_parse_and_save(self):
        body = json.dumps({"query": {"pages": {"1": {"extract": "Mars is a planet.\n== History ==\nHumans have watched Mars for thousands of years. It is red."}}}})
        self.assertIn("thousands of years", wiki("Mars", "en", lambda u: body))
        d = tempfile.mkdtemp()
        try:
            r = grow(d, ["Mars", "Moon"], ("en",), fetch=lambda u: body, log=lambda *_: None)
            self.assertEqual(len(r), 2)
            self.assertTrue(os.path.exists(os.path.join(d, "knowledge", "wiki", "en_mars.txt")))
        finally:
            shutil.rmtree(d)

    def test_synth_has_multi_sentence_answers(self):
        kb = Index()
        for i in range(30):
            kb.add(f"Planet number {i} has strange weather patterns. It also has moons around number {i}. Scientists study planet {i} closely.")
        rows = synth(kb, 80, np.random.default_rng(0))
        self.assertTrue(any(len(a.split(". ")) > 1 for _, _, a in rows))


if __name__ == "__main__":
    unittest.main()
