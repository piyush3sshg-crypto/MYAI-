import os
import re
import io
import csv
import sys
import json
import math
import time
import random
import argparse
import urllib.request
import urllib.parse
import unicodedata
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "ai/llm.py (the GPT/transformer module) needs numpy, which is the "
        "one exception to MYAI's zero-dependency design. Install it with:\n"
        "    pip install numpy --break-system-packages\n"
        "The rest of MYAI (main.py, verify.py, data_science_report.py, "
        "ai/neural.py, ai/rnn.py) does not need numpy and will run without it."
    ) from _e

ROOT = os.environ.get("MYAI_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_G = [True]


class no_grad:
    def __enter__(self):
        self.o = _G[0]
        _G[0] = False

    def __exit__(self, *a):
        _G[0] = self.o


def _acc(t, g):
    if t.rg:
        t.g = g if t.g is None else t.g + g


def _unb(g, shape):
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    def __init__(self, d, p=(), rg=False):
        self.d = d
        self.g = None
        self.p = p
        self.f = None
        self.rg = bool(rg or (_G[0] and any(x.rg for x in p)))

    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            n, done = stack.pop()
            if done:
                order.append(n)
                continue
            if id(n) in seen:
                continue
            seen.add(id(n))
            stack.append((n, True))
            for c in n.p:
                if id(c) not in seen:
                    stack.append((c, False))
        self.g = np.ones_like(self.d)
        for n in reversed(order):
            if n.f is not None and n.g is not None and n.rg:
                n.f()


def add(a, b):
    o = Tensor(a.d + b.d, (a, b))

    def f():
        _acc(a, _unb(o.g, a.d.shape))
        _acc(b, _unb(o.g, b.d.shape))

    o.f = f
    return o


def mulc(a, c):
    o = Tensor(a.d * c, (a,))

    def f():
        _acc(a, _unb(o.g * c, a.d.shape))

    o.f = f
    return o


def matmul(a, b):
    o = Tensor(np.matmul(a.d, b.d), (a, b))

    def f():
        g = o.g
        if b.d.ndim == 2 and a.d.ndim > 2:
            if a.rg:
                _acc(a, np.matmul(g, b.d.T))
            if b.rg:
                _acc(b, a.d.reshape(-1, a.d.shape[-1]).T @ g.reshape(-1, g.shape[-1]))
        else:
            if a.rg:
                _acc(a, _unb(np.matmul(g, np.swapaxes(b.d, -1, -2)), a.d.shape))
            if b.rg:
                _acc(b, _unb(np.matmul(np.swapaxes(a.d, -1, -2), g), b.d.shape))

    o.f = f
    return o


def reshape(a, shape):
    o = Tensor(a.d.reshape(shape), (a,))

    def f():
        _acc(a, o.g.reshape(a.d.shape))

    o.f = f
    return o


def transpose(a, axes):
    o = Tensor(a.d.transpose(axes), (a,))
    inv = np.argsort(axes)

    def f():
        _acc(a, o.g.transpose(inv))

    o.f = f
    return o


def softmax(a, mask=None):
    x = a.d if mask is None else a.d + mask
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    y = e / e.sum(-1, keepdims=True)
    o = Tensor(y, (a,))

    def f():
        g = o.g
        _acc(a, y * (g - (g * y).sum(-1, keepdims=True)))

    o.f = f
    return o


def layernorm(x, gm, bt, eps=1e-5):
    mu = x.d.mean(-1, keepdims=True)
    var = x.d.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x.d - mu) * inv
    o = Tensor(xh * gm.d + bt.d, (x, gm, bt))

    def f():
        g = o.g
        n = x.d.shape[-1]
        lead = tuple(range(g.ndim - 1))
        _acc(gm, (g * xh).sum(axis=lead))
        _acc(bt, g.sum(axis=lead))
        dxh = g * gm.d
        s1 = dxh.sum(-1, keepdims=True)
        s2 = (dxh * xh).sum(-1, keepdims=True)
        _acc(x, inv * (dxh - s1 / n - xh * s2 / n))

    o.f = f
    return o


def gelu(a):
    c = math.sqrt(2.0 / math.pi)
    x = a.d
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    o = Tensor(0.5 * x * (1 + t), (a,))

    def f():
        du = c * (1 + 3 * 0.044715 * x ** 2)
        _acc(a, o.g * (0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du))

    o.f = f
    return o


def embed(w, ids):
    o = Tensor(w.d[ids], (w,))

    def f():
        g = np.zeros_like(w.d)
        np.add.at(g, ids, o.g)
        _acc(w, g)

    o.f = f
    return o


def drop(a, p, rng):
    if p <= 0 or not _G[0]:
        return a
    m = ((rng.random(a.d.shape) >= p) / (1.0 - p)).astype(a.d.dtype)
    o = Tensor(a.d * m, (a,))

    def f():
        _acc(a, o.g * m)

    o.f = f
    return o


def rope(x, cos, sin):
    h = x.d.shape[-1] // 2
    r = np.concatenate([-x.d[..., h:], x.d[..., :h]], -1)
    o = Tensor(x.d * cos + r * sin, (x,))

    def f():
        g = o.g
        gs = g * sin
        _acc(x, g * cos + np.concatenate([gs[..., h:], -gs[..., :h]], -1))

    o.f = f
    return o


def xent(logits, tgt, w):
    v = logits.d.shape[-1]
    x = logits.d.reshape(-1, v)
    x = x - x.max(-1, keepdims=True)
    lp = x - np.log(np.exp(x).sum(-1, keepdims=True))
    ar = np.arange(x.shape[0])
    n = max(float(w.sum()), 1.0)
    o = Tensor(np.array(-(lp[ar, tgt] * w).sum() / n, dtype=logits.d.dtype), (logits,))

    def f():
        p = np.exp(lp)
        p[ar, tgt] -= 1.0
        _acc(logits, (p * (w / n)[:, None] * o.g).reshape(logits.d.shape).astype(logits.d.dtype))

    o.f = f
    return o


PRESETS = {
    "tiny": dict(dim=64, layers=2, heads=4, ctx=128, vocab=600, ff=4, drop=0.05, max_new=48),
    "small": dict(dim=128, layers=4, heads=4, ctx=192, vocab=1500, ff=4, drop=0.05, max_new=64),
    "base": dict(dim=256, layers=6, heads=8, ctx=256, vocab=3000, ff=4, drop=0.1, max_new=96),
}


class GPT:
    def __init__(self, cfg, seed=0, dtype=np.float32):
        self.cfg = cfg
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        v, d, n, h, ff = cfg["vocab_total"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"]
        self.P = {}
        std = 0.02
        res = std / math.sqrt(2 * n)

        def w(name, shape, s):
            self.P[name] = Tensor((self.rng.standard_normal(shape) * s).astype(dtype), rg=True)

        def const(name, shape, val):
            self.P[name] = Tensor(np.full(shape, val, dtype=dtype), rg=True)

        w("emb", (v, d), std)
        for i in range(n):
            for k in ("q", "k", "v"):
                w(f"{i}.w{k}", (d, d), std)
            w(f"{i}.wo", (d, d), res)
            w(f"{i}.f1", (d, d * ff), std)
            const(f"{i}.b1", (d * ff,), 0.0)
            w(f"{i}.f2", (d * ff, d), res)
            const(f"{i}.b2", (d,), 0.0)
            const(f"{i}.g1", (d,), 1.0)
            const(f"{i}.c1", (d,), 0.0)
            const(f"{i}.g2", (d,), 1.0)
            const(f"{i}.c2", (d,), 0.0)
        const("gf", (d,), 1.0)
        const("cf", (d,), 0.0)
        hd = d // h
        inv = 1.0 / (10000 ** (np.arange(0, hd, 2) / hd))
        ang = np.outer(np.arange(cfg["ctx"]), inv)
        self.cos = np.concatenate([np.cos(ang)] * 2, -1)[None, None].astype(dtype)
        self.sin = np.concatenate([np.sin(ang)] * 2, -1)[None, None].astype(dtype)

    def count(self):
        return int(sum(p.d.size for p in self.P.values()))

    def forward(self, ids):
        c = self.cfg
        B, T = ids.shape
        d, h = c["dim"], c["heads"]
        hd = d // h
        P = self.P
        x = embed(P["emb"], ids)
        mask = np.triu(np.full((T, T), -1e9, dtype=self.dtype), 1)
        cs, sn = self.cos[:, :, :T], self.sin[:, :, :T]
        scale = 1.0 / math.sqrt(hd)

        def heads(t):
            return transpose(reshape(t, (B, T, h, hd)), (0, 2, 1, 3))

        for i in range(c["layers"]):
            z = layernorm(x, P[f"{i}.g1"], P[f"{i}.c1"])
            q = rope(heads(matmul(z, P[f"{i}.wq"])), cs, sn)
            k = rope(heads(matmul(z, P[f"{i}.wk"])), cs, sn)
            v = heads(matmul(z, P[f"{i}.wv"]))
            att = softmax(mulc(matmul(q, transpose(k, (0, 1, 3, 2))), scale), mask)
            att = drop(att, c["drop"], self.rng)
            y = reshape(transpose(matmul(att, v), (0, 2, 1, 3)), (B, T, d))
            x = add(x, drop(matmul(y, P[f"{i}.wo"]), c["drop"], self.rng))
            z = layernorm(x, P[f"{i}.g2"], P[f"{i}.c2"])
            m = gelu(add(matmul(z, P[f"{i}.f1"]), P[f"{i}.b1"]))
            m = add(matmul(m, P[f"{i}.f2"]), P[f"{i}.b2"])
            x = add(x, drop(m, c["drop"], self.rng))
        x = layernorm(x, P["gf"], P["cf"])
        return matmul(x, transpose(P["emb"], (1, 0)))

    def loss(self, x, y, w):
        return xent(self.forward(x), y.reshape(-1), w.reshape(-1))

    def generate(self, ids, max_new, temp=0.8, top_k=40, top_p=0.92, rep=1.15, stop=(), ban=(), rng=None):
        rng = rng or np.random.default_rng()
        ids, out, lp = list(ids), [], 0.0
        with no_grad():
            for _ in range(max_new):
                ctx = ids[-self.cfg["ctx"]:]
                lg = self.forward(np.array([ctx])).d[0, -1].astype(np.float64)
                for t in set(out[-64:]):
                    lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
                for t in ban:
                    lg[t] = -1e9
                lg = lg / max(temp, 1e-4)
                if top_k and top_k < lg.size:
                    kth = np.partition(lg, -top_k)[-top_k]
                    lg = np.where(lg < kth, -1e9, lg)
                p = np.exp(lg - lg.max())
                p /= p.sum()
                if top_p < 1.0:
                    o = np.argsort(-p)
                    cum = np.cumsum(p[o])
                    keep = cum - p[o] < top_p
                    q = np.zeros_like(p)
                    q[o[keep]] = p[o[keep]]
                    p = q / q.sum()
                t = int(rng.choice(p.size, p=p))
                lp += math.log(max(p[t], 1e-12))
                if t in stop:
                    break
                out.append(t)
                ids.append(t)
        return out, lp / max(len(out), 1)

    def save(self, path):
        np.savez_compressed(path, **{k: v.d for k, v in self.P.items()})

    def load(self, path):
        z = np.load(path)
        for k in self.P:
            self.P[k].d = z[k].astype(self.dtype)


class AdamW:
    def __init__(self, P, b1=0.9, b2=0.95, eps=1e-8):
        self.P, self.b1, self.b2, self.eps, self.t = P, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v.d) for k, v in P.items()}
        self.v = {k: np.zeros_like(v.d) for k, v in P.items()}

    def step(self, lr, wd=0.05, clip=1.0):
        self.t += 1
        gn = math.sqrt(sum(float((p.g ** 2).sum()) for p in self.P.values() if p.g is not None))
        sc = min(1.0, clip / (gn + 1e-6))
        for k, p in self.P.items():
            if p.g is None:
                continue
            g = p.g * sc
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p.d -= (lr * (mh / (np.sqrt(vh) + self.eps) + (wd * p.d if p.d.ndim > 1 else 0))).astype(p.d.dtype)
            p.g = None
        return gn


SYMS = r"!-/:-@\[-`{-~"
PAT = re.compile(rf" ?[^\s{SYMS}]+| ?[{SYMS}]+|\s+")
WORD = re.compile(rf"[^\s{SYMS}]+")
SPL = re.compile(r"(?<=[.!?\u0964])\s+")
SPECIAL = ["<pad>", "<bos>", "<eos>", "<usr>", "<bot>", "<sys>"]


def norm(t):
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFC", t)).strip()


class BPE:
    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.cache = {}
        self._tb()

    def _tb(self):
        self.tb = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.tb.append(self.tb[a] + self.tb[b])

    @property
    def base(self):
        return 256 + len(self.merges)

    @property
    def size(self):
        return self.base + len(SPECIAL)

    def sid(self, name):
        return self.base + SPECIAL.index(name)

    def fit(self, text, target):
        words = Counter(m.group() for m in PAT.finditer(text))
        seqs = [list(w.encode("utf-8")) for w in words]
        freqs = list(words.values())
        pairs, where = Counter(), defaultdict(set)
        for i, sq in enumerate(seqs):
            for p in zip(sq, sq[1:]):
                pairs[p] += freqs[i]
                where[p].add(i)
        self.merges = []
        while 256 + len(self.merges) + len(SPECIAL) < target and pairs:
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            new = 256 + len(self.merges)
            self.merges.append(best)
            for i in list(where[best]):
                sq, f = seqs[i], freqs[i]
                for p in zip(sq, sq[1:]):
                    pairs[p] -= f
                out, j = [], 0
                while j < len(sq):
                    if j < len(sq) - 1 and sq[j] == best[0] and sq[j + 1] == best[1]:
                        out.append(new)
                        j += 2
                    else:
                        out.append(sq[j])
                        j += 1
                seqs[i] = out
                for p in zip(out, out[1:]):
                    pairs[p] += f
                    where[p].add(i)
            pairs = Counter({k: v for k, v in pairs.items() if v > 0})
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.cache = {}
        self._tb()
        return self

    def _word(self, w):
        r = self.cache.get(w)
        if r is not None:
            return r
        sq = list(w.encode("utf-8"))
        while len(sq) > 1:
            best, br = None, 1 << 30
            for p in zip(sq, sq[1:]):
                k = self.rank.get(p)
                if k is not None and k < br:
                    best, br = p, k
            if best is None:
                break
            out, j = [], 0
            while j < len(sq):
                if j < len(sq) - 1 and sq[j] == best[0] and sq[j + 1] == best[1]:
                    out.append(256 + br)
                    j += 2
                else:
                    out.append(sq[j])
                    j += 1
            sq = out
        self.cache[w] = sq
        return sq

    def enc(self, text):
        r = []
        for m in PAT.finditer(text):
            r.extend(self._word(m.group()))
        return r

    def dec(self, ids):
        return b"".join(self.tb[i] for i in ids if i < self.base).decode("utf-8", errors="replace")

    def dump(self):
        return {"merges": [list(m) for m in self.merges]}


class Index:
    def __init__(self):
        self.docs, self.tf, self.lens, self.df, self.total = [], [], [], Counter(), 0

    def add(self, doc):
        w = WORD.findall(doc.lower())
        c = Counter(w)
        self.docs.append(doc)
        self.tf.append(c)
        self.lens.append(len(w))
        self.df.update(c.keys())
        self.total += len(w)

    def idf(self, w):
        n, d = len(self.docs), self.df.get(w, 0)
        return math.log(1 + (n - d + 0.5) / (d + 0.5))

    def query(self, q, k=3, skip=frozenset()):
        if not self.docs:
            return []
        avg = max(self.total / len(self.docs), 1e-9)
        qs = set(WORD.findall(q.lower()))
        sc = []
        for i, c in enumerate(self.tf):
            if i in skip:
                continue
            v = 0.0
            for w in qs:
                f = c.get(w, 0)
                if f:
                    v += self.idf(w) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self.lens[i] / avg))
            if v > 0:
                sc.append((v, i))
        sc.sort(reverse=True)
        return [(self.docs[i], v) for v, i in sc[:k]]


PROFILE_RULES = [
    ("name", re.compile(rf"(?:my name is|i am called|i'm called|mera naam|mera nam|mera name)\s+([^\s{SYMS}]+)", re.I)),
    ("city", re.compile(rf"(?:i live in|i stay in|i am from|main)\s+([^\s{SYMS}]+)\s+(?:mein\s+rehta|mein\s+rehti|mein\s+rahta|mein\s+rahti|se hoon|se hu)|(?:i live in|i stay in|i am from)\s+([^\s{SYMS}]+)", re.I)),
    ("likes", re.compile(rf"(?:i like|i love|mujhe)\s+([^\s{SYMS}]+(?:\s+[^\s{SYMS}]+)?)\s*(?:pasand|hai)?", re.I)),
]


class Memory:
    def __init__(self, path, cap=6000):
        self.path, self.cap = path, cap
        self.turns, self.profile, self.ix = [], {}, Index()
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    z = json.load(fh)
                self.turns, self.profile = z.get("turns", []), z.get("profile", {})
            except (OSError, ValueError):
                self.turns, self.profile = [], {}
        self._reindex()

    def _reindex(self):
        self.ix = Index()
        for t in self.turns:
            self.ix.add(t["t"])

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"turns": self.turns, "profile": self.profile}, fh, ensure_ascii=False)
        os.replace(tmp, self.path)

    def add(self, role, text, src="user"):
        self.turns.append({"r": role, "t": text, "s": src, "ts": time.time()})
        self.ix.add(text)
        if len(self.turns) > self.cap:
            self.turns = self.turns[-self.cap:]
            self._reindex()
        self.save()

    def recent(self, n):
        return self.turns[-n:]

    def recall(self, q, k=2, tail=6):
        skip = set(range(max(len(self.turns) - tail, 0), len(self.turns)))
        return [d for d, _ in self.ix.query(q, k, skip)]

    def learn(self, text):
        for key, rx in PROFILE_RULES:
            m = rx.search(text)
            if not m:
                continue
            val = next((g for g in m.groups() if g), None)
            if not val:
                continue
            if key == "likes":
                cur = self.profile.setdefault("likes", [])
                if val.lower() not in cur:
                    cur.append(val.lower())
                    self.profile["likes"] = cur[-8:]
            else:
                self.profile[key] = val
        self.save()

    def profile_line(self):
        p = self.profile
        parts = []
        if "name" in p:
            parts.append(f"user name: {p['name']}")
        if "city" in p:
            parts.append(f"city: {p['city']}")
        if p.get("likes"):
            parts.append("likes: " + ", ".join(p["likes"]))
        return ". ".join(parts)


SKIP_DIRS = {".git", "__pycache__", "models", "archive", "node_modules", ".github", "logs", "tests", "venv", ".venv"}
SKIP_FILES = {"experiments.json", "llm_conversation.json", "llm_runs.json", "dialogues.jsonl"}
TEXT_EXT = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl"}


def _strings(o, out):
    if isinstance(o, str):
        if len(o.split()) >= 3 or len(o) >= 25:
            out.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            _strings(v, out)
    elif isinstance(o, list):
        if 2 <= len(o) <= 4 and all(isinstance(x, str) and len(x.split()) <= 3 for x in o):
            out.append(" ".join(x.replace("_", " ") for x in o) + ".")
        else:
            for v in o:
                _strings(v, out)


def _read(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    if ext in (".json", ".jsonl"):
        items = []
        try:
            if ext == ".jsonl":
                for ln in raw.splitlines():
                    if ln.strip():
                        _strings(json.loads(ln), items)
            else:
                _strings(json.loads(raw), items)
        except ValueError:
            return norm(raw)
        return "\n".join(items)
    if ext in (".csv", ".tsv"):
        items = []
        for row in csv.reader(io.StringIO(raw), delimiter="\t" if ext == ".tsv" else ","):
            items.extend(c for c in row if len(c.split()) >= 3)
        return "\n".join(items)
    return raw


def harvest(root):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS and not d.startswith(".")]
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if f in SKIP_FILES or os.path.splitext(f)[1].lower() not in TEXT_EXT:
                continue
            try:
                if os.path.getsize(p) > 8_000_000:
                    continue
                t = _read(p)
            except OSError:
                continue
            t = "\n".join(norm(x) for x in t.splitlines() if x.strip())
            if len(t) > 20:
                out.append((os.path.relpath(p, root), t))
    return out


def chunk(text, size=280):
    cur, res = "", []
    for s in SPL.split(text.replace("\n", " ")):
        s = s.strip()
        if not s:
            continue
        if cur and len(cur) + len(s) > size:
            res.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        res.append(cur)
    return res


TEMPL = [
    "{k} kya hai?", "tell me about {k}", "{k} ke baare mein batao", "what do you know about {k}?",
    "explain {k}", "{k} samjhao", "{k} se related batao", "what is {k}",
]


def synth(kb, n, rng):
    docs = [[x for x in SPL.split(c) if 3 <= len(x.split()) <= 40] for c in kb.docs]
    docs = [d for d in docs if d]
    flat = [x for d in docs for x in d]
    if len(flat) < 2:
        return []
    res = []
    for _ in range(n):
        d = docs[int(rng.integers(len(docs)))]
        i = int(rng.integers(len(d)))
        st = " ".join(d[i:i + 2]) if len(d) > 1 and rng.random() < 0.5 else d[i]
        ws = [w for w in WORD.findall(st.lower()) if len(w) > 3]
        if not ws:
            continue
        k = max(ws, key=lambda w: kb.idf(w))
        ds = flat[int(rng.integers(len(flat)))]
        ctx = [st, ds]
        rng.shuffle(ctx)
        res.append((" ".join(ctx), TEMPL[int(rng.integers(len(TEMPL)))].format(k=k), st))
    return res


def _http(url):
    rq = urllib.request.Request(url, headers={"User-Agent": "MYAI/1.0 (personal learning project)"})
    with urllib.request.urlopen(rq, timeout=25) as r:
        return r.read().decode("utf-8")


def wiki(title, lang="en", fetch=None):
    q = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json", "titles": title,
    })
    z = json.loads((fetch or _http)(f"https://{lang}.wikipedia.org/w/api.php?{q}"))
    for pg in z.get("query", {}).get("pages", {}).values():
        t = pg.get("extract", "")
        if t:
            t = re.sub(r"=+[^=\n]+=+", "", t)
            return "\n".join(norm(x) for x in t.splitlines() if len(x.split()) >= 4)
    return ""


def grow(root, topics, langs=("en",), fetch=None, cap=60000, log=print):
    d = os.path.join(root, "knowledge", "wiki")
    os.makedirs(d, exist_ok=True)
    saved = []
    for lang in langs:
        for t in topics:
            t = t.strip()
            if not t:
                continue
            try:
                text = wiki(t, lang, fetch)
            except (OSError, ValueError) as e:
                log(f"skip {lang}:{t} ({e})")
                continue
            if not text:
                log(f"empty {lang}:{t}")
                continue
            name = re.sub(r"\W+", "_", t.lower()).strip("_") or "topic"
            path = os.path.join(d, f"{lang}_{name}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text[:cap])
            saved.append((path, min(len(text), cap)))
            log(f"saved {lang}:{t} {min(len(text), cap)} chars")
    return saved


def assemble(tok, sys_text, hist, q, room):
    sid, bos, usr, bot = tok.sid("<sys>"), tok.sid("<bos>"), tok.sid("<usr>"), tok.sid("<bot>")
    qi = tok.enc(q)[: max(room // 3, 8)]
    si = tok.enc(sys_text)[: int(room * 0.45)] if sys_text else []
    budget = room - len(qi) - len(si) - 4
    hs = []
    for r, t in reversed(hist):
        ti = tok.enc(t)[:96]
        if len(ti) + 1 > budget:
            break
        hs = [usr if r == "u" else bot] + ti + hs
        budget -= len(ti) + 1
    return [bos, sid] + si + hs + [usr] + qi + [bot]


class LLMCore:
    def __init__(self, preset="small", root=None, seed=0):
        self.root = root or ROOT
        self.preset = preset
        self.dir = os.path.join(self.root, "models", "llm")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mem = Memory(os.path.join(self.root, "memory", "llm_conversation.json"))
        self.tok, self.net, self.steps, self.val = None, None, 0, None
        self.kb = Index()
        self.refresh()
        self._load()

    def refresh(self):
        self.sources = harvest(self.root)
        self.kb = Index()
        for _, t in self.sources:
            for c in chunk(t):
                self.kb.add(c)

    def _cfg(self):
        c = dict(PRESETS[self.preset])
        c["vocab_total"] = self.tok.size
        return c

    def _load(self):
        cp, tp, wp = (os.path.join(self.dir, n) for n in ("cfg.json", "tok.json", "w.npz"))
        if not all(os.path.exists(p) for p in (cp, tp, wp)):
            return False
        with open(cp, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        with open(tp, "r", encoding="utf-8") as fh:
            self.tok = BPE(json.load(fh)["merges"])
        self.steps = cfg.pop("steps", 0)
        self.val = cfg.pop("val", None)
        self.preset = cfg.pop("preset", self.preset)
        self.net = GPT(cfg, self.seed)
        self.net.load(wp)
        return True

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        cfg = dict(self.net.cfg)
        cfg["steps"], cfg["preset"], cfg["val"] = self.steps, self.preset, self.val
        with open(os.path.join(self.dir, "cfg.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        with open(os.path.join(self.dir, "tok.json"), "w", encoding="utf-8") as fh:
            json.dump(self.tok.dump(), fh)
        self.net.save(os.path.join(self.dir, "w.npz"))

    def corpus(self):
        return "\n".join(t for _, t in self.sources)

    def dialogs(self):
        res = []
        p = os.path.join(self.root, "data", "dialogues.jsonl")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        z = json.loads(ln)
                        res.append((z.get("sys", ""), z["user"], z["bot"]))
                    except (ValueError, KeyError):
                        continue
        ts = self.mem.turns
        for i in range(1, len(ts)):
            if ts[i]["r"] == "b" and ts[i].get("s") == "tool" and ts[i - 1]["r"] == "u":
                res.append((self.mem.profile_line(), ts[i - 1]["t"], ts[i]["t"]))
        return res

    def build(self):
        text = self.corpus() + "\n" + "\n".join(f"{u} {b}" for _, u, b in self.dialogs())
        if len(text) < 200:
            raise RuntimeError("training text too small: add files to data/ or knowledge/")
        self.tok = BPE().fit(text, PRESETS[self.preset]["vocab"])
        self.net = GPT(self._cfg(), self.seed)
        self.steps, self.val = 0, None

    def _examples(self, nsynth=None):
        ctx = self.net.cfg["ctx"]
        pairs = list(self.dialogs())
        pairs += synth(self.kb, nsynth if nsynth is not None else min(600, 3 * len(self.kb.docs) + 40), self.rng)
        ex = []
        for s, u, b in pairs:
            ans = self.tok.enc(b)[: ctx // 3] + [self.tok.sid("<eos>")]
            pr = assemble(self.tok, s, [], u, ctx - len(ans) - 1)
            ids = pr + ans
            ex.append((np.array(ids, dtype=np.int64), np.array([0] * len(pr) + [1] * len(ans), dtype=np.float32)))
        return ex

    def _lm_ids(self):
        eos = self.tok.sid("<eos>")
        ids = []
        for _, t in self.sources:
            for c in chunk(t, 600):
                ids.extend(self.tok.enc(c))
                ids.append(eos)
        return np.array(ids, dtype=np.int64)

    def _batch(self, ids, ex, B, T, sft):
        if sft and ex:
            pick = [ex[int(i)] for i in self.rng.integers(len(ex), size=B)]
            L = min(max(len(a) for a, _ in pick), T + 1)
            x = np.full((B, L - 1), self.tok.sid("<pad>"), dtype=np.int64)
            y = np.full((B, L - 1), self.tok.sid("<pad>"), dtype=np.int64)
            w = np.zeros((B, L - 1), dtype=np.float32)
            for r, (a, m) in enumerate(pick):
                a, m = a[:L], m[:L]
                n = len(a) - 1
                x[r, :n], y[r, :n], w[r, :n] = a[:-1], a[1:], m[1:]
            return x, y, w
        if len(ids) < T + 2:
            ids = np.tile(ids, (T + 2) // max(len(ids), 1) + 1)
        st = self.rng.integers(0, len(ids) - T - 1, size=B)
        x = np.stack([ids[s:s + T] for s in st])
        y = np.stack([ids[s + 1:s + T + 1] for s in st])
        return x, y, np.ones(y.shape, dtype=np.float32)

    def train(self, steps=600, batch=8, lr=3e-3, sft_ratio=0.5, every=25, log=print):
        if self.net is None:
            self.build()
        ids, ex = self._lm_ids(), self._examples()
        T = self.net.cfg["ctx"]
        cut = max(int(len(ids) * 0.95), T + 2)
        tr, va = ids[:cut], ids[cut:] if len(ids) - cut > T + 2 else ids[:cut]
        opt = AdamW(self.net.P)
        t0, run, hist = time.time(), 0.0, []
        for s in range(1, steps + 1):
            f = min(1.0, s / 40.0) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
            sft = bool(ex) and self.rng.random() < sft_ratio
            x, y, w = self._batch(tr, ex, batch, T, sft)
            L = self.net.loss(x, y, w)
            L.backward()
            opt.step(lr * f)
            run = float(L.d) if s == 1 else 0.95 * run + 0.05 * float(L.d)
            self.steps += 1
            if s % every == 0 or s == steps:
                vx, vy, vw = self._batch(va, [], batch, T, False)
                with no_grad():
                    vl = float(self.net.loss(vx, vy, vw).d)
                self.val = vl
                hist.append((self.steps, round(run, 4), round(vl, 4)))
                log(f"step {self.steps} train {run:.3f} val {vl:.3f} ppl {math.exp(min(vl, 20)):.1f} {time.time() - t0:.0f}s")
                self.save()
        self.save()
        self._log_run(hist, steps, len(ids), len(ex))
        return hist

    def _log_run(self, hist, steps, ntok, nex):
        p = os.path.join(self.root, "ml", "llm_runs.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        runs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    runs = json.load(fh)
            except (OSError, ValueError):
                runs = []
        runs.append({
            "ts": time.time(), "preset": self.preset, "params": self.net.count(), "vocab": self.tok.size,
            "steps": steps, "total_steps": self.steps, "lm_tokens": ntok, "sft_examples": nex,
            "curve": hist[-5:], "final_val": hist[-1][2] if hist else None,
        })
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(runs, fh, ensure_ascii=False)

    @property
    def ready(self):
        return self.net is not None and self.tok is not None and self.steps > 0

    def extractive(self, q):
        hits = self.kb.query(q, 3)
        if not hits:
            return None, 0.0
        qs = set(WORD.findall(q.lower()))
        best, bs = None, -1.0
        for doc, sc in hits:
            for s in SPL.split(doc):
                v = sum(self.kb.idf(w) for w in set(WORD.findall(s.lower())) & qs)
                if v > bs:
                    best, bs = s.strip(), v
        return best, hits[0][1]

    def think(self, text, temp=0.7, min_conf=-3.2):
        rec = self.mem.recall(text, 2)
        docs = [d for d, _ in self.kb.query(text, 2)]
        parts = [x for x in (self.mem.profile_line(), " ".join(docs), " ".join(rec)) if x]
        hist = [(t["r"], t["t"]) for t in self.mem.recent(8)]
        if not self.ready:
            return None, -99.0
        room = self.net.cfg["ctx"] - self.net.cfg["max_new"]
        pr = assemble(self.tok, " | ".join(parts), hist, text, room)
        eos = self.tok.sid("<eos>")
        ban = [self.tok.sid(n) for n in SPECIAL if n != "<eos>"]
        out, lp = self.net.generate(pr, self.net.cfg["max_new"], temp=temp, stop=(eos,), ban=ban, rng=self.rng)
        return norm(self.tok.dec(out)), lp

    def personal(self, text):
        p = self.mem.profile
        q = text.lower()
        if "name" in p and re.search(r"what is my name|who am i|mera naam kya|mera naam batao|do you know my name", q):
            return f"Aapka naam {p['name']} hai."
        if "city" in p and re.search(r"where do i live|mera shehar|mai kahan|main kahan", q):
            return f"Aap {p['city']} mein rehte ho."
        if p.get("likes") and re.search(r"what do i like|mujhe kya pasand", q):
            return "Aapko pasand hai: " + ", ".join(p["likes"]) + "."
        return None

    def respond(self, text, record=True, src="llm"):
        text = norm(text)
        self.mem.learn(text)
        direct = self.personal(text)
        ans, conf = (None, -99.0) if direct else self.think(text)
        ext, kscore = self.extractive(text)
        trusted = self.val is not None and self.val < 2.6
        if direct:
            out, s = direct, "memory"
        elif ans and trusted and conf >= -2.5 and len(ans.split()) >= 2:
            out, s = ans, "llm"
        elif ext and kscore > 1.0:
            out, s = ext, "kb"
        elif ans and trusted:
            out, s = ans, "llm"
        else:
            out, s = "Is sawal ka jawab abhi mere paas nahi hai. Mujhe data/ ya knowledge/ mein jankari do aur train karo.", "none"
        if record:
            self.mem.add("u", text)
            self.mem.add("b", out, s)
        return out

    def teach(self, question, answer):
        p = os.path.join(self.root, "data", "dialogues.jsonl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"sys": self.mem.profile_line(), "user": question, "bot": answer}, ensure_ascii=False) + "\n")

    def status(self):
        return {
            "ready": self.ready, "preset": self.preset, "steps": self.steps,
            "val_loss": self.val, "params": self.net.count() if self.net else 0, "vocab": self.tok.size if self.tok else 0,
            "sources": len(self.sources), "chunks": len(self.kb.docs), "turns": len(self.mem.turns),
            "profile": self.mem.profile,
        }


def _reply_text(r):
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        for k in ("text", "message", "response", "reply", "answer", "result"):
            if k in r:
                return str(r[k])
        return json.dumps(r, ensure_ascii=False, default=str)
    for k in ("text", "message", "response", "reply", "answer", "result"):
        if hasattr(r, k):
            return str(getattr(r, k))
    return str(r)


def _repack(r, out):
    if r is None or isinstance(r, str):
        return out
    if isinstance(r, dict):
        z = dict(r)
        for k in ("text", "message", "response", "reply", "answer", "result"):
            if k in z:
                z[k] = out
                break
        else:
            z["text"] = out
        if "kind" in z:
            z["kind"] = "handled"
        return z
    try:
        for k in ("text", "message", "response", "reply", "answer", "result"):
            if hasattr(r, k):
                setattr(r, k, out)
                break
        if hasattr(r, "kind"):
            r.kind = "handled"
        return r
    except (AttributeError, TypeError):
        return out


def wire(system, core=None, method=None):
    core = core or LLMCore()
    names = (method,) if method else ("respond", "handle", "process", "reply", "ask", "chat", "run", "think", "answer")
    fn, name = None, None
    for n in names:
        c = getattr(system, n, None)
        if callable(c):
            fn, name = c, n
            break
    if fn is None:
        raise AttributeError("no handler method found on system")

    def wrapped(text, *a, **k):
        r = fn(text, *a, **k)
        kind = r.get("kind") if isinstance(r, dict) else getattr(r, "kind", None)
        if r is None or (kind is not None and kind != "handled"):
            return _repack(r, core.respond(text))
        core.mem.learn(text)
        core.mem.add("u", text)
        core.mem.add("b", _reply_text(r)[:400], "tool")
        return r

    setattr(system, name, wrapped)
    system.llm = core
    return core


def chat(core):
    print("MYAI LLM | /teach sawal => jawab | /good | /grow topic1,topic2 | /status | /forget | /exit")
    last = None
    while True:
        try:
            t = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t:
            continue
        if t == "/exit":
            break
        if t == "/status":
            print(json.dumps(core.status(), ensure_ascii=False, indent=2))
        elif t == "/forget":
            core.mem.turns, core.mem.profile = [], {}
            core.mem._reindex()
            core.mem.save()
            print("memory cleared")
        elif t == "/good":
            if last:
                core.teach(*last)
                print("saved as good answer, run: python -m ai.llm train")
            else:
                print("nothing to save yet")
        elif t.startswith("/grow"):
            grow(core.root, t[5:].split(","), ("en",))
            core.refresh()
            print("chunks:", len(core.kb.docs), "| now run: python -m ai.llm train")
        elif t.startswith("/teach") and "=>" in t:
            q, a = t[6:].split("=>", 1)
            core.teach(q.strip(), a.strip())
            print("saved to data/dialogues.jsonl, run: python -m ai.llm train")
        else:
            r = core.respond(t)
            last = (t, r)
            print("ai>", r)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ai.llm")
    ap.add_argument("cmd", choices=["build", "train", "chat", "ask", "status", "grow"])
    ap.add_argument("--langs", default="en")
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    core = LLMCore(a.preset)
    if a.cmd == "build" or (a.cmd == "train" and a.fresh):
        core.build()
        core.save()
        print(json.dumps(core.status(), ensure_ascii=False))
    if a.cmd == "train":
        core.train(a.steps, a.batch, a.lr)
    elif a.cmd == "chat":
        chat(core)
    elif a.cmd == "grow":
        grow(core.root, a.text.split(","), tuple(a.langs.split(",")))
    elif a.cmd == "ask":
        print(core.respond(a.text))
    elif a.cmd == "status":
        print(json.dumps(core.status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
