# MYAI

A personal AI engine built from scratch in pure Python — symbolic
reasoning, planning, a hand-written neural network (real
backpropagation, no PyTorch/TensorFlow), an RNN, a transformer, text
generation/summarization, and a full data-science layer (EDA,
cross-validation, statistical testing, reporting).

**Zero external dependencies for the core system.** `main.py`, `verify.py`,
`data_science_report.py`, `api_server.py`, and everything under `ai/neural.py`,
`ai/rnn.py`, `ai/transformer.py`, `ai/vision.py` run on the Python standard
library alone, so the core project works inside Termux with no `pip install`
required.

**One documented exception:** `ai/llm.py` (the GPT-style tensor/autograd
module, trained on Wikipedia text via `wiki()`/`grow()`) uses `numpy` for
performance. It is not imported by `main.py` or `verify.py`, so you don't
need numpy unless you specifically use `ai/llm.py` or run `tests/test_llm.py`.
If you do:
```bash
pip install numpy --break-system-packages
```

## Quick start

```bash
python3 verify.py              # sanity-check the whole project (100+ checks)
python3 main.py                # run the full reasoning + AI demo
python3 data_science_report.py # run the data-science pipeline + generate a report
```

## What's inside

| Area | Files | What it does |
|---|---|---|
| Symbolic reasoning | `Facts.py`, `rules.py`, `graph_kb.py`, `deduction.py`, `induction.py`, `abduction.py` | Forward-chaining, knowledge graphs, inductive/deductive/abductive reasoning |
| Planning | `planning/` | Blocks-world action planning |
| Neural nets | `ai/neural.py`, `ai/rnn.py`, `ai/transformer.py`, `ai/vision.py` | Hand-written feedforward net, char-RNN with real BPTT, basic transformer, toy vision |
| Language | `parser.py`, `intent.py`, `reader.py`, `writer.py`, `Dialogue.py` | Parsing, intent classification, summarization/Q&A, Markov-chain text generation |
| Data science | `data/`, `analytics/`, `ml/` | Datasets, EDA, TF-IDF, baselines, cross-validation, stats, SVG charts, reports |
| Memory | `memory/` | Episodic memory, working memory |
| Orchestration | `system.py`, `agent.py`, `scheduler.py`, `tools.py` | Ties everything into one `System` object |
| Tests | `tests/`, `verify.py` | Unit tests + a 100-point end-to-end verification script |

Legacy/duplicate files that are no longer imported anywhere live in
`archive/legacy_duplicates/` — see the README there for what they were.

## Data science pipeline

```bash
python3 data_science_report.py
```

This runs, on `data/spam_dataset.csv` (a template-generated 160-row
ham/spam dataset — see the honesty note in `data/sample_data.py`):

1. **EDA** — class balance, missing-value report
2. **Feature engineering** — TF-IDF term importance per class
3. **Cross-validation** — k-fold accuracy/F1
4. **Repeated CV with confidence intervals** — not just one point estimate
5. **Baseline comparison** — is the model actually better than a
   majority-class guess, with statistical significance?
6. **Report generation** — `reports/spam_classifier/report.md` +
   3 SVG charts (bar/heatmap/line — no matplotlib needed)

### Baselines available (`ml/baseline.py`)

- `MajorityClassBaseline` — always predicts the most common class
- `StratifiedRandomBaseline` — random, weighted by class frequency
- `LogisticRegressionBaseline` — real gradient-descent logistic regression over TF-IDF
- `KNNBaseline` — k-nearest-neighbors over TF-IDF cosine similarity

### Hyperparameter tuning (`ml/hyperparameter_search.py`)

```python
from ml.hyperparameter_search import grid_search
results = grid_search(dataset, param_grid={"hidden_size": [4, 8, 16]}, ...)
```

## Model persistence

Trained models can be saved/reloaded instead of retraining every run:

```python
brain.enable_neural_intents(model_path="models/intent_classifier.json")
# first run: trains and saves; later runs: loads instantly
```

`ai/neural.py`, `ai/rnn.py`, `ai/transformer.py`, and `ai/vision.py`
all expose `.save(path)` / `.load(path)`.

## Word embeddings

`ml/embeddings.py` — a real skip-gram model with negative sampling,
trained from scratch (gradient descent, sigmoid, no numpy):

```python
from ml.embeddings import WordEmbeddings
emb = WordEmbeddings(dim=16, window=2, seed=42).fit(texts, epochs=30)
emb.most_similar("father")   # -> [('son', 0.92), ('parent', 0.92), ...]
```

This replaces bag-of-words/TF-IDF (no semantic relationship between
words) with dense vectors that actually cluster related words together.

## REST API

`api_server.py` — a real HTTP API using only the standard library
(`http.server`, no Flask):

```bash
python3 api_server.py 8000
curl -X POST http://localhost:8000/predict/intent \
     -H "Content-Type: application/json" -d '{"text": "hi there"}'
curl -X POST http://localhost:8000/predict/spam \
     -H "Content-Type: application/json" -d '{"text": "click here to claim your free prize"}'
```

Includes: input length limits, control-character stripping, a
per-IP rate limiter (30 req/min by default), and clean 400/429/500
JSON error responses instead of stack traces.

## Logging

`core/logging_setup.py` wires up Python's standard `logging` module —
structured, timestamped logs to `logs/myai.log` (rotated at 1MB, 3
backups kept) plus console output. Used throughout `api_server.py`;
call `get_logger()` anywhere else you want it.

## Continuous integration

`.github/workflows/verify.yml` runs `verify.py`, all unit tests, the
data science pipeline, and an API smoke test on every push — across
Python 3.10/3.11/3.12 — once this repo is pushed to GitHub.

## Running tests

```bash
python3 verify.py   # file existence, imports, compile check, runtime sanity
python3 -c "
import sys; sys.path.insert(0, '.')
import tests.test_core as t1, tests.test_data_science as t2, tests.test_advanced_data_science as t3
for mod in (t1, t2, t3):
    for name in [f for f in dir(mod) if f.startswith('test_')]:
        getattr(mod, name)()
print('all tests passed')
"
```

(If `pytest` is installed — `pip install pytest --break-system-packages`
— you can run `pytest tests/` directly instead.)

## Known limitations (honest, as of the last review)

- Datasets are still small by real-world standards (73–160 rows)
- RNN/transformer are toy-scale — not competitive text generation
- No API/web UI — terminal-only
- No logging/monitoring for long-running or production use
- Feature representation is TF-IDF/bag-of-words — no learned embeddings

See `archive/legacy_duplicates/README.md` for the file-naming cleanup
history, and `ml/experiments.json` for a log of past training runs.
