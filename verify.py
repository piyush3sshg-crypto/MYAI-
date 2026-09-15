import os
import subprocess
import sys


ROOT_FILES = [
    "Datastructures.py",
    "Dialogue.py",
    "Facts.py",
    "abduction.py",
    "agent.py",
    "config.py",
    "deduction.py",
    "general_knowledge.py",
    "graph_kb.py",
    "induction.py",
    "intent.py",
    "main.py",
    "math_engine.py",
    "parser.py",
    "reader.py",
    "rules.py",
    "scheduler.py",
    "system.py",
    "tools.py",
    "writer.py",
]

PACKAGE_FILES = [
    "ai/__init__.py",
    "ai/neural.py",
    "ai/rnn.py",
    "ai/transformer.py",
    "ai/vision.py",
    "core/__init__.py",
    "core/similarity.py",
    "core/tokenizee.py",
    "memory/__init__.py",
    "memory/episodic.py",
    "memory/working_memory.py",
    "data/__init__.py",
    "data/dataset.py",
    "data/sample_data.py",
    "data/spam_dataset.csv",
    "analytics/__init__.py",
    "analytics/eda.py",
    "ml/__init__.py",
    "ml/metrics.py",
    "ml/cross_validation.py",
    "ml/experiment_tracker.py",
    "ml/text_features.py",
    "ml/baseline.py",
    "ml/stats.py",
    "ml/visualize.py",
    "ml/report.py",
    "ml/hyperparameter_search.py",
    "ml/embeddings.py",
    "core/logging_setup.py",
    "api_server.py",
    "data_science_report.py",
    "README.md",
    "planning/__init__.py",
    "planning/action.py",
    "planning/planner.py",
    "planning/robust_planner.py",
]

passed = 0
failed = 0


def check(label, ok, extra=""):
    global passed, failed

    if ok:
        print(f"[OK]   {label}")
        passed += 1
    else:
        print(f"[FAIL] {label}")
        if extra:
            print(f"       {extra}")
        failed += 1


print("=== MYAI PROJECT VERIFICATION ===")

print("\n=== STEP 1: File existence ===")
for path in ROOT_FILES + PACKAGE_FILES:
    check(f"exists: {path}", os.path.isfile(path))


print("\n=== STEP 2: Python compile check ===")
result = subprocess.run(
    [sys.executable, "-m", "compileall", "-q", "."],
    capture_output=True,
    text=True,
)
check(
    "all Python files compile",
    result.returncode == 0,
    result.stderr.strip(),
)


print("\n=== STEP 3: Core import check ===")
modules = [
    "Facts",
    "rules",
    "graph_kb",
    "induction",
    "math_engine",
    "parser",
    "intent",
    "reader",
    "writer",
    "system",
    "planning.action",
    "planning.planner",
    "planning.robust_planner",
    "memory.working_memory",
    "memory.episodic",
    "core.similarity",
    "core.tokenizee",
    "ai.neural",
    "ai.rnn",
    "ai.transformer",
    "ai.vision",
    "data.dataset",
    "data.sample_data",
    "analytics.eda",
    "ml.metrics",
    "ml.cross_validation",
    "ml.experiment_tracker",
    "ml.text_features",
    "ml.baseline",
    "ml.stats",
    "ml.visualize",
    "ml.report",
    "ml.hyperparameter_search",
    "ml.embeddings",
    "core.logging_setup",
]

for module in modules:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    check(
        f"imports: {module}",
        result.returncode == 0,
        result.stderr.strip(),
    )


print("\n=== STEP 4: main.py end-to-end ===")
result = subprocess.run(
    [sys.executable, "main.py"],
    capture_output=True,
    text=True,
    timeout=300,
)

check(
    "main.py exits cleanly",
    result.returncode == 0,
    result.stderr.strip(),
)

expected_sections = [
    "Family Reasoning",
    "Knowledge Graph Closure",
    "Blocks World Planning",
    "Dialogue Simulation",
    "Inductive Learning Demo",
    "Reflection",
    "Neural Intent Classifier",
    "Writer: Text Generation",
    "Neural Writer",
    "Reader: Summarization + Q&A",
]

for section in expected_sections:
    check(
        f"main.py contains: {section}",
        section in result.stdout,
    )


print("\n=== STEP 5: Runtime output sanity ===")
check(
    "family reasoning produced output",
    "grandparent:" in result.stdout,
)

check(
    "planner produced an action",
    "action:" in result.stdout,
)

check(
    "math reasoning works",
    "x = 3.0" in result.stdout,
)

check(
    "neural intent classifier trained",
    "training report:" in result.stdout,
)

check(
    "reader produced summary",
    "summary:" in result.stdout,
)


print("\n" + "=" * 50)
print(f"TOTAL PASSED: {passed}")
print(f"TOTAL FAILED: {failed}")

if failed == 0:
    print("MYAI VERIFICATION PASSED")
else:
    print("MYAI VERIFICATION FAILED")

print("=" * 50)

sys.exit(0 if failed == 0 else 1)
