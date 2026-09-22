# Archived duplicate files

These files were moved out of the active codebase on 2026-09-13 because
nothing imported them and they were shadowing the real, active modules —
this was the "Mymodel.py / mymodel.py / mymodel5.py" naming confusion
noted in the project documentation.

| Archived file   | Duplicate of        | Status                                                   |
|------------------|----------------------|-----------------------------------------------------------|
| `Mymodel.py`     | `mymodel.py`         | Different (older) `MyAIModel` implementation, unused      |
| `mymodel5.py`    | `mymodel.py`         | Byte-for-byte identical copy, unused                       |
| `neurol1.py`     | `ai/neural.py`       | Older pre-validation-split copy of the classifier, unused  |

**The active files are:** `mymodel.py` (imported by `test_multimodal.py`)
and `ai/neural.py` (imported by `system.py` and `mymodel.py`).

If you need something from these archived files, pull it into the active
file deliberately — don't import from `archive/`.

## 2026-09-22 cleanup

Two more issues from the same root cause (duplicate/mis-saved files) were
found and fixed:

1. A root-level `llm.py` existed alongside `ai/llm.py`. Nothing imported the
   root copy, but it turned out to be the **newer** of the two (it had
   `wiki()`/`grow()` Wikipedia-fetching support and an improved `synth()`
   that the "active" `ai/llm.py` was missing). It has been merged into
   `ai/llm.py`, which is now the single up-to-date copy. The stale root file
   was removed rather than archived, since archiving the *worse* copy under
   a name matching the *better* one would just recreate the same confusion.
2. A stray file named `test_` (no extension) at the project root turned out
   to be a mis-saved copy of a newer `tests/test_llm.py` — it had 20 more
   lines (two extra test cases covering `wiki()`/`grow()`) than the version
   that was actually in `tests/`. It has been moved to replace
   `tests/test_llm.py` under its correct name. All 9 tests in that file
   pass against the merged `ai/llm.py` above.
