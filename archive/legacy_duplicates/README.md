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
