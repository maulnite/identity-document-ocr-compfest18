# Ordered Python execution

These entry points mirror the requested Phase 1–18 order. Run them from the
project root with the project environment, for example:

```bash
.venv_ocr/bin/python scripts/01_Workspace_Inspection.py
.venv_ocr/bin/python scripts/02_Ground_Truth_Audit_EDA.py
# ...
.venv_ocr/bin/python scripts/18_Final_Conclusions.py
```

All phase files use the central `Config` and the tested implementations in
`identity_ocr_compfest18.py`. They are safe to rerun: expensive outputs use
content-addressed caches, and the final phase refuses to execute the notebook
unless full-data and 100-image test-split coverage checks pass.

The 632 CSV-matched images are the labeled benchmark. Images 633–732 are the
competition test split: they receive full inference but never contribute to
accuracy metrics or GT-dependent configuration selection.
