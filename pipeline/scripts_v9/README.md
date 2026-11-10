# V9 — address template audit pack

V9 measures the best address string obtainable by selecting short spans from
the OCR lines that already exist. It is ground-truth analysis only, never test
inference. The outputs show whether the next effort should be parser/layout
rules or recognition/capture quality.

It writes:

- `audit_summary.json` and `template_oracle_summary.csv`
- `oracle_span_audit.csv`
- `review_tasks.csv`, ready for reviewer annotations
- `overlays/`, numbered OCR-box images matching the review CSV

```bash
.venv_ocr/bin/python scripts_v9/01_address_template_audit.py
```
