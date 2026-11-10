# V8 — weakly supervised address span selector

V8 uses completed V3 OCR results and labelled address strings only to make
training labels for individual OCR lines.  Its inference features are OCR text,
confidence, geometry, document family, and quality group—not ground truth.
It trains and validates by identity, selects one compact multi-line address
span, and releases test candidates only after a held-out safety gate passes.

It deliberately does not rerun OCR: V7 established that recognition can see
more address-like text, while V8 targets the remaining line-boundary problem.

```bash
.venv_ocr/bin/python scripts_v8/01_weakly_supervised_address_span_selector.py
```
