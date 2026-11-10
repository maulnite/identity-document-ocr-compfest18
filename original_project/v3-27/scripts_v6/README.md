# V6 — fuzzy embedding identity graph

V6 is an address-only experiment that consumes completed V3, V4, and V5
artifacts.  It does not rerun OCR, save face crops, modify the baseline, or
force a document into a hard identity cluster.

For each document pair within the same split, a logistic pair linker receives
the existing OCR-Siamese, DINO document, fusion, and optional ArcFace face
embedding similarities, plus non-ground-truth OCR compatibility signals.  Its
held-out-calibrated output is a soft identity probability.  Only high-precision
edges can vote, but their probabilities remain continuous weights.  Fuzzy
address agreement then pools each document's safe neighbours.  At least two
independent neighbours must support a replacement; ambiguous votes leave the
original V3 address unchanged.  A disjoint identity hold-out must also show a
strict address-EM gain before any candidate is released for the test set.

Run from the project root:

```bash
.venv_ocr/bin/python scripts_v6/01_fuzzy_embedding_address_consensus.py
```

Outputs are stored in `results_v6_fuzzy_embedding/`.  The key files are
`linker_validation.json`, `metrics_fuzzy_address_consensus.json`,
`fuzzy_address_decisions.csv`, and `test_predictions_fuzzy_address_consensus.csv`.
