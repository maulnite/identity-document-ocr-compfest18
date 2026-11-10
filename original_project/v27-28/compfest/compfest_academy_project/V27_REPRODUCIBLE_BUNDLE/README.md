# Portable V27 Reproduction Bundle

This bundle reproduces the validated downstream V27 release from frozen tabular evidence. It regenerates V24 (Ukraine Latin-route gate), V25 (MyKad address-format gate), V26 (MyKad name-span ranker), and V27 (Portugal/Spain fuzzy peer medoids), then verifies the final output against the retained reference result.

## Scope

This is intentionally **not** a full raw-image OCR rerun. The original detector/recognizer models, image caches, and GPU runtime are not needed for V24--V27 and are not included. The frozen V3 OCR tables, V17 field proposals, and V22 checkpoint are the exact inputs consumed by this final chain.

## Run

```bash
python -m venv .venv
. .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_v27.py
```

A successful run prints `"verification": "passed"`. To regenerate after a previous run, use:

```bash
python run_v27.py --force
```

Only generated `results_v24_*` through `results_v27_*` folders inside this bundle are replaced by `--force`. The retained reference output is never modified.

## Optional V28 Generic-ID improvement

V28 is the newer, validated improvement for route-unknown `GENERIC_ID` records. It preserves V27 and produces `results_v28_generic_unknown_peer_consensus/`:

```bash
python run_v28.py
```

See `reports/V28_GENERIC_ID_EXPERIMENT.md` for the held-out evidence and limitation.

## Included inputs and outputs

- `results_v22_final_permit_ukraine_consensus/predictions_released.parquet`: frozen V22 starting checkpoint.
- `results_v17_per_layout_generic_mrz/field_proposals.csv`: V17 OCR field proposals used by V24 and V27.
- `results_v3_stable/ocr_*.parquet`: three MyKad OCR tables used by V26.
- `results/test_predictions.csv`: retained official test release; it is copied, not changed.
- `reference_outputs/results_v27_fuzzy_peer_medoid_names/`: the best V27 result, including CSV, Parquet, metrics, validation, fuzzy candidates, and test-release copy.
- `reports/`: final metric report, cleanup manifest, and cleanup record.
- `run_v27.py`, `identity_ocr_compfest18.py`, and the required source chain (`scripts_v3`, `scripts_v8`, `scripts_v10`--`scripts_v18`, `scripts_v24`--`scripts_v28`).

## Expected final metrics

- Macro Field EM: 0.6650843881856541
- Full Record Accuracy: 0.37658227848101267
- Normalized Edit Similarity: 0.7970206430331956
- Address-present similarity: 0.8692761684708366
- Official test SHA-256: `d30030f546124d12c6c7e514b8dc702b3631dcf209f8b00a91532400aac60624`

## Privacy

The frozen tables and prediction files include identity-document labels/predictions. Share this bundle only with people who are authorized to access this competition/project data.