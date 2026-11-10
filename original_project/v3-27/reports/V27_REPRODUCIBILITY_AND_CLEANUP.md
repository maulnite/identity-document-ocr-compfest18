# V27 Reproducibility and Cleanup Manifest

## Retained final chain

The following artifacts are retained because V27 depends on them directly or they contain the frozen GPU/OCR evidence needed to rerun the final stages:

- `results/`
- `results_v3_stable/`
- `results_v16_all_malaysian_address_cards/`
- `results_v17_per_layout_generic_mrz/`
- `results_v22_final_permit_ukraine_consensus/`
- `results_v24_ukraine_latin_route_gate/`
- `results_v25_mykad_address_format_gate/`
- `results_v26_mykad_name_span_ranker/`
- `results_v27_fuzzy_peer_medoid_names/`
- `scripts_v3/`, `scripts_v8/`, `scripts_v10/` through `scripts_v18/`, `scripts_v22/`, `scripts_v24/` through `scripts_v27/`
- `identity_ocr_compfest18.py`, `third_party/`, `.venv_ocr/`, `config/`, `tools/`, `figures/`, `archive/`, and `FINAL PROJECT DSA/`.

The V27 pipeline can be regenerated from the frozen V22 predecessor without rerunning full OCR:

```bash
.venv_ocr/bin/python scripts_v24/01_ukraine_latin_route_gate.py
.venv_ocr/bin/python scripts_v25/01_mykad_address_format_gate.py
.venv_ocr/bin/python scripts_v26/01_mykad_name_span_ranker.py
.venv_ocr/bin/python scripts_v27/01_fuzzy_peer_medoid_names.py
```

V17 GPU evidence is retained in `results_v17_per_layout_generic_mrz/`. The heavy raw/preprocessing cache is not needed to inspect or reproduce the frozen V27 outputs.

## Removed artifacts

The cleanup removes superseded result folders, experiments that do not feed V27, their matching non-dependency scripts, generated Python bytecode, old run logs, obsolete experiment caches, the disposable preprocessing cache, and the unused docaligner virtual environment.

Removing the 27 GB preprocessing cache means a future full raw-image OCR rebuild would need to regenerate preprocessing assets. It does not change V27 results, reports, released benchmark CSV/Parquet, or the official test output.

## Cleanup execution

Completed on the A100 workspace on 2026-08-15. It removed 33 superseded folders and released approximately 29,301,654,598 bytes (27.29 GiB) before the small log and bytecode cleanup. The exact remote record is `reports/CLEANUP_V27_RESULT.md`.
