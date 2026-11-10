#!/usr/bin/env python3
"""Run and verify the portable downstream V27 reproduction.

This executes the frozen V24 -> V25 -> V26 -> V27 stages from the bundled
V22/V17/V3 tables. Raw-image OCR, models, and a GPU are deliberately not used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
GENERATED = (
    "results_v24_ukraine_latin_route_gate",
    "results_v25_mykad_address_format_gate",
    "results_v26_mykad_name_span_ranker",
    "results_v27_fuzzy_peer_medoid_names",
)
STAGES = (
    "scripts_v24/01_ukraine_latin_route_gate.py",
    "scripts_v25/01_mykad_address_format_gate.py",
    "scripts_v26/01_mykad_name_span_ranker.py",
    "scripts_v27/01_fuzzy_peer_medoid_names.py",
)
REFERENCE = ROOT / "reference_outputs" / "results_v27_fuzzy_peer_medoid_names"


def install_ocr_stubs() -> None:
    """The imported historic code defines GPU classes but never executes them."""
    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Module=object)
    torch.Tensor = object
    torch.no_grad = lambda: (lambda function: function)
    sys.modules["torch"] = torch
    sys.modules["cv2"] = types.ModuleType("cv2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> None:
    actual_path = ROOT / GENERATED[-1] / "predictions_released.parquet"
    expected_path = REFERENCE / "predictions_released.parquet"
    actual, expected = pd.read_parquet(actual_path), pd.read_parquet(expected_path)
    columns = [
        "filename", "name_pred", "birth_date_pred", "address_pred", "name_exact", "dob_exact", "address_exact",
        "macro_field_exact", "full_record_correct", "record_cer", "normalized_edit_similarity",
    ]
    actual, expected = actual[columns].reset_index(drop=True), expected[columns].reset_index(drop=True)
    if not actual.equals(expected):
        differing = int(actual.ne(expected).any(axis=1).sum())
        raise RuntimeError(f"Reproduced V27 differs from the retained reference in {differing} rows")
    actual_metrics = json.loads((ROOT / GENERATED[-1] / "metrics.json").read_text(encoding="utf-8"))
    expected_metrics = json.loads((REFERENCE / "metrics.json").read_text(encoding="utf-8"))
    if actual_metrics != expected_metrics:
        raise RuntimeError("Reproduced V27 metrics differ from the retained reference")
    official_hash = sha256(ROOT / "results" / "test_predictions.csv")
    if sha256(ROOT / GENERATED[-1] / "test_predictions.csv") != official_hash:
        raise RuntimeError("The copied final test release differs from the retained official file")
    print(json.dumps({"verification": "passed", "official_test_sha256": official_hash, "labeled_records": 632}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace only generated V24--V27 folders inside this bundle")
    parser.add_argument("--no-verify", action="store_true", help="do not compare the generated V27 result with the retained reference")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    existing = [ROOT / name for name in GENERATED if (ROOT / name).exists()]
    if existing and not args.force:
        raise SystemExit("Generated folders already exist. Run again with --force.")
    for path in existing:
        shutil.rmtree(path)
    install_ocr_stubs()
    for relative_path in STAGES:
        print(f"Running {relative_path}", flush=True)
        runpy.run_path(str(ROOT / relative_path), run_name="__main__")
    if not args.no_verify:
        verify()


if __name__ == "__main__":
    main()