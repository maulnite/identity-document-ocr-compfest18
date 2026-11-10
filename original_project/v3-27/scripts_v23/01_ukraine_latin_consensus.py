#!/usr/bin/env python3
"""V23: re-evaluate peer consensus after Ukrainian Latin-name routing fix.

The new V17 parser reads the Latin transliteration printed beside Ukrainian
Cyrillic.  This stage only transfers a value when independent scans in the
same identity cluster agree, and evaluates it on held-out labelled documents.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
V18_PATH = PROJECT / "scripts_v18" / "01_group_consensus_generic_fields.py"
V17_RESULTS = PROJECT / "results_v17_per_layout_generic_mrz"
V22_RESULTS = PROJECT / "results_v22_final_permit_ukraine_consensus"
RESULTS = PROJECT / "results_v23_ukraine_latin_consensus"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V18 = load_module(V18_PATH, "v18_v23_support")
V3, V14, core, GENERIC = V18.V3, V18.V14, V18.core, V18.GENERIC


def text(value: Any) -> str:
    return core.output_text(value)


def save(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[c for c in frame if c.endswith("_detail")], errors="ignore"))


def run() -> None:
    base = pd.read_parquet(V22_RESULTS / "predictions_released.parquet").sort_values(
        "filename", key=lambda values: values.map(V3.natural_key)
    ).reset_index(drop=True)
    proposals = pd.read_csv(V17_RESULTS / "field_proposals.csv").fillna("")
    consensus = V18.consensus_candidates(base, proposals)
    V14.atomic_csv(RESULTS / "group_consensus_candidates.csv", consensus)
    print(json.dumps({
        "stage": "ukraine_latin_peer_consensus_prepared",
        "documents": int(len(consensus)),
        "name_candidates": int(consensus.name_consensus.map(lambda value: bool(text(value))).sum()),
        "dob_candidates": int(consensus.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "gpu_evidence_reused": True,
    }, ensure_ascii=False), flush=True)

    calibration = V18.calibration_files(consensus)
    gates = V18.derive_gates(base, consensus, calibration)
    candidate = V18.apply_gates(base, consensus, gates)
    V14.atomic_json(RESULTS / "consensus_gates.json", gates)
    validation_files = set(base.loc[
        base.document_family.eq(GENERIC) & ~base.filename.isin(calibration), "filename"
    ])
    before, after = V18.family_summary(base, validation_files), V18.family_summary(candidate, validation_files)
    non_regression = all(
        after[field]["exact_match"] >= before[field]["exact_match"]
        and after[field]["mean_similarity"] >= before[field]["mean_similarity"]
        for field in ("name", "dob") if before[field]["images"]
    )
    gain = (
        after["name"]["mean_similarity"] - before["name"]["mean_similarity"]
        + after["dob"]["mean_similarity"] - before["dob"]["mean_similarity"]
    )
    release = bool(non_regression and gain >= 0.04)
    validation = {
        "scope": "V23 peer-only consensus after GPU-backed Ukrainian Latin transliteration routing",
        "calibration_documents": len(calibration), "validation_documents": len(validation_files),
        "gates": gates, "generic_baseline": before, "generic_candidate": after,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    released = candidate if release else base.copy()
    save("predictions_candidate", candidate)
    save("predictions_released", released)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "consensus_documents": int(len(consensus)),
        "name_consensus_candidates": int(consensus.name_consensus.map(lambda value: bool(text(value))).sum()),
        "dob_consensus_candidates": int(consensus.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "safe_consensus_routes": int(sum(item["safe"] for item in gates.values())),
        "gpu_evidence_reused": True, "test_release_gate_passed": release,
        "official_submission_overwritten": False,
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V23 Ukrainian Latin-name peer consensus", "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
