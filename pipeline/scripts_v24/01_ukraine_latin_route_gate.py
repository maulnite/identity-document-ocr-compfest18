#!/usr/bin/env python3
"""V24: release only the validated Ukrainian Latin-name consensus fix.

V23's whole-generic release threshold is intentionally broad.  It diluted a
perfectly validated 20-document Ukrainian layout improvement.  This stage
keeps the same leave-one-out candidates and requires both a safe Ukraine-name
calibration gate and non-regression across every remaining Generic-ID record.
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
RESULTS = PROJECT / "results_v24_ukraine_latin_route_gate"
ROUTE_FIELD = "UKRAINE_PASSPORT:name"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V18 = load_module(V18_PATH, "v18_v24_support")
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
    calibration = V18.calibration_files(consensus)
    gates = V18.derive_gates(base, consensus, calibration)
    candidate = V18.apply_gates(base, consensus, gates)

    validation_files = set(base.loc[
        base.document_family.eq(GENERIC) & ~base.filename.isin(calibration), "filename"
    ])
    before, after = V18.family_summary(base, validation_files), V18.family_summary(candidate, validation_files)
    non_regression = all(
        after[field]["exact_match"] >= before[field]["exact_match"]
        and after[field]["mean_similarity"] >= before[field]["mean_similarity"]
        for field in ("name", "dob") if before[field]["images"]
    )
    route_gate = gates.get(ROUTE_FIELD, {})
    release = bool(route_gate.get("safe", False) and non_regression)
    validation = {
        "scope": "V24 isolated Ukrainian Latin-name consensus, peer-only and route-gated",
        "calibration_documents": len(calibration), "validation_documents": len(validation_files),
        "route_gate": route_gate, "all_gates": gates,
        "generic_baseline": before, "generic_candidate": after,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    released = candidate if release else base.copy()
    save("predictions_candidate", candidate)
    save("predictions_released", released)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "ukraine_name_candidates": int(consensus.loc[consensus.route.eq("UKRAINE_PASSPORT"), "name_consensus"].map(lambda value: bool(text(value))).sum()),
        "ukraine_name_gate_safe": bool(route_gate.get("safe", False)),
        "gpu_evidence_reused": True, "test_release_gate_passed": release,
        "official_submission_overwritten": False,
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V24 Ukrainian Latin-name route gate", "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
