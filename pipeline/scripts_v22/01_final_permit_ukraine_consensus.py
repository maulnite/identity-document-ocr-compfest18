#!/usr/bin/env python3
"""V22: safely extend peer-only consensus to Hong Kong and Ukrainian IDs.

V17 already performed the expensive A100 re-recognition.  This stage starts
from V21, evaluates the new Ukraine/HK proposals using only repeated-identity
peers, and releases a change only when held-out labelled records are not
regressed.  It never alters the official test submission.
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
V21_RESULTS = PROJECT / "results_v21_extended_generic_consensus"
RESULTS = PROJECT / "results_v22_final_permit_ukraine_consensus"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V18 = load_module(V18_PATH, "v18_v22_support")
V3 = V18.V3
V14 = V18.V14
core = V18.core
GENERIC = V18.GENERIC


def text(value: Any) -> str:
    return core.output_text(value)


def atomic_predictions(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(
        RESULTS / f"{name}.csv",
        frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"),
    )


def run() -> None:
    base = pd.read_parquet(V21_RESULTS / "predictions_released.parquet").sort_values(
        "filename", key=lambda values: values.map(V3.natural_key)
    ).reset_index(drop=True)
    proposals = pd.read_csv(V17_RESULTS / "field_proposals.csv").fillna("")
    consensus = V18.consensus_candidates(base, proposals)
    V14.atomic_csv(RESULTS / "group_consensus_candidates.csv", consensus)
    print(json.dumps({
        "stage": "final_permit_ukraine_consensus_prepared",
        "documents": int(len(consensus)),
        "name_candidates": int(consensus.name_consensus.map(lambda value: bool(text(value))).sum()),
        "dob_candidates": int(consensus.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "gpu_evidence_source": str(V17_RESULTS / "gpu_field_lines.csv"),
    }, ensure_ascii=False), flush=True)

    calibration = V18.calibration_files(consensus)
    gates = V18.derive_gates(base, consensus, calibration)
    candidate = V18.apply_gates(base, consensus, gates)
    V14.atomic_json(RESULTS / "consensus_gates.json", gates)

    validation_files = set(base.loc[
        base.document_family.eq(GENERIC) & ~base.filename.isin(calibration), "filename"
    ])
    before = V18.family_summary(base, validation_files)
    after = V18.family_summary(candidate, validation_files)
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
        "scope": "V18 peer-only Generic-ID consensus extended with V17 Hong Kong/Ukraine A100 evidence",
        "calibration_documents": len(calibration),
        "validation_documents": len(validation_files),
        "gates": gates,
        "generic_baseline": before,
        "generic_candidate": after,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    released = candidate if release else base.copy()
    atomic_predictions("predictions_candidate", candidate)
    atomic_predictions("predictions_released", released)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "consensus_documents": int(len(consensus)),
        "name_consensus_candidates": int(consensus.name_consensus.map(lambda value: bool(text(value))).sum()),
        "dob_consensus_candidates": int(consensus.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "safe_consensus_routes": int(sum(item["safe"] for item in gates.values())),
        "gpu_evidence_reused": True,
        "test_release_gate_passed": release,
        "official_submission_overwritten": False,
        "released_generic": V18.family_summary(
            released, set(released.loc[released.document_family.eq(GENERIC), "filename"])
        ),
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V22 final Hong Kong/Ukraine Generic-ID consensus",
        "gpu_evidence_reused": True,
        "official_results_untouched": True,
    }, ensure_ascii=False), flush=True)
    run()
