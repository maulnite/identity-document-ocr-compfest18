#!/usr/bin/env python3
"""V25: held-out-gated MyKad address-format repair.

This repair changes no alphanumeric character: it separates an initial address
label from its house number (for example ``NO.12`` -> ``NO 12``).  It is only
released if a group-disjoint calibration split and its holdout both improve
addresses that are actually present in the ground truth.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V18_PATH = PROJECT / "scripts_v18" / "01_group_consensus_generic_fields.py"
V24_RESULTS = PROJECT / "results_v24_ukraine_latin_route_gate"
RESULTS = PROJECT / "results_v25_mykad_address_format_gate"
SEED = 31


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V18 = load_module(V18_PATH, "v18_v25_support")
V3, V14, core = V18.V3, V18.V14, V18.core


def repair_address(value: Any) -> str:
    """Separate only an initial alpha label and first numeric character."""
    value = core.output_text(value)
    return re.sub(r"^([A-Z]+)[. ]*([0-9])", r"\1 \2", value.upper())


def address_present_metrics(frame: pd.DataFrame, filenames: set[str]) -> dict[str, Any]:
    subset = frame.loc[
        frame.filename.isin(filenames)
        & frame.has_ground_truth.astype(bool)
        & frame.address_gt.map(lambda value: bool(core.output_text(value)))
    ].copy()
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None}
    return {
        "images": int(len(subset)),
        "exact_match": float(subset.address_pred.eq(subset.address_gt).mean()),
        "mean_similarity": float(sum(
            ratio(str(prediction), str(truth))
            for prediction, truth in zip(subset.address_pred, subset.address_gt)
        ) / len(subset) / 100),
    }


def apply_repair(base: pd.DataFrame) -> pd.DataFrame:
    output = base.copy()
    mask = output.document_family.eq("MYKAD")
    output.loc[mask, "address_pred"] = output.loc[mask, "address_pred"].map(repair_address)
    return core.score_prediction_rows(output)


def save(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[c for c in frame if c.endswith("_detail")], errors="ignore"))


def run() -> None:
    base = pd.read_parquet(V24_RESULTS / "predictions_released.parquet").sort_values(
        "filename", key=lambda values: values.map(V3.natural_key)
    ).reset_index(drop=True)
    labelled = base.loc[
        base.document_family.eq("MYKAD")
        & base.has_ground_truth.astype(bool)
        & base.address_gt.map(lambda value: bool(core.output_text(value)))
    ].copy()
    groups = labelled.identity_group.map(lambda value: core.output_text(value) or "__" + str(value))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    calibration_pos, validation_pos = next(splitter.split(labelled, groups=groups))
    calibration_files = set(labelled.iloc[calibration_pos].filename)
    validation_files = set(labelled.iloc[validation_pos].filename)
    candidate = apply_repair(base)
    before_calibration = address_present_metrics(base, calibration_files)
    after_calibration = address_present_metrics(candidate, calibration_files)
    safe = bool(
        before_calibration["images"] >= 20
        and after_calibration["exact_match"] > before_calibration["exact_match"]
        and after_calibration["mean_similarity"] >= before_calibration["mean_similarity"]
    )
    before_validation = address_present_metrics(base, validation_files)
    after_validation = address_present_metrics(candidate, validation_files)
    non_regression = bool(
        after_validation["exact_match"] >= before_validation["exact_match"]
        and after_validation["mean_similarity"] >= before_validation["mean_similarity"]
    )
    release = bool(safe and non_regression and after_validation["exact_match"] > before_validation["exact_match"])
    released = candidate if release else base.copy()
    validation = {
        "scope": "MyKad present-address only; initial alpha-label/house-number separator repair",
        "calibration_documents": len(calibration_files), "validation_documents": len(validation_files),
        "calibration_baseline": before_calibration, "calibration_candidate": after_calibration,
        "validation_baseline": before_validation, "validation_candidate": after_validation,
        "format_gate_safe": safe, "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    save("predictions_candidate", candidate)
    save("predictions_released", released)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    all_present = set(labelled.filename)
    metrics.update({
        "address_present_exact_match": address_present_metrics(released, all_present)["exact_match"],
        "address_present_similarity": address_present_metrics(released, all_present)["mean_similarity"],
        "address_format_gate_safe": safe,
        "gpu_evidence_reused": True,
        "test_release_gate_passed": release,
        "official_submission_overwritten": False,
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V25 MyKad address format gate", "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
