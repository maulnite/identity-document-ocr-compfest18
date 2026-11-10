#!/usr/bin/env python3
"""V20: peer-only Chile-ID birth-date consensus using V17 GPU OCR evidence."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
V17_PATH = PROJECT / "scripts_v17" / "01_per_layout_generic_mrz_gpu.py"
V17_RESULTS = PROJECT / "results_v17_per_layout_generic_mrz"
V19_RESULTS = PROJECT / "results_v19_spain_component_consensus"
RESULTS = PROJECT / "results_v20_chile_dob_consensus"
ROUTE = "CHILE_ID"
MIN_SUPPORT = 2
MIN_DOMINANCE = 0.60


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V17 = load_module(V17_PATH, "v17_v20_support")
core = V17.core
V3 = V17.V3
V14 = V17.V14


def text(value: Any) -> str:
    return core.output_text(value)


def consensus(base: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    frame = base.merge(proposals, on="filename", how="inner")
    frame = frame.loc[(frame.route == ROUTE) & frame.identity_group.map(lambda value: bool(text(value)))].copy()
    records: list[dict[str, Any]] = []
    for identity_group, group in frame.groupby("identity_group", sort=True):
        rows = list(group.itertuples(index=False))
        for target in rows:
            values = [V17.normalize_date_extended(peer.dob_candidate) for peer in rows if peer.filename != target.filename]
            values = [value for value in values if len(value) == 10]
            if values:
                winner, support = Counter(values).most_common(1)[0]
                dominance = support / len(values)
            else:
                winner, support, dominance = "", 0, 0.0
            accepted = support >= MIN_SUPPORT and dominance >= MIN_DOMINANCE
            records.append({
                "filename": target.filename, "route": ROUTE, "identity_group": identity_group,
                "peer_documents": max(0, len(rows) - 1), "dob_consensus": winner if accepted else "",
                "support": support, "valid_peer_dates": len(values), "dominance": dominance,
            })
    return pd.DataFrame(records)


def metric(frame: pd.DataFrame, filenames: set[str]) -> dict[str, Any]:
    return V17.field_metrics(frame, filenames, "dob")


def score(base: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    output = base.copy()
    mask = output.filename.isin(values.index)
    output.loc[mask, "birth_date_pred"] = output.loc[mask, "filename"].map(values).fillna("")
    return core.score_prediction_rows(output)


def run() -> None:
    base = pd.read_parquet(V19_RESULTS / "predictions_released.parquet").sort_values("filename", key=lambda values: values.map(V3.natural_key)).reset_index(drop=True)
    proposals = pd.read_csv(V17_RESULTS / "field_proposals.csv").fillna("")
    candidates = consensus(base, proposals)
    V14.atomic_csv(RESULTS / "chile_dob_consensus_candidates.csv", candidates)
    print(json.dumps({
        "stage": "chile_dob_consensus_prepared", "documents": int(len(candidates)),
        "candidates": int(candidates.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "gpu_evidence_source": str(V17_RESULTS / "gpu_field_lines.csv"),
    }, ensure_ascii=False), flush=True)

    filenames = sorted(set(candidates.loc[candidates.dob_consensus.map(lambda value: bool(text(value))), "filename"]))
    calibration_count = min(len(filenames), max(3, math.ceil(len(filenames) / 3)))
    calibration = set(filenames[:calibration_count])
    calibration_values = candidates.loc[
        candidates.filename.isin(calibration) & candidates.dob_consensus.map(lambda value: bool(text(value))),
        ["filename", "dob_consensus"],
    ].set_index("filename").dob_consensus
    before_calibration = metric(base, set(calibration_values.index))
    calibration_trial = score(base, calibration_values) if len(calibration_values) else base
    after_calibration = metric(calibration_trial, set(calibration_values.index))
    safe = bool(
        before_calibration["images"] >= 3 and
        after_calibration["exact_match"] >= before_calibration["exact_match"] and
        after_calibration["mean_similarity"] >= before_calibration["mean_similarity"] + 0.04
    )
    gate = {"route": ROUTE, "field": "dob", "safe": safe, "baseline": before_calibration, "candidate": after_calibration}

    release_values = candidates.loc[
        candidates.dob_consensus.map(lambda value: bool(text(value))), ["filename", "dob_consensus"]
    ].set_index("filename").dob_consensus
    candidate = score(base, release_values) if safe else base.copy()
    validation_files = set(base.loc[base.document_family.eq("GENERIC_ID") & ~base.filename.isin(calibration), "filename"])
    before = {field: V17.field_metrics(base, validation_files, field) for field in ("name", "dob")}
    after = {field: V17.field_metrics(candidate, validation_files, field) for field in ("name", "dob")}
    non_regression = all(
        after[field]["exact_match"] >= before[field]["exact_match"] and after[field]["mean_similarity"] >= before[field]["mean_similarity"]
        for field in ("name", "dob") if before[field]["images"]
    )
    release = bool(non_regression and safe)
    validation = {
        "scope": "Chile-ID peer-only birth-date consensus on top of V19",
        "min_peer_support": MIN_SUPPORT, "min_dominance": MIN_DOMINANCE,
        "calibration_documents": len(calibration), "validation_documents": len(validation_files),
        "gate": gate, "generic_baseline": before, "generic_candidate": after,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    released = candidate if release else base.copy()
    for name, frame in (("predictions_candidate", candidate), ("predictions_released", released)):
        V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
        V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"))
    V14.atomic_json(RESULTS / "dob_gate.json", gate)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "chile_documents": int(len(candidates)),
        "dob_consensus_candidates": int(candidates.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "safe_component_routes": int(safe), "gpu_evidence_reused": True,
        "test_release_gate_passed": release, "official_submission_overwritten": False,
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V20 Chile DOB consensus", "gpu_evidence_reused": True, "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
