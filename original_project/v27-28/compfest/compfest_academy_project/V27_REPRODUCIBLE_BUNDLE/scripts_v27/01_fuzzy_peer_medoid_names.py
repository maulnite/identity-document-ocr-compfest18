#!/usr/bin/env python3
"""V27: route-gated fuzzy leave-one-out name medoids for repeated identities.

Portugal citizen cards and Spanish DNI scans can contain complementary OCR
character errors.  Exact-vote consensus discards them.  This stage selects the
peer-only string with the highest average normalized similarity (a fuzzy
medoid), never allowing a target image to vote for itself.
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


PROJECT = Path(__file__).resolve().parents[1]
V18_PATH = PROJECT / "scripts_v18" / "01_group_consensus_generic_fields.py"
V17_RESULTS = PROJECT / "results_v17_per_layout_generic_mrz"
V26_RESULTS = PROJECT / "results_v26_mykad_name_span_ranker"
RESULTS = PROJECT / "results_v27_fuzzy_peer_medoid_names"
ROUTES = ("PORTUGAL_CITIZEN_CARD", "SPAIN_ID")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V18 = load_module(V18_PATH, "v18_v27_support")
V3, V14, core = V18.V3, V18.V14, V18.core


def text(value: Any) -> str:
    return core.output_text(value)


def canonical(value: Any) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", text(value).upper()).strip()


def medoid(values: list[str]) -> str:
    values = [text(value) for value in values if canonical(value)]
    if not values:
        return ""
    return max(values, key=lambda value: sum(ratio(canonical(value), canonical(peer)) for peer in values) / len(values))


def candidates(base: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    frame = base.merge(proposals.loc[proposals.route.isin(ROUTES), ["filename", "route", "name_candidate"]], on="filename", how="inner")
    frame = frame.loc[frame.identity_group.map(lambda value: bool(text(value)))].copy()
    records: list[dict[str, Any]] = []
    for (route, identity_group), group in frame.groupby(["route", "identity_group"], sort=True):
        rows = list(group.itertuples(index=False))
        for target in rows:
            peers = [peer.name_candidate for peer in rows if peer.filename != target.filename]
            value = medoid(peers)
            records.append({
                "filename": target.filename, "route": route, "identity_group": identity_group,
                "peer_documents": len(rows) - 1, "name_medoid": value,
            })
    return pd.DataFrame(records)


def apply(base: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    output = base.copy()
    mask = output.filename.isin(values.index)
    output.loc[mask, "name_pred"] = output.loc[mask, "filename"].map(values).fillna("")
    return core.score_prediction_rows(output)


def metric(frame: pd.DataFrame, filenames: set[str]) -> dict[str, Any]:
    subset = frame.loc[frame.filename.isin(filenames) & frame.has_ground_truth.astype(bool)].copy()
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None}
    return {
        "images": int(len(subset)), "exact_match": float(subset.name_pred.eq(subset.name_gt).mean()),
        "mean_similarity": float(sum(ratio(str(pred), str(gt)) for pred, gt in zip(subset.name_pred, subset.name_gt)) / len(subset) / 100),
    }


def save(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"))


def run() -> None:
    base = pd.read_parquet(V26_RESULTS / "predictions_released.parquet").sort_values(
        "filename", key=lambda values: values.map(V3.natural_key)
    ).reset_index(drop=True)
    proposals = pd.read_csv(V17_RESULTS / "field_proposals.csv").fillna("")
    peer_candidates = candidates(base, proposals)
    V14.atomic_csv(RESULTS / "fuzzy_peer_medoid_candidates.csv", peer_candidates)
    gates: dict[str, Any] = {}
    accepted_values: list[pd.Series] = []
    for route in ROUTES:
        routed = peer_candidates.loc[(peer_candidates.route.eq(route)) & peer_candidates.name_medoid.map(lambda value: bool(text(value)))].copy()
        files = sorted(set(routed.filename))
        calibration = set(files[::3])
        validation = set(files) - calibration
        values = routed.set_index("filename").name_medoid
        trial = apply(base, values)
        before_calibration, after_calibration = metric(base, calibration), metric(trial, calibration)
        before_validation, after_validation = metric(base, validation), metric(trial, validation)
        safe = bool(
            before_calibration["images"] >= 3
            and after_calibration["exact_match"] >= before_calibration["exact_match"]
            and after_calibration["mean_similarity"] >= before_calibration["mean_similarity"] + 0.04
            and after_validation["exact_match"] >= before_validation["exact_match"]
            and after_validation["mean_similarity"] >= before_validation["mean_similarity"]
        )
        gates[route] = {
            "safe": safe, "candidate_documents": len(files),
            "calibration_baseline": before_calibration, "calibration_candidate": after_calibration,
            "validation_baseline": before_validation, "validation_candidate": after_validation,
        }
        if safe:
            accepted_values.append(values)
    all_values = pd.concat(accepted_values).loc[lambda series: ~series.index.duplicated(keep="last")] if accepted_values else pd.Series(dtype=str)
    candidate = apply(base, all_values) if len(all_values) else base.copy()
    release = bool(accepted_values)
    released = candidate if release else base.copy()
    validation = {
        "scope": "Portugal/Spain fuzzy leave-one-out peer medoids", "routes": gates,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    save("predictions_candidate", candidate)
    save("predictions_released", released)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    previous = json.loads((V26_RESULTS / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({
        "fuzzy_medoid_safe_routes": int(sum(gate["safe"] for gate in gates.values())),
        "address_present_documents": previous.get("address_present_documents"),
        "address_present_exact_match": previous.get("address_present_exact_match"),
        "address_present_similarity": previous.get("address_present_similarity"),
        "gpu_evidence_reused": True, "test_release_gate_passed": release,
        "official_submission_overwritten": False,
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V27 fuzzy leave-one-out peer medoid names", "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
