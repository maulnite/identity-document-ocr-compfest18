#!/usr/bin/env python3
"""V18: leave-one-document-out identity-group consensus for Generic IDs.

V17 performs the expensive A100 re-recognition and produces field candidates
for each recognised document layout.  This stage addresses a different failure
mode: several scans of the *same visual identity cluster* contain unreadable
fields, while one or more peer scans contain a stable OCR reading.  A target
never votes for itself; a value is transferred only when at least two peers in
the same visual cluster and card layout agree, with at least 60% dominance.

The approach intentionally leaves MRZ, MyKad/MyPR, address fields, unknown
layouts, and the official test submission untouched.  It consumes V17's GPU
SVTRv2 evidence rather than re-running OCR: consensus and scoring are small
tabular operations for which CUDA would provide no benefit.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
V17_PATH = PROJECT / "scripts_v17" / "01_per_layout_generic_mrz_gpu.py"
V17_RESULTS = PROJECT / "results_v17_per_layout_generic_mrz"
RESULTS = PROJECT / "results_v18_group_consensus_generic"
GENERIC = "GENERIC_ID"
MIN_PEER_SUPPORT = 2
MIN_DOMINANCE = 0.60


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V17 = load_module(V17_PATH, "v17_v18_support")
core = V17.core
V3 = V17.V3
V14 = V17.V14


def text(value: Any) -> str:
    return core.output_text(value)


def canonical(field: str, value: Any) -> str:
    value = text(value)
    if field == "name":
        return V17.compact(value)
    return V17.normalize_date_extended(value)


def valid_value(field: str, value: Any) -> bool:
    if field == "name":
        return V17.alpha_text(value)
    normalized = V17.normalize_date_extended(value)
    return bool(normalized and len(normalized) == 10)


def representative(values: list[str], field: str) -> str:
    """Keep the most frequent surface spelling inside an agreed canonical bin."""
    if field == "dob":
        return canonical(field, values[0])
    counts = Counter(text(value) for value in values)
    return counts.most_common(1)[0][0]


def consensus_candidates(base: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    """Produce peer-only candidates keyed by route and visual identity cluster."""
    columns = ["filename", "route", "identity_group", "name_candidate", "dob_candidate"]
    merged = base.merge(proposals, on="filename", how="inner")
    merged = merged.loc[
        merged.document_family.eq(GENERIC) & merged.identity_group.map(lambda value: bool(text(value))),
        columns,
    ].copy()
    records: list[dict[str, Any]] = []
    for (route, identity_group), group in merged.groupby(["route", "identity_group"], sort=True):
        rows = list(group.itertuples(index=False))
        for target in rows:
            record: dict[str, Any] = {
                "filename": target.filename,
                "route": route,
                "identity_group": identity_group,
                "peer_documents": max(0, len(rows) - 1),
            }
            for field, column in (("name", "name_candidate"), ("dob", "dob_candidate")):
                bins: dict[str, list[str]] = {}
                for peer in rows:
                    if peer.filename == target.filename:
                        continue
                    value = getattr(peer, column)
                    if not valid_value(field, value):
                        continue
                    key = canonical(field, value)
                    if key:
                        bins.setdefault(key, []).append(text(value))
                if not bins:
                    record.update({f"{field}_consensus": "", f"{field}_support": 0, f"{field}_valid_peers": 0, f"{field}_dominance": 0.0})
                    continue
                winner_key, winner_values = max(bins.items(), key=lambda item: (len(item[1]), item[0]))
                valid_peers = sum(len(values) for values in bins.values())
                support = len(winner_values)
                dominance = support / valid_peers
                accepted = support >= MIN_PEER_SUPPORT and dominance >= MIN_DOMINANCE
                record.update({
                    f"{field}_consensus": representative(winner_values, field) if accepted else "",
                    f"{field}_support": support,
                    f"{field}_valid_peers": valid_peers,
                    f"{field}_dominance": dominance,
                    f"{field}_winner_key": winner_key,
                })
            records.append(record)
    return pd.DataFrame(records)


def calibration_files(consensus: pd.DataFrame) -> set[str]:
    """Deterministic document-level split; peer evidence remains self-excluded."""
    selected: set[str] = set()
    for route, group in consensus.groupby("route", sort=True):
        candidates = group.loc[
            group.name_consensus.map(lambda value: bool(text(value))) |
            group.dob_consensus.map(lambda value: bool(text(value))),
            "filename",
        ].sort_values().tolist()
        # Every third filename is calibration.  The split uses no label/value
        # information and works even when a document layout has one large
        # identity cluster, unlike a group-held-out split.
        selected.update(candidates[::3])
    return selected


def metric(frame: pd.DataFrame, filenames: set[str], field: str) -> dict[str, Any]:
    return V17.field_metrics(frame, filenames, field)


def score_trial(base: pd.DataFrame, values: pd.Series, field: str) -> pd.DataFrame:
    output = base.copy()
    prediction_column = "name_pred" if field == "name" else "birth_date_pred"
    mask = output.filename.isin(values.index)
    output.loc[mask, prediction_column] = output.loc[mask, "filename"].map(values).fillna("")
    return core.score_prediction_rows(output)


def derive_gates(base: pd.DataFrame, consensus: pd.DataFrame, calibration: set[str]) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    for route, group in consensus.groupby("route", sort=True):
        for field, column in (("name", "name_consensus"), ("dob", "dob_consensus")):
            routed = group.loc[
                group.filename.isin(calibration) & group[column].map(lambda value: bool(text(value))),
                ["filename", column],
            ]
            filenames = set(routed.filename)
            before = metric(base, filenames, field)
            trial = score_trial(base, routed.set_index("filename")[column], field) if filenames else base
            after = metric(trial, filenames, field)
            safe = bool(
                before["images"] >= 3
                and after["exact_match"] >= before["exact_match"]
                and after["mean_similarity"] >= before["mean_similarity"] + 0.04
            )
            gates[f"{route}:{field}"] = {
                "route": route, "field": field, "safe": safe,
                "baseline": before, "candidate": after,
            }
    return gates


def apply_gates(base: pd.DataFrame, consensus: pd.DataFrame, gates: dict[str, dict[str, Any]]) -> pd.DataFrame:
    output = base.copy()
    for field, column, prediction_column in (("name", "name_consensus", "name_pred"), ("dob", "dob_consensus", "birth_date_pred")):
        allowed = {item["route"] for item in gates.values() if item["field"] == field and item["safe"]}
        routed = consensus.loc[
            consensus.route.isin(allowed) & consensus[column].map(lambda value: bool(text(value))),
            ["filename", column],
        ]
        if routed.empty:
            continue
        values = routed.drop_duplicates("filename").set_index("filename")[column]
        mask = output.filename.isin(values.index)
        output.loc[mask, prediction_column] = output.loc[mask, "filename"].map(values).fillna("")
    return core.score_prediction_rows(output)


def family_summary(frame: pd.DataFrame, filenames: set[str]) -> dict[str, Any]:
    return {"name": metric(frame, filenames, "name"), "dob": metric(frame, filenames, "dob")}


def atomic_predictions(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"))


def run() -> None:
    base = pd.read_parquet(V17_RESULTS / "predictions_released.parquet").sort_values("filename", key=lambda values: values.map(V3.natural_key)).reset_index(drop=True)
    proposals = pd.read_csv(V17_RESULTS / "field_proposals.csv").fillna("")
    consensus = consensus_candidates(base, proposals)
    V14.atomic_csv(RESULTS / "group_consensus_candidates.csv", consensus)
    print(json.dumps({
        "stage": "group_consensus_prepared",
        "documents": int(len(consensus)),
        "name_candidates": int(consensus.name_consensus.map(lambda value: bool(text(value))).sum()),
        "dob_candidates": int(consensus.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "gpu_evidence_source": str(V17_RESULTS / "gpu_field_lines.csv"),
    }, ensure_ascii=False), flush=True)

    calibration = calibration_files(consensus)
    gates = derive_gates(base, consensus, calibration)
    candidate = apply_gates(base, consensus, gates)
    V14.atomic_json(RESULTS / "consensus_gates.json", gates)

    generic_validation = set(base.loc[
        base.document_family.eq(GENERIC) & ~base.filename.isin(calibration), "filename"
    ])
    before = family_summary(base, generic_validation)
    after = family_summary(candidate, generic_validation)
    non_regression = all(
        after[field]["exact_match"] >= before[field]["exact_match"] and
        after[field]["mean_similarity"] >= before[field]["mean_similarity"]
        for field in ("name", "dob") if before[field]["images"]
    )
    gain = (after["name"]["mean_similarity"] - before["name"]["mean_similarity"]) + (after["dob"]["mean_similarity"] - before["dob"]["mean_similarity"])
    release = bool(non_regression and gain >= 0.04)
    validation = {
        "scope": "Generic-ID leave-one-document-out visual-identity-group consensus; V17 GPU evidence reused",
        "min_peer_support": MIN_PEER_SUPPORT,
        "min_dominance": MIN_DOMINANCE,
        "calibration_documents": len(calibration),
        "validation_documents": len(generic_validation),
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
        "test_release_gate_passed": release,
        "official_submission_overwritten": False,
        "gpu_evidence_reused": True,
        "released_generic": family_summary(released, set(released.loc[released.document_family.eq(GENERIC), "filename"])),
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V18 Generic-ID leave-one-document-out consensus", "gpu_evidence_reused": True, "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
