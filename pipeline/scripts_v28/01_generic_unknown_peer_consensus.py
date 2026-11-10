#!/usr/bin/env python3
"""V28: route-unknown Generic-ID leave-one-out peer consensus.

Only documents still un-routed by V17 are considered.  A target never uses its
own prediction: name uses a fuzzy peer medoid with >=2 peers; DOB uses a
normalized-date peer medoid with >=3 peers.  Both changes require calibration
improvement and holdout non-regression before release.  The official test CSV
is copied untouched.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz.fuzz import ratio

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import identity_ocr_compfest18 as core

BASE = PROJECT / "reference_outputs" / "results_v27_fuzzy_peer_medoid_names"
PROPOSALS = PROJECT / "results_v17_per_layout_generic_mrz" / "field_proposals.csv"
OFFICIAL_TEST = PROJECT / "results" / "test_predictions.csv"
RESULTS = PROJECT / "results_v28_generic_unknown_peer_consensus"
NAME_MIN_PEERS = 2
DOB_MIN_PEERS = 3


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


def name_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", core.output_text(value).upper()).strip()


def name_medoid(values: list[str]) -> str:
    values = [core.output_text(value) for value in values if name_key(value)]
    if not values:
        return ""
    return max(values, key=lambda value: sum(ratio(name_key(value), name_key(peer)) for peer in values) / len(values))


def normalized_date(value: Any) -> str:
    return core.normalize_date(value)


def dob_medoid(values: list[str]) -> str:
    values = [normalized_date(value) for value in values if normalized_date(value)]
    if not values:
        return ""
    return max(values, key=lambda value: sum(ratio(value, peer) for peer in values) / len(values))


def peer_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group, chunk in frame.groupby("identity_group", sort=True):
        rows = list(chunk.itertuples(index=False))
        for target in rows:
            name_peers = [peer.name_pred for peer in rows if peer.filename != target.filename and name_key(peer.name_pred)]
            dob_peers = [normalized_date(peer.birth_date_pred) for peer in rows if peer.filename != target.filename and normalized_date(peer.birth_date_pred)]
            records.append({
                "filename": target.filename,
                "identity_group": group,
                "group_documents": len(rows),
                "name_peer_medoid": name_medoid(name_peers),
                "name_peer_support": len(name_peers),
                "dob_peer_medoid": dob_medoid(dob_peers),
                "dob_peer_support": len(dob_peers),
            })
    return pd.DataFrame(records)


def apply(base: pd.DataFrame, candidates: pd.DataFrame, use_name: bool, use_dob: bool) -> pd.DataFrame:
    # identity_group already belongs to the base frame; avoid a merge suffix.
    consensus = candidates.drop(columns=["identity_group"], errors="ignore")
    output = base.copy().merge(consensus, on="filename", how="left")
    if use_name:
        mask = output.name_peer_support.fillna(0).ge(NAME_MIN_PEERS) & output.name_peer_medoid.fillna("").map(lambda value: bool(name_key(value)))
        output.loc[mask, "name_pred"] = output.loc[mask, "name_peer_medoid"]
    if use_dob:
        mask = output.dob_peer_support.fillna(0).ge(DOB_MIN_PEERS) & output.dob_peer_medoid.fillna("").ne("")
        output.loc[mask, "birth_date_pred"] = output.loc[mask, "dob_peer_medoid"]
    output = output.drop(columns=[column for column in consensus.columns if column != "filename"], errors="ignore")
    return core.score_prediction_rows(output)


def field_summary(frame: pd.DataFrame, filenames: set[str], prefix: str) -> dict[str, Any]:
    subset = frame.loc[frame.filename.isin(filenames) & frame.has_ground_truth.astype(bool)]
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None}
    return {
        "images": int(len(subset)),
        "exact_match": float(subset[f"{prefix}_exact"].mean()),
        "mean_similarity": float(subset[f"{prefix}_similarity"].mean()),
    }


def field_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame.loc[frame.has_ground_truth.astype(bool)]
    latency = valid.ocr_latency_ms.dropna() if "ocr_latency_ms" in valid else pd.Series(dtype=float)
    return {
        "Name EM": float(valid.name_exact.mean()),
        "DOB EM": float(valid.dob_exact.mean()),
        "Address EM": float(valid.address_exact.mean()),
        "Macro Field EM": float(valid.macro_field_exact.mean()),
        "Full Record Accuracy": float(valid.full_record_correct.mean()),
        "CER": float(valid.record_cer.mean()),
        "Normalized Edit Similarity": float(valid.normalized_edit_similarity.mean()),
        "P50": float(latency.quantile(0.50)) if len(latency) else float("nan"),
        "P95": float(latency.quantile(0.95)) if len(latency) else float("nan"),
        "Labeled N": int(len(valid)),
    }


def save(name: str, frame: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RESULTS / f"{name}.parquet", index=False)
    frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore").to_csv(RESULTS / f"{name}.csv", index=False, encoding="utf-8")


def accepted(before: dict[str, Any], calibration: dict[str, Any], validation_before: dict[str, Any], validation_after: dict[str, Any]) -> bool:
    return bool(
        before["images"] >= 20
        and calibration["exact_match"] >= before["exact_match"]
        and calibration["mean_similarity"] >= before["mean_similarity"] + 0.04
        and validation_after["exact_match"] >= validation_before["exact_match"]
        and validation_after["mean_similarity"] >= validation_before["mean_similarity"]
    )


def run() -> None:
    base = pd.read_parquet(BASE / "predictions_released.parquet").sort_values("filename", key=lambda values: values.map(natural_key)).reset_index(drop=True)
    proposals = pd.read_csv(PROPOSALS).fillna("")
    routed = base.merge(proposals[["filename", "route"]], on="filename", how="left").fillna({"route": ""})
    unknown = routed.loc[routed.document_family.eq("GENERIC_ID") & routed.route.eq("")].copy()
    candidates = peer_candidates(unknown)
    candidates.to_csv(RESULTS / "generic_unknown_peer_candidates.csv", index=False)
    labelled = unknown.loc[unknown.has_ground_truth.astype(bool)].copy()
    calibration_files = set(sorted(labelled.filename, key=natural_key)[::3])
    validation_files = set(labelled.filename) - calibration_files

    name_trial = apply(base, candidates, use_name=True, use_dob=False)
    dob_trial = apply(base, candidates, use_name=False, use_dob=True)
    name_before_calibration = field_summary(base, calibration_files, "name")
    name_after_calibration = field_summary(name_trial, calibration_files, "name")
    name_before_validation = field_summary(base, validation_files, "name")
    name_after_validation = field_summary(name_trial, validation_files, "name")
    dob_before_calibration = field_summary(base, calibration_files, "dob")
    dob_after_calibration = field_summary(dob_trial, calibration_files, "dob")
    dob_before_validation = field_summary(base, validation_files, "dob")
    dob_after_validation = field_summary(dob_trial, validation_files, "dob")
    name_safe = accepted(name_before_calibration, name_after_calibration, name_before_validation, name_after_validation)
    dob_safe = accepted(dob_before_calibration, dob_after_calibration, dob_before_validation, dob_after_validation)

    candidate = apply(base, candidates, use_name=True, use_dob=True)
    released = candidate if name_safe and dob_safe else base.copy()
    validation = {
        "scope": "Unknown Generic-ID only; leave-one-out visual-identity peer medoids",
        "labelled_unknown_documents": len(labelled), "unknown_documents_all_splits": len(unknown),
        "calibration_documents": len(calibration_files), "validation_documents": len(validation_files),
        "name_gate": {"min_peers": NAME_MIN_PEERS, "safe": name_safe, "calibration_baseline": name_before_calibration, "calibration_candidate": name_after_calibration, "validation_baseline": name_before_validation, "validation_candidate": name_after_validation},
        "dob_gate": {"min_peers": DOB_MIN_PEERS, "safe": dob_safe, "calibration_baseline": dob_before_calibration, "calibration_candidate": dob_after_calibration, "validation_baseline": dob_before_validation, "validation_candidate": dob_after_validation},
        "test_release_gate_passed": bool(name_safe and dob_safe),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    save("predictions_candidate", candidate)
    save("predictions_released", released)
    (RESULTS / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=core.json_default), encoding="utf-8")
    metrics = field_metrics(released)
    metrics.update({
        "generic_unknown_name_peer_gate_safe": name_safe,
        "generic_unknown_dob_peer_gate_safe": dob_safe,
        "generic_unknown_name_changes": int((candidates.name_peer_support.ge(NAME_MIN_PEERS) & candidates.name_peer_medoid.map(lambda value: bool(name_key(value)))).sum()),
        "generic_unknown_dob_changes": int((candidates.dob_peer_support.ge(DOB_MIN_PEERS) & candidates.dob_peer_medoid.ne("")).sum()),
        "gpu_evidence_reused": True,
        "test_release_gate_passed": bool(name_safe and dob_safe),
        "official_submission_overwritten": False,
    })
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=core.json_default), encoding="utf-8")
    shutil.copy2(OFFICIAL_TEST, RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    run()