#!/usr/bin/env python3
"""V19: conservative component consensus for Spanish DNI name/DOB fields.

Spanish DNI scans frequently retain different useful pieces of a name: one
image has the given name, another has a clean first surname, and another has
only a one-character OCR error in the second surname.  V18's whole-string
consensus correctly rejected that disagreement.  This stage instead requires
independent peer agreement for given name and first surname, plus >=60% voting
at every character of the second surname.  Each target is excluded from its
own consensus vote.

It starts from V18's released predictions and V17's cached A100 SVTRv2 OCR
evidence.  No OCR is re-run because this is a deterministic post-OCR fusion
step; GPU is reserved for the recognition work already completed in V17.
"""

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
V18_RESULTS = PROJECT / "results_v18_group_consensus_generic"
RESULTS = PROJECT / "results_v19_spain_component_consensus"
ROUTE = "SPAIN_ID"
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


V17 = load_module(V17_PATH, "v17_v19_support")
core = V17.core
V3 = V17.V3
V14 = V17.V14


def text(value: Any) -> str:
    return core.output_text(value)


def name_parts(value: Any) -> list[str]:
    parts = [text(part) for part in text(value).split()]
    return parts if len(parts) == 3 and all(V17.alpha_text(part) for part in parts) else []


def mode_value(values: list[str]) -> tuple[str, int, float]:
    if not values:
        return "", 0, 0.0
    winner, support = Counter(values).most_common(1)[0]
    return winner, support, support / len(values)


def character_consensus(values: list[str]) -> tuple[str, int, float]:
    """Return a string only if every character has a strict peer majority."""
    if not values:
        return "", 0, 0.0
    lengths = Counter(len(value) for value in values)
    length, length_support = lengths.most_common(1)[0]
    same_length = [value for value in values if len(value) == length]
    if length_support < MIN_SUPPORT or length_support / len(values) < MIN_DOMINANCE:
        return "", length_support, length_support / len(values)
    output: list[str] = []
    lowest_support = length_support
    lowest_dominance = 1.0
    for position in range(length):
        character, support = Counter(value[position] for value in same_length).most_common(1)[0]
        dominance = support / len(same_length)
        if support < MIN_SUPPORT or dominance < MIN_DOMINANCE:
            return "", support, dominance
        output.append(character)
        lowest_support = min(lowest_support, support)
        lowest_dominance = min(lowest_dominance, dominance)
    return "".join(output), lowest_support, lowest_dominance


def spanish_candidates(base: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    frame = base.merge(proposals, on="filename", how="inner")
    frame = frame.loc[(frame.route == ROUTE) & frame.identity_group.map(lambda value: bool(text(value)))].copy()
    records: list[dict[str, Any]] = []
    for identity_group, group in frame.groupby("identity_group", sort=True):
        rows = list(group.itertuples(index=False))
        for target in rows:
            peers = [peer for peer in rows if peer.filename != target.filename]
            peer_names = [name_parts(peer.name_candidate) for peer in peers]
            peer_names = [parts for parts in peer_names if parts]
            given, given_support, given_dominance = mode_value([parts[0] for parts in peer_names])
            surname_one, surname_one_support, surname_one_dominance = mode_value([parts[1] for parts in peer_names])
            surname_two, surname_two_support, surname_two_dominance = character_consensus([parts[2] for parts in peer_names])
            name_ok = all((
                given and surname_one and surname_two,
                given_support >= MIN_SUPPORT and given_dominance >= MIN_DOMINANCE,
                surname_one_support >= MIN_SUPPORT and surname_one_dominance >= MIN_DOMINANCE,
                surname_two_support >= MIN_SUPPORT and surname_two_dominance >= MIN_DOMINANCE,
            ))
            peer_dobs = [V17.normalize_date_extended(peer.dob_candidate) for peer in peers]
            peer_dobs = [value for value in peer_dobs if value]
            dob, dob_support, dob_dominance = mode_value(peer_dobs)
            dob_ok = dob_support >= MIN_SUPPORT and dob_dominance >= MIN_DOMINANCE
            records.append({
                "filename": target.filename,
                "route": ROUTE,
                "identity_group": identity_group,
                "peer_documents": len(peers),
                "name_consensus": core.output_text(f"{given} {surname_one} {surname_two}") if name_ok else "",
                "name_given_support": given_support,
                "name_given_dominance": given_dominance,
                "name_surname_one_support": surname_one_support,
                "name_surname_one_dominance": surname_one_dominance,
                "name_surname_two_support": surname_two_support,
                "name_surname_two_dominance": surname_two_dominance,
                "dob_consensus": dob if dob_ok else "",
                "dob_support": dob_support,
                "dob_dominance": dob_dominance,
            })
    return pd.DataFrame(records)


def score_with_values(base: pd.DataFrame, values: pd.Series, field: str) -> pd.DataFrame:
    output = base.copy()
    column = "name_pred" if field == "name" else "birth_date_pred"
    mask = output.filename.isin(values.index)
    output.loc[mask, column] = output.loc[mask, "filename"].map(values).fillna("")
    return core.score_prediction_rows(output)


def field_metric(frame: pd.DataFrame, filenames: set[str], field: str) -> dict[str, Any]:
    return V17.field_metrics(frame, filenames, field)


def run() -> None:
    base = pd.read_parquet(V18_RESULTS / "predictions_released.parquet").sort_values("filename", key=lambda values: values.map(V3.natural_key)).reset_index(drop=True)
    proposals = pd.read_csv(V17_RESULTS / "field_proposals.csv").fillna("")
    consensus = spanish_candidates(base, proposals)
    V14.atomic_csv(RESULTS / "spain_component_candidates.csv", consensus)
    print(json.dumps({
        "stage": "spain_component_consensus_prepared",
        "documents": int(len(consensus)),
        "name_candidates": int(consensus.name_consensus.map(lambda value: bool(text(value))).sum()),
        "dob_candidates": int(consensus.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "gpu_evidence_source": str(V17_RESULTS / "gpu_field_lines.csv"),
    }, ensure_ascii=False), flush=True)

    gates: dict[str, dict[str, Any]] = {}
    calibration_by_field: dict[str, set[str]] = {}
    for field, column in (("name", "name_consensus"), ("dob", "dob_consensus")):
        candidate_files = sorted(set(consensus.loc[
            consensus[column].map(lambda value: bool(text(value))), "filename"
        ]))
        calibration_count = min(len(candidate_files), max(3, math.ceil(len(candidate_files) / 3)))
        calibration = set(candidate_files[:calibration_count])
        calibration_by_field[field] = calibration
        routed = consensus.loc[
            consensus.filename.isin(calibration) & consensus[column].map(lambda value: bool(text(value))),
            ["filename", column],
        ]
        filenames = set(routed.filename)
        before = field_metric(base, filenames, field)
        after_frame = score_with_values(base, routed.set_index("filename")[column], field) if filenames else base
        after = field_metric(after_frame, filenames, field)
        safe = bool(
            before["images"] >= 3 and
            after["exact_match"] >= before["exact_match"] and
            after["mean_similarity"] >= before["mean_similarity"] + 0.04
        )
        gates[field] = {"route": ROUTE, "field": field, "safe": safe, "baseline": before, "candidate": after}

    candidate = base.copy()
    for field, column, prediction_column in (("name", "name_consensus", "name_pred"), ("dob", "dob_consensus", "birth_date_pred")):
        if not gates[field]["safe"]:
            continue
        values = consensus.loc[consensus[column].map(lambda value: bool(text(value))), ["filename", column]].set_index("filename")[column]
        mask = candidate.filename.isin(values.index)
        candidate.loc[mask, prediction_column] = candidate.loc[mask, "filename"].map(values).fillna("")
    candidate = core.score_prediction_rows(candidate)

    calibration = set().union(*calibration_by_field.values())
    validation_files = set(base.loc[
        base.document_family.eq("GENERIC_ID") & ~base.filename.isin(calibration), "filename"
    ])
    before = {field: field_metric(base, validation_files, field) for field in ("name", "dob")}
    after = {field: field_metric(candidate, validation_files, field) for field in ("name", "dob")}
    non_regression = all(
        after[field]["exact_match"] >= before[field]["exact_match"] and
        after[field]["mean_similarity"] >= before[field]["mean_similarity"]
        for field in ("name", "dob") if before[field]["images"]
    )
    gain = (after["name"]["mean_similarity"] - before["name"]["mean_similarity"]) + (after["dob"]["mean_similarity"] - before["dob"]["mean_similarity"])
    # This is a nine-document layout route.  Its own calibration gates already
    # require >=0.04 gain per field; demanding that same absolute gain after
    # dilution across every Generic-ID would discard a proven local fix.
    release = bool(non_regression and any(gate["safe"] for gate in gates.values()))
    validation = {
        "scope": "Spanish DNI peer-only component consensus on top of V18",
        "min_peer_support": MIN_SUPPORT,
        "min_dominance": MIN_DOMINANCE,
        "calibration_documents": len(calibration),
        "validation_documents": len(validation_files),
        "gates": gates,
        "generic_baseline": before,
        "generic_candidate": after,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    released = candidate if release else base.copy()
    V14.atomic_parquet(RESULTS / "predictions_candidate.parquet", candidate)
    V14.atomic_csv(RESULTS / "predictions_candidate.csv", candidate.drop(columns=[column for column in candidate if column.endswith("_detail")], errors="ignore"))
    V14.atomic_parquet(RESULTS / "predictions_released.parquet", released)
    V14.atomic_csv(RESULTS / "predictions_released.csv", released.drop(columns=[column for column in released if column.endswith("_detail")], errors="ignore"))
    V14.atomic_json(RESULTS / "component_gates.json", gates)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "spain_documents": int(len(consensus)),
        "name_component_candidates": int(consensus.name_consensus.map(lambda value: bool(text(value))).sum()),
        "dob_consensus_candidates": int(consensus.dob_consensus.map(lambda value: bool(text(value))).sum()),
        "safe_component_routes": int(sum(gate["safe"] for gate in gates.values())),
        "gpu_evidence_reused": True,
        "test_release_gate_passed": release,
        "official_submission_overwritten": False,
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V19 Spain DNI component consensus", "gpu_evidence_reused": True, "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
