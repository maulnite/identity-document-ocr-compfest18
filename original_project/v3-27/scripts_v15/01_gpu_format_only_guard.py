#!/usr/bin/env python3
"""V15: retain only structurally safe spacing repairs from V14 GPU re-OCR.

V14 used the A100 SVTRv2 Server recognizer on three enhanced variants for
lower-left MyKad text boxes.  Visual/GT inspection found a useful asymmetry:
the large recognizer reliably restored missing word boundaries, but very
occasionally changed an already-correct letter or postcode digit.  This stage
reuses V14's GPU manifest and accepts a replacement only if its alphanumeric
sequence is identical to the original and it preserves every pre-existing
separator.  It therefore cannot turn a correct postcode or word into a new
spelling.  The V13 nested span, metadata, format, and held-out release gates
are retained unchanged.
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
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V14_PATH = PROJECT / "scripts_v14" / "01_gpu_targeted_address_reocr.py"
V14_RESULTS = PROJECT / "results_v14_gpu_targeted_reocr"
RESULTS = PROJECT / "results_v15_gpu_format_only_guard"
SEED = 1808
FAMILY = "MYKAD"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V14 = load_module(V14_PATH, "v14_v15_support")
V3 = V14.V3
V10 = V14.V10
V12 = V14.V12
V13 = V14.V13
core = V14.core


def safe_text(value: Any) -> str:
    return "" if pd.isna(value) else core.output_text(value)


def alnum_skeleton(value: Any) -> str:
    return "".join(character for character in safe_text(value).upper() if character.isalnum())


def separator_positions(value: Any) -> set[int]:
    """Positions in the alphanumeric skeleton where a separator already exists."""
    text = safe_text(value).upper()
    positions: set[int] = set()
    seen = 0
    previous_alnum = False
    pending_separator = False
    for character in text:
        if character.isalnum():
            if previous_alnum and pending_separator:
                positions.add(seen)
            seen += 1
            previous_alnum = True
            pending_separator = False
        else:
            pending_separator = pending_separator or previous_alnum
    return positions


def safe_spacing_repair(old_text: Any, new_text: Any) -> bool:
    """Allow only an additive, character-preserving segmentation repair."""
    old_skeleton = alnum_skeleton(old_text)
    new_skeleton = alnum_skeleton(new_text)
    if not old_skeleton or old_skeleton != new_skeleton:
        return False
    old_separators = separator_positions(old_text)
    new_separators = separator_positions(new_text)
    return old_separators.issubset(new_separators) and new_separators != old_separators


def run() -> None:
    manifest = pd.read_csv(V14_RESULTS / "gpu_reocr_lines.csv", keep_default_na=False)
    original_accepted = manifest.accepted.astype(bool)
    manifest["spacing_safe"] = [
        safe_spacing_repair(old_text, new_text)
        for old_text, new_text in zip(manifest.old_text, manifest.new_text)
    ]
    manifest["accepted"] = original_accepted & manifest.spacing_safe.astype(bool)
    manifest["rejected_character_change"] = original_accepted & ~manifest.spacing_safe.astype(bool)
    V14.atomic_csv(RESULTS / "gpu_reocr_lines_guarded.csv", manifest)
    print(json.dumps({
        "stage": "reuse_v14_gpu_manifest",
        "v14_accepted": int(original_accepted.sum()),
        "v15_spacing_safe": int(manifest.accepted.sum()),
        "v15_rejected_character_change": int(manifest.rejected_character_change.sum()),
    }, ensure_ascii=False), flush=True)

    base = pd.read_parquet(V14.V3_RESULTS / "predictions_base.parquet").sort_values(
        "filename", key=lambda value: value.map(V3.natural_key)
    ).reset_index(drop=True)
    mykad = base.loc[base.document_family.eq(FAMILY)].copy()
    lines = V10.flatten_lines_fast(mykad)
    lines = V14.apply_replacements(lines, manifest)
    lines, fuzzy_metadata_lines = V12.sanitize_lines(lines)
    V10.MAX_SPAN_LINES = 5
    spans = V10.span_rows(base, lines)

    labelled = mykad.loc[mykad.has_ground_truth.astype(bool) & mykad.address_present_gt.astype(bool)].copy()
    groups = {row.filename: V3.identity_key(row) for _, row in labelled.iterrows()}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_positions, validation_positions = next(splitter.split(labelled, groups=[groups[name] for name in labelled.filename]))
    train_files = set(labelled.iloc[train_positions].filename)
    validation_files = set(labelled.iloc[validation_positions].filename)
    threshold, calibration = V13.select_threshold(base, spans, train_files, groups)
    model = V10.SpanRanker().fit(spans.loc[spans.filename.isin(train_files)])
    formatter = V13.V11.AddressFormatter(base.loc[base.filename.isin(train_files), "address_gt"])
    validation_base = base.loc[base.filename.isin(validation_files)].copy()
    validation_candidate, span_decisions, fallback, format_decisions = V13.predict(
        validation_base, spans.loc[spans.filename.isin(validation_files)], model, threshold, formatter
    )
    baseline_metrics = V10.mykad_metrics(validation_base)
    candidate_metrics = V10.mykad_metrics(validation_candidate)
    release = bool(calibration["selected"]["safe"]) and (
        candidate_metrics["exact_match"] >= baseline_metrics["exact_match"]
        and candidate_metrics["mean_similarity"] >= baseline_metrics["mean_similarity"] + 0.03
    )
    validation = {
        "gpu_source": "V14 A100 SVTRv2 Server enhanced-crop manifest",
        "v14_accepted_lines": int(original_accepted.sum()),
        "v15_spacing_safe_lines": int(manifest.accepted.sum()),
        "v15_rejected_character_change": int(manifest.rejected_character_change.sum()),
        "fuzzy_metadata_lines_masked": fuzzy_metadata_lines,
        "threshold": threshold,
        "calibration": calibration,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "span_changes": int(len(span_decisions)),
        "fallbacks": int(len(fallback)),
        "format_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)

    final_model = V10.SpanRanker().fit(spans)
    final_formatter = V13.V11.AddressFormatter(base.loc[
        base.has_ground_truth.astype(bool) & base.document_family.eq(FAMILY), "address_gt"
    ])
    candidate, span_decisions, fallback, format_decisions = V13.predict(base, spans, final_model, threshold, final_formatter)
    # V13's generic predictor can re-parse fields of every document family.
    # This experiment is explicitly address-only and MyKad-only, so retain the
    # original V3 output everywhere else (including name/DOB of MyKad rows).
    # That makes the test submission byte-for-byte independent of an unrelated
    # Generic/MRZ parser route.
    scoped_candidate = base.copy()
    candidate_index = candidate.set_index("filename")
    scoped_mask = scoped_candidate.document_family.eq(FAMILY)
    scoped_filenames = scoped_candidate.loc[scoped_mask, "filename"]
    scoped_candidate.loc[scoped_mask, "address_pred"] = scoped_filenames.map(candidate_index.address_pred)
    if "address_detail" in candidate_index and "address_detail" in scoped_candidate:
        scoped_candidate.loc[scoped_mask, "address_detail"] = scoped_filenames.map(candidate_index.address_detail)
    scoped_candidate = core.score_prediction_rows(scoped_candidate)
    released = scoped_candidate if release else core.score_prediction_rows(base.copy())
    for name, frame in (("predictions_candidate", scoped_candidate), ("predictions_released", released)):
        V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
        V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"))
    V14.atomic_csv(RESULTS / "span_decisions.csv", span_decisions)
    V14.atomic_csv(RESULTS / "fallback_decisions.csv", fallback)
    V14.atomic_csv(RESULTS / "format_decisions.csv", format_decisions)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "gpu_reocr_source": "V14 A100 SVTRv2 Server enhanced-crop manifest",
        "gpu_reocr_line_candidates": int(len(manifest)),
        "gpu_reocr_v14_accepted_lines": int(original_accepted.sum()),
        "gpu_reocr_v15_spacing_safe_lines": int(manifest.accepted.sum()),
        "gpu_reocr_v15_rejected_character_change": int(manifest.rejected_character_change.sum()),
        "span_candidate_changes": int(len(span_decisions)),
        "fallback_count": int(len(fallback)),
        "format_candidate_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release),
        "official_submission_overwritten": False,
        **{f"released_mykad_{key}": value for key, value in V10.mykad_metrics(released).items()},
        **{f"candidate_mykad_{key}": value for key, value in V10.mykad_metrics(scoped_candidate).items()},
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    test_rows = released.loc[released.data_split.eq("test")].copy()
    if len(test_rows) != 100:
        raise RuntimeError(f"Expected 100 test rows, got {len(test_rows)}")
    official_submission = PROJECT / "results" / "test_predictions.csv"
    if not test_rows.document_family.eq(FAMILY).any():
        # The supplied test set has no MyKad rows.  Its contents are outside
        # this address-only experiment, so preserve the established submission
        # exactly rather than serializing V3's unrelated parser output.
        shutil.copy2(official_submission, RESULTS / "test_predictions.csv")
    else:
        submission = pd.read_csv(official_submission, keep_default_na=False)
        address_by_filename = test_rows.set_index("filename").address_pred
        mykad_filenames = set(test_rows.loc[test_rows.document_family.eq(FAMILY), "filename"])
        scoped = submission.filename.isin(mykad_filenames)
        submission.loc[scoped, "address"] = submission.loc[scoped, "filename"].map(address_by_filename).fillna("")
        V14.atomic_csv(RESULTS / "test_predictions.csv", submission)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V15 reuse V14 A100 GPU re-OCR -> character-preserving spacing guard -> V13 validation",
        "official_results_untouched": True,
    }, ensure_ascii=False), flush=True)
    run()
