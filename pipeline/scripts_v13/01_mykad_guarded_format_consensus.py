#!/usr/bin/env python3
"""V13: V12 visual guard plus train-only address boundary restoration.

This is the strictly validated composition of V12 (metadata/ID masking,
five-line spans, and baseline fallback) and V11's conservative whitespace
restoration.  The vocabulary seen by the formatter is learned only from each
training fold; selected fallback rows are deliberately left untouched.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
V10_PATH = PROJECT / "scripts_v10" / "01_supervised_mykad_address_span_ranker.py"
V11_PATH = PROJECT / "scripts_v11" / "01_mykad_address_format_consensus.py"
V12_PATH = PROJECT / "scripts_v12" / "01_mykad_visual_error_guard.py"
V3_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v13_mykad_guarded_format"
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


V3 = load_module(V3_PATH, "v3_v13_support")
V10 = load_module(V10_PATH, "v10_v13_support")
V11 = load_module(V11_PATH, "v11_v13_support")
V12 = load_module(V12_PATH, "v12_v13_support")
core = V3.core


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=core.json_default), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def format_after_guard(frame: pd.DataFrame, formatter: Any, fallback: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restore boundaries only in new span output, never fallback baseline."""
    output = frame.copy()
    excluded = set(fallback.filename.astype(str)) if not fallback.empty else set()
    selected = output.document_family.eq(FAMILY) & output.get("span_address", pd.Series(index=output.index, dtype=object)).notna() & ~output.filename.astype(str).isin(excluded)
    original = output.loc[selected, "address_pred"].copy()
    output.loc[selected, "address_pred"] = original.map(formatter.format)
    changed = selected & output.address_pred.map(V10.V8.clean).ne(frame.address_pred.map(V10.V8.clean))
    decisions = output.loc[changed, ["filename", "data_split", "span_address", "address_pred", "span_probability", "span_line_ids"]].rename(columns={"address_pred": "formatted_address", "span_address": "unformatted_span"})
    return core.score_prediction_rows(output), decisions


def inner_split(train_files: set[str], groups: dict[str, str]) -> tuple[set[str], set[str]]:
    files = sorted(train_files)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    fit_positions, calibration_positions = next(splitter.split(files, groups=[groups[name] for name in files]))
    return {files[position] for position in fit_positions}, {files[position] for position in calibration_positions}


def predict(base: pd.DataFrame, spans: pd.DataFrame, model: Any, threshold: float, formatter: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    guarded, span_decisions, fallback = V12.predict(base, spans, model, threshold)
    formatted, format_decisions = format_after_guard(guarded, formatter, fallback)
    return formatted, span_decisions, fallback, format_decisions


def select_threshold(base: pd.DataFrame, spans: pd.DataFrame, train_files: set[str], groups: dict[str, str]) -> tuple[float, dict[str, Any]]:
    fit_files, calibration_files = inner_split(train_files, groups)
    model = V10.SpanRanker().fit(spans.loc[spans.filename.isin(fit_files)])
    formatter = V11.AddressFormatter(base.loc[base.filename.isin(fit_files), "address_gt"])
    calibration_base = base.loc[base.filename.isin(calibration_files)].copy()
    baseline_metrics = V10.mykad_metrics(calibration_base)
    options: list[dict[str, Any]] = []
    for threshold in V10.THRESHOLDS:
        candidate, span_decisions, fallback, format_decisions = predict(calibration_base, spans.loc[spans.filename.isin(calibration_files)], model, float(threshold), formatter)
        metrics = V10.mykad_metrics(candidate)
        safe = metrics["exact_match"] >= baseline_metrics["exact_match"] and metrics["mean_similarity"] >= baseline_metrics["mean_similarity"] + 0.03
        options.append({"threshold": float(threshold), "safe": bool(safe), "span_changes": int(len(span_decisions)), "fallbacks": int(len(fallback)), "format_changes": int(len(format_decisions)), **metrics})
    safe_options = [option for option in options if option["safe"]]
    selected = max(safe_options, key=lambda option: (option["exact_match"], option["mean_similarity"], -option["format_changes"])) if safe_options else max(options, key=lambda option: (option["exact_match"], option["mean_similarity"], -option["format_changes"]))
    return float(selected["threshold"]), {"fit_images": len(fit_files), "calibration_images": len(calibration_files), "baseline": baseline_metrics, "selected": selected, "options": options}


def run() -> None:
    base = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda value: value.map(V3.natural_key)).reset_index(drop=True)
    mykad = base.loc[base.document_family.eq(FAMILY)].copy()
    lines = V10.flatten_lines_fast(mykad)
    lines, fuzzy_metadata_lines = V12.sanitize_lines(lines)
    V10.MAX_SPAN_LINES = 5
    spans = V10.span_rows(base, lines)
    labelled = mykad.loc[mykad.has_ground_truth.astype(bool) & mykad.address_present_gt.astype(bool)].copy()
    groups = {row.filename: V3.identity_key(row) for _, row in labelled.iterrows()}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_positions, validation_positions = next(splitter.split(labelled, groups=[groups[name] for name in labelled.filename]))
    train_files = set(labelled.iloc[train_positions].filename)
    validation_files = set(labelled.iloc[validation_positions].filename)
    threshold, calibration = select_threshold(base, spans, train_files, groups)
    model = V10.SpanRanker().fit(spans.loc[spans.filename.isin(train_files)])
    formatter = V11.AddressFormatter(base.loc[base.filename.isin(train_files), "address_gt"])
    validation_base = base.loc[base.filename.isin(validation_files)].copy()
    validation_candidate, span_decisions, fallback, format_decisions = predict(validation_base, spans.loc[spans.filename.isin(validation_files)], model, threshold, formatter)
    baseline_metrics = V10.mykad_metrics(validation_base)
    candidate_metrics = V10.mykad_metrics(validation_candidate)
    release = bool(calibration["selected"]["safe"]) and candidate_metrics["exact_match"] >= baseline_metrics["exact_match"] and candidate_metrics["mean_similarity"] >= baseline_metrics["mean_similarity"] + 0.03
    validation = {
        "family": FAMILY, "train_images": len(train_files), "validation_images": len(validation_files),
        "fuzzy_metadata_lines_masked": fuzzy_metadata_lines, "max_span_lines": V10.MAX_SPAN_LINES,
        "threshold": threshold, "calibration": calibration, "baseline": baseline_metrics,
        "candidate": candidate_metrics, "span_changes": int(len(span_decisions)), "fallbacks": int(len(fallback)), "format_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    final_model = V10.SpanRanker().fit(spans)
    final_formatter = V11.AddressFormatter(base.loc[base.has_ground_truth.astype(bool) & base.document_family.eq(FAMILY), "address_gt"])
    candidate, span_decisions, fallback, format_decisions = predict(base, spans, final_model, threshold, final_formatter)
    released = candidate if release else core.score_prediction_rows(base.copy())
    atomic_parquet(RESULTS / "predictions_candidate.parquet", candidate)
    atomic_csv(RESULTS / "predictions_candidate.csv", candidate.drop(columns=[column for column in candidate if column.endswith("_detail")], errors="ignore"))
    atomic_parquet(RESULTS / "predictions_released.parquet", released)
    atomic_csv(RESULTS / "predictions_released.csv", released.drop(columns=[column for column in released if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "span_decisions.csv", span_decisions)
    atomic_csv(RESULTS / "fallback_decisions.csv", fallback)
    atomic_csv(RESULTS / "format_decisions.csv", format_decisions)
    atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "span_candidate_changes": int(len(span_decisions)), "fallback_count": int(len(fallback)), "format_candidate_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release), "official_submission_overwritten": False,
        **{f"released_mykad_{key}": value for key, value in V10.mykad_metrics(released).items()},
        **{f"candidate_mykad_{key}": value for key, value in V10.mykad_metrics(candidate).items()},
    })
    atomic_json(RESULTS / "metrics.json", metrics)
    submission = released.loc[released.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, got {len(submission)}")
    atomic_csv(RESULTS / "test_predictions.csv", submission)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V13 V12 visual guard + train-only MyKad spacing formatter", "ocr_rerun": False, "gpu_required": False, "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
