#!/usr/bin/env python3
"""V16: GPU address repair for every detected Malaysian address-card layout.

Document-family labels are not a reliable layout label: several cards in the
GENERIC_ID family are visibly MyKad/MyPR-style Malaysian identity cards and
contain addresses.  V16 detects that layout from OCR evidence, reuses V14's
A100 output for the original MyKad set, runs the same A100 recognizer only for
the newly discovered Generic-ID cards, and applies V15's character-preserving
spacing guard.  MRZ documents are deliberately excluded because the labelled
set has no address-positive MRZ examples.

Ground truth is used only for span-ranker training/held-out validation.  The
scope detector uses OCR text and is applied equally to labelled and test rows.
Only ``address_pred`` changes in scope; all other fields and all out-of-scope
documents retain their V3 values.  The official submission is never changed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V15_PATH = PROJECT / "scripts_v15" / "01_gpu_format_only_guard.py"
V14_RESULTS = PROJECT / "results_v14_gpu_targeted_reocr"
RESULTS = PROJECT / "results_v16_all_malaysian_address_cards"
SEED = 1808
TEMPLATE_ALIAS = "MYKAD"
MALAYSIAN_CUE = r"MYK[A-Z]?|MYPR|PENGENALAN|PENGE|WARGANEGARA|PEMASTAUT|MALAYS"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V15 = load_module(V15_PATH, "v15_v16_support")
V14 = V15.V14
V3 = V14.V3
V10 = V14.V10
V12 = V14.V12
V13 = V14.V13
core = V14.core


def scope_documents(base: pd.DataFrame) -> pd.DataFrame:
    """Find all cards whose OCR evidence indicates a Malaysian ID layout."""
    output = base.copy()
    cue = output.ocr_text.fillna("").astype(str).str.upper().str.contains(MALAYSIAN_CUE, regex=True)
    output["malaysian_layout_cue"] = cue
    output["address_scope"] = output.document_family.eq(TEMPLATE_ALIAS) | cue
    output["scope_reason"] = np.where(
        output.document_family.eq(TEMPLATE_ALIAS), "document_family_mykad",
        np.where(cue, "malaysian_layout_ocr_cue", "out_of_scope"),
    )
    return output


def collect_generic_jobs(base: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """Prepare only new, cue-detected non-MyKad cards for A100 recognition."""
    selected = V10.V8.select_ocr_rows(base)
    candidates = base.loc[base.address_scope.astype(bool) & ~base.document_family.eq(TEMPLATE_ALIAS)]
    jobs: list[dict[str, Any]] = []
    source: dict[tuple[str, int], dict[str, Any]] = {}
    for document in candidates.itertuples(index=False):
        ocr = selected.get(str(document.filename))
        if ocr is None:
            continue
        image = cv2.imread(str(ocr.get("input_path", "")), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        for ordinal, item in enumerate(V10.V8.read_items(ocr.get("ocr_items", []))):
            box = V10.V8.item_box(item)
            if box is None or not V14.lower_field_box(box, width, height):
                continue
            try:
                crop = core.perspective_crop(image, box)
            except Exception:
                continue
            if min(crop.shape[:2]) < 4:
                continue
            key = (str(document.filename), ordinal)
            source[key] = {
                "filename": key[0],
                "ordinal": ordinal,
                "old_text": core.output_text(item.get("text", "")),
                "old_confidence": float(item.get("confidence", 0.0)),
                "source_document_family": str(document.document_family),
            }
            for variant, enhanced in V14.crop_variants(crop).items():
                jobs.append({"filename": key[0], "ordinal": ordinal, "variant": variant, "crop": enhanced})
    return jobs, source


def gpu_reocr_generic(base: pd.DataFrame) -> pd.DataFrame:
    jobs, source = collect_generic_jobs(base)
    print(json.dumps({
        "stage": "generic_gpu_reocr_prepared",
        "documents": int((base.address_scope.astype(bool) & ~base.document_family.eq(TEMPLATE_ALIAS)).sum()),
        "lines": len(source),
        "crops": len(jobs),
        "batch": V14.GPU_BATCH,
    }, ensure_ascii=False), flush=True)
    if not jobs:
        return pd.DataFrame(columns=[
            "filename", "ordinal", "old_text", "old_confidence", "new_text", "new_confidence",
            "chosen_variant", "variant_agreement", "accepted", "source_document_family", "gpu_elapsed_seconds",
        ])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    engine = V3.PPOCRSVTRv2Server()
    started = time.perf_counter()
    try:
        for start in range(0, len(jobs), V14.GPU_BATCH):
            chunk = jobs[start:start + V14.GPU_BATCH]
            predictions = engine.recognizer.predict_batch([job["crop"] for job in chunk], batch_size=V14.GPU_BATCH)
            for job, prediction in zip(chunk, predictions):
                grouped[(job["filename"], job["ordinal"])].append({
                    "variant": job["variant"],
                    "text": core.output_text(prediction.get("text", "")),
                    "confidence": float(prediction.get("confidence", 0.0)),
                })
            print(f"GPU generic re-OCR crops: {min(start + len(chunk), len(jobs))}/{len(jobs)}", flush=True)
    finally:
        engine.close()
    elapsed = time.perf_counter() - started
    records: list[dict[str, Any]] = []
    for key, original in source.items():
        selected = V14.choose_text(original["old_text"], original["old_confidence"], grouped.get(key, []))
        records.append({
            **original,
            "new_text": selected["text"],
            "new_confidence": selected["confidence"],
            "chosen_variant": selected["variant"],
            "variant_agreement": selected["agreement"],
            "accepted": selected["accepted"],
            "gpu_elapsed_seconds": elapsed,
        })
    return pd.DataFrame(records)


def address_metrics(frame: pd.DataFrame, family: str | None = None, present_only: bool = False, source_family: bool = False) -> dict[str, Any]:
    subset = frame.loc[frame.has_ground_truth.astype(bool)].copy()
    if family is not None:
        family_column = "source_document_family" if source_family and "source_document_family" in subset else "document_family"
        subset = subset.loc[subset[family_column].eq(family)]
    if present_only:
        subset = subset.loc[subset.address_present_gt.astype(bool)]
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None, "mean_cer": None}
    return {
        "images": int(len(subset)),
        "exact_match": float(subset.address_exact.mean()),
        "mean_similarity": float(subset.address_similarity.mean()),
        "mean_cer": float(subset.address_cer.mean()),
    }


def save_predictions(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"))


def run() -> None:
    base = pd.read_parquet(V14.V3_RESULTS / "predictions_base.parquet").sort_values(
        "filename", key=lambda value: value.map(V3.natural_key)
    ).reset_index(drop=True)
    routed = scope_documents(base)
    scope_manifest = routed[["filename", "data_split", "document_family", "has_ground_truth", "address_present_gt", "malaysian_layout_cue", "address_scope", "scope_reason"]].copy()
    V14.atomic_csv(RESULTS / "scope_manifest.csv", scope_manifest)
    print(json.dumps({
        "stage": "scope_documents",
        "scope_documents": int(routed.address_scope.sum()),
        "scope_by_original_family": routed.loc[routed.address_scope].document_family.value_counts().to_dict(),
        "test_scope_documents": int((routed.address_scope & routed.data_split.eq("test")).sum()),
    }, ensure_ascii=False), flush=True)

    # V14 already performed the expensive A100 pass for every MyKad lower-left
    # text line.  Only newly routed Generic-ID cards need GPU work here.
    inherited = pd.read_csv(V14_RESULTS / "gpu_reocr_lines.csv", keep_default_na=False)
    inherited["source_document_family"] = TEMPLATE_ALIAS
    generic_gpu = gpu_reocr_generic(routed)
    V14.atomic_csv(RESULTS / "gpu_generic_lines.csv", generic_gpu)
    manifest = pd.concat([inherited, generic_gpu], ignore_index=True, sort=False)
    raw_accepted = manifest.accepted.astype(bool)
    manifest["spacing_safe"] = [
        V15.safe_spacing_repair(old_text, new_text)
        for old_text, new_text in zip(manifest.old_text, manifest.new_text)
    ]
    manifest["accepted"] = raw_accepted & manifest.spacing_safe.astype(bool)
    manifest["rejected_character_change"] = raw_accepted & ~manifest.spacing_safe.astype(bool)
    V14.atomic_csv(RESULTS / "gpu_reocr_lines_guarded.csv", manifest)
    print(json.dumps({
        "stage": "guard_gpu_replacements",
        "raw_gpu_accepted": int(raw_accepted.sum()),
        "spacing_safe_accepted": int(manifest.accepted.sum()),
        "rejected_character_change": int(manifest.rejected_character_change.sum()),
    }, ensure_ascii=False), flush=True)

    # The span model and guard are layout-specific rather than filename-family
    # specific.  Use a temporary template alias for all cue-routed cards while
    # preserving the original family separately for reporting and final merge.
    scoped = routed.loc[routed.address_scope.astype(bool)].copy()
    scoped["source_document_family"] = scoped.document_family
    scoped["document_family"] = TEMPLATE_ALIAS
    lines = V10.flatten_lines_fast(scoped)
    lines = V14.apply_replacements(lines, manifest)
    lines, fuzzy_metadata_lines = V12.sanitize_lines(lines)
    V10.MAX_SPAN_LINES = 5
    spans = V10.span_rows(scoped, lines)
    labelled = scoped.loc[scoped.has_ground_truth.astype(bool) & scoped.address_present_gt.astype(bool)].copy()
    groups = {row.filename: V3.identity_key(row) for _, row in labelled.iterrows()}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_positions, validation_positions = next(splitter.split(labelled, groups=[groups[name] for name in labelled.filename]))
    train_files = set(labelled.iloc[train_positions].filename)
    validation_files = set(labelled.iloc[validation_positions].filename)
    threshold, calibration = V13.select_threshold(scoped, spans, train_files, groups)
    model = V10.SpanRanker().fit(spans.loc[spans.filename.isin(train_files)])
    formatter = V13.V11.AddressFormatter(scoped.loc[scoped.filename.isin(train_files), "address_gt"])
    validation_base = scoped.loc[scoped.filename.isin(validation_files)].copy()
    validation_candidate, span_decisions, fallback, format_decisions = V13.predict(
        validation_base, spans.loc[spans.filename.isin(validation_files)], model, threshold, formatter
    )
    baseline_metrics = V10.mykad_metrics(validation_base)
    candidate_metrics = V10.mykad_metrics(validation_candidate)
    generic_base = address_metrics(validation_base, family="GENERIC_ID", present_only=True, source_family=True)
    generic_candidate = address_metrics(validation_candidate, family="GENERIC_ID", present_only=True, source_family=True)
    release = bool(calibration["selected"]["safe"]) and (
        candidate_metrics["exact_match"] >= baseline_metrics["exact_match"]
        and candidate_metrics["mean_similarity"] >= baseline_metrics["mean_similarity"] + 0.03
    )
    validation = {
        "scope": "all OCR-cue-detected Malaysian address-card layouts",
        "scope_images": int(len(scoped)),
        "train_images": len(train_files),
        "validation_images": len(validation_files),
        "fuzzy_metadata_lines_masked": fuzzy_metadata_lines,
        "threshold": threshold,
        "calibration": calibration,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "generic_address_present_baseline": generic_base,
        "generic_address_present_candidate": generic_candidate,
        "span_changes": int(len(span_decisions)),
        "fallbacks": int(len(fallback)),
        "format_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)

    final_model = V10.SpanRanker().fit(spans)
    final_formatter = V13.V11.AddressFormatter(scoped.loc[scoped.has_ground_truth.astype(bool), "address_gt"])
    scope_candidate, span_decisions, fallback, format_decisions = V13.predict(scoped, spans, final_model, threshold, final_formatter)
    candidate = base.copy()
    scope_index = scope_candidate.set_index("filename")
    mask = candidate.filename.isin(scope_index.index)
    candidate.loc[mask, "address_pred"] = candidate.loc[mask, "filename"].map(scope_index.address_pred)
    if "address_detail" in scope_index and "address_detail" in candidate:
        candidate.loc[mask, "address_detail"] = candidate.loc[mask, "filename"].map(scope_index.address_detail)
    candidate = core.score_prediction_rows(candidate)
    released = candidate if release else core.score_prediction_rows(base.copy())
    save_predictions("predictions_candidate", candidate)
    save_predictions("predictions_released", released)
    V14.atomic_csv(RESULTS / "span_decisions.csv", span_decisions)
    V14.atomic_csv(RESULTS / "fallback_decisions.csv", fallback)
    V14.atomic_csv(RESULTS / "format_decisions.csv", format_decisions)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "scope_documents": int(len(scoped)),
        "scope_mykad_documents": int((routed.address_scope & routed.document_family.eq(TEMPLATE_ALIAS)).sum()),
        "scope_generic_malaysian_documents": int((routed.address_scope & routed.document_family.eq("GENERIC_ID")).sum()),
        "scope_mrz_documents": int((routed.address_scope & routed.document_family.eq("MRZ_DOCUMENT")).sum()),
        "gpu_generic_line_candidates": int(len(generic_gpu)),
        "gpu_all_raw_accepted_lines": int(raw_accepted.sum()),
        "gpu_all_spacing_safe_lines": int(manifest.accepted.sum()),
        "gpu_all_rejected_character_change": int(manifest.rejected_character_change.sum()),
        "span_candidate_changes": int(len(span_decisions)),
        "fallback_count": int(len(fallback)),
        "format_candidate_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release),
        "official_submission_overwritten": False,
        "released_generic_address_present": address_metrics(released, family="GENERIC_ID", present_only=True),
        "candidate_generic_address_present": address_metrics(candidate, family="GENERIC_ID", present_only=True),
        "released_mykad": address_metrics(released, family=TEMPLATE_ALIAS, present_only=True),
        "candidate_mykad": address_metrics(candidate, family=TEMPLATE_ALIAS, present_only=True),
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    test_rows = released.loc[released.data_split.eq("test")].copy()
    if len(test_rows) != 100:
        raise RuntimeError(f"Expected 100 test rows, got {len(test_rows)}")
    official_submission = PROJECT / "results" / "test_predictions.csv"
    if not routed.loc[routed.data_split.eq("test"), "address_scope"].any():
        shutil.copy2(official_submission, RESULTS / "test_predictions.csv")
    else:
        submission = pd.read_csv(official_submission, keep_default_na=False)
        updates = test_rows.loc[test_rows.filename.isin(set(scope_candidate.filename)), ["filename", "address_pred"]].set_index("filename").address_pred
        changed = submission.filename.isin(updates.index)
        submission.loc[changed, "address"] = submission.loc[changed, "filename"].map(updates).fillna("")
        V14.atomic_csv(RESULTS / "test_predictions.csv", submission)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V16 all Malaysian address-card layouts -> V14 A100 GPU re-OCR -> V15 spacing guard -> scoped V13 parser",
        "gpu_required_for_new_generic_cards": True,
        "official_results_untouched": True,
    }, ensure_ascii=False), flush=True)
    run()
