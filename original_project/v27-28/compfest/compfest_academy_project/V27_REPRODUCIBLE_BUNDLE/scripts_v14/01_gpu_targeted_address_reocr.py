#!/usr/bin/env python3
"""V14: GPU re-recognition of MyKad lower-field OCR boxes.

V13 fixed layout and whitespace errors.  V14 targets the remaining recognition
errors: it reuses the V3 detector quadrilaterals, crops only lower-left MyKad
field boxes, and sends 3x Lanczos/CLAHE/unsharp variants through the CUDA
SVTRv2 Server recognizer.  A consensus guard decides whether any enhanced text
replaces the original OCR line.  V13's nested validation remains mandatory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V13_PATH = PROJECT / "scripts_v13" / "01_mykad_guarded_format_consensus.py"
V3_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v14_gpu_targeted_reocr"
SEED = 1808
FAMILY = "MYKAD"
GPU_BATCH = 256
VARIANTS = ("lanczos3", "clahe3", "unsharp3")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V13 = load_module(V13_PATH, "v13_v14_support")
V3 = V13.V3
V10 = V13.V10
V12 = V13.V12
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


def crop_variants(crop: np.ndarray) -> dict[str, np.ndarray]:
    enlarged = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if enlarged.ndim == 3 else enlarged
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    unsharp = cv2.addWeighted(clahe, 1.55, cv2.GaussianBlur(clahe, (0, 0), 1.05), -0.55, 0)
    return {
        "lanczos3": enlarged if enlarged.ndim == 3 else cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR),
        "clahe3": cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        "unsharp3": cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR),
    }


def lower_field_box(box: np.ndarray, width: int, height: int) -> bool:
    center = box.mean(axis=0)
    # Canonical MyKad address is on the lower-left.  This intentionally avoids
    # the nationality/religion/gender block under the portrait at the right.
    return bool(center[0] / max(1, width) <= 0.72 and center[1] / max(1, height) >= 0.43)


def choose_text(existing_text: str, existing_confidence: float, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    options = [{"variant": "existing", "text": core.output_text(existing_text), "confidence": float(existing_confidence)}]
    options.extend(candidate for candidate in candidates if core.output_text(candidate.get("text", "")))
    if len(options) == 1:
        return {**options[0], "agreement": 0.0, "accepted": False}
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for option in options:
        agreements = [
            fuzz.ratio(V10.V8.clean(option["text"]), V10.V8.clean(other["text"])) / 100.0
            for other in options if other is not option and V10.V8.clean(other["text"])
        ]
        agreement = float(np.mean(agreements)) if agreements else 0.0
        score = 0.58 * option["confidence"] + 0.42 * agreement
        scored.append((score, agreement, option))
    _, agreement, best = max(scored, key=lambda item: (item[0], item[1], item[2]["confidence"], len(V10.V8.clean(item[2]["text"]))))
    accepted = bool(
        best["variant"] != "existing"
        and agreement >= 0.84
        and best["confidence"] >= float(existing_confidence) - 0.03
        and V10.V8.clean(best["text"]) != V10.V8.clean(existing_text)
    )
    return {**best, "agreement": agreement, "accepted": accepted}


def collect_gpu_jobs(base: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    selected = V10.V8.select_ocr_rows(base)
    jobs: list[dict[str, Any]] = []
    source: dict[tuple[str, int], dict[str, Any]] = {}
    for document in base.loc[base.document_family.eq(FAMILY)].itertuples(index=False):
        ocr = selected.get(str(document.filename))
        if ocr is None:
            continue
        image = cv2.imread(str(ocr.get("input_path", "")), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        for ordinal, item in enumerate(V10.V8.read_items(ocr.get("ocr_items", []))):
            box = V10.V8.item_box(item)
            if box is None or not lower_field_box(box, width, height):
                continue
            try:
                crop = core.perspective_crop(image, box)
            except Exception:
                continue
            if min(crop.shape[:2]) < 4:
                continue
            key = (str(document.filename), ordinal)
            source[key] = {
                "filename": str(document.filename), "ordinal": ordinal,
                "old_text": core.output_text(item.get("text", "")), "old_confidence": float(item.get("confidence", 0.0)),
            }
            for variant, enhanced in crop_variants(crop).items():
                jobs.append({"filename": key[0], "ordinal": ordinal, "variant": variant, "crop": enhanced})
    return jobs, source


def gpu_reocr(base: pd.DataFrame) -> pd.DataFrame:
    jobs, source = collect_gpu_jobs(base)
    print(json.dumps({"stage": "gpu_reocr_prepared", "lines": len(source), "crops": len(jobs), "batch": GPU_BATCH}, ensure_ascii=False), flush=True)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    engine = V3.PPOCRSVTRv2Server()
    started = time.perf_counter()
    try:
        for start in range(0, len(jobs), GPU_BATCH):
            chunk = jobs[start:start + GPU_BATCH]
            predictions = engine.recognizer.predict_batch([job["crop"] for job in chunk], batch_size=GPU_BATCH)
            for job, prediction in zip(chunk, predictions):
                grouped[(job["filename"], job["ordinal"])].append({
                    "variant": job["variant"], "text": core.output_text(prediction.get("text", "")),
                    "confidence": float(prediction.get("confidence", 0.0)),
                })
            print(f"GPU re-OCR crops: {min(start + len(chunk), len(jobs))}/{len(jobs)}", flush=True)
    finally:
        engine.close()
    records: list[dict[str, Any]] = []
    for key, original in source.items():
        selected = choose_text(original["old_text"], original["old_confidence"], grouped.get(key, []))
        records.append({
            **original, "new_text": selected["text"], "new_confidence": selected["confidence"],
            "chosen_variant": selected["variant"], "variant_agreement": selected["agreement"], "accepted": selected["accepted"],
        })
    manifest = pd.DataFrame(records)
    manifest["gpu_elapsed_seconds"] = time.perf_counter() - started
    return manifest


def apply_replacements(lines: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    output = lines.copy()
    accepted = manifest.loc[manifest.accepted.astype(bool)].set_index(["filename", "ordinal"])
    replacements = 0
    for index, row in output.iterrows():
        key = (str(row.filename), int(row.ordinal))
        if key not in accepted.index:
            continue
        replacement = accepted.loc[key]
        text = core.output_text(replacement.new_text)
        normalized = V10.V8.clean(text)
        letters = sum(character.isalpha() for character in normalized)
        digits = sum(character.isdigit() for character in normalized)
        output.at[index, "text"] = text
        output.at[index, "ocr_confidence"] = float(replacement.new_confidence)
        output.at[index, "alpha_ratio"] = letters / max(1, letters + digits)
        output.at[index, "digit_ratio"] = digits / max(1, letters + digits)
        output.at[index, "street_cue"] = float(bool(V10.V8.STREET_PATTERN.search(normalized)))
        output.at[index, "metadata"] = float(bool(V10.V8.METADATA_PATTERN.search(normalized)))
        output.at[index, "identifier"] = float(bool(V10.V8.ID_PATTERN.search(normalized)))
        output.at[index, "date_like"] = float(bool(core.normalize_date(normalized)))
        replacements += 1
    print(json.dumps({"stage": "apply_gpu_replacements", "accepted_line_replacements": replacements}, ensure_ascii=False), flush=True)
    return output


def run() -> None:
    base = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda value: value.map(V3.natural_key)).reset_index(drop=True)
    mykad = base.loc[base.document_family.eq(FAMILY)].copy()
    manifest = gpu_reocr(base)
    atomic_csv(RESULTS / "gpu_reocr_lines.csv", manifest)
    lines = V10.flatten_lines_fast(mykad)
    lines = apply_replacements(lines, manifest)
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
    validation_candidate, span_decisions, fallback, format_decisions = V13.predict(validation_base, spans.loc[spans.filename.isin(validation_files)], model, threshold, formatter)
    baseline_metrics = V10.mykad_metrics(validation_base)
    candidate_metrics = V10.mykad_metrics(validation_candidate)
    release = bool(calibration["selected"]["safe"]) and candidate_metrics["exact_match"] >= baseline_metrics["exact_match"] and candidate_metrics["mean_similarity"] >= baseline_metrics["mean_similarity"] + 0.03
    validation = {
        "gpu": {"available": bool(torch.cuda.is_available()), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "batch": GPU_BATCH, "variants": list(VARIANTS)},
        "line_candidates": int(len(manifest)), "accepted_line_replacements": int(manifest.accepted.astype(bool).sum()),
        "fuzzy_metadata_lines_masked": fuzzy_metadata_lines, "threshold": threshold, "calibration": calibration,
        "baseline": baseline_metrics, "candidate": candidate_metrics, "span_changes": int(len(span_decisions)), "fallbacks": int(len(fallback)), "format_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    final_model = V10.SpanRanker().fit(spans)
    final_formatter = V13.V11.AddressFormatter(base.loc[base.has_ground_truth.astype(bool) & base.document_family.eq(FAMILY), "address_gt"])
    candidate, span_decisions, fallback, format_decisions = V13.predict(base, spans, final_model, threshold, final_formatter)
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
        "gpu_reocr_line_candidates": int(len(manifest)), "gpu_reocr_accepted_lines": int(manifest.accepted.astype(bool).sum()),
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
    print(json.dumps({"pipeline": "V14 lower-left MyKad boxes -> 3 GPU enhanced variants -> V13 validation", "gpu_required": True, "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
