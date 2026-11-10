#!/usr/bin/env python3
"""V7: address-only ROI geometry recovery and GPU OCR.

V7 is deliberately narrower than another full-document OCR run.  It consumes
the V3 adaptive OCR result, finds the existing address-label/field evidence,
and OCRs only that region after fine deskew and 2x enlargement.  Existing V3
quadrilateral text boxes are transformed into the ROI; all crops from raw,
grayscale, CLAHE, and gamma versions are batched through SVTRv2 on CUDA.

The script never uses ground-truth text as an OCR feature and never overwrites
the official submission.  A disjoint identity hold-out must improve non-empty
address exact match without reducing overall address EM before its test
candidates are released.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
V3_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v7_address_roi"
CACHE = PROJECT / "cache_v7_address_roi"

SEED = 1808
ANCHOR_THRESHOLD = 0.58
MIN_VARIANT_SCORE = 0.56
MAX_TEXT_LENGTH = 180
GPU_RECOGNITION_BATCH = 256


def load_v3() -> Any:
    spec = importlib.util.spec_from_file_location("v3_address_roi_support", V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load V3 helper: {V3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_v3()
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


def clean(value: Any) -> str:
    return V3.clean(value)


def read_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def parse_node_ids(value: Any) -> list[int]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return [int(item) for item in parsed] if isinstance(parsed, (list, tuple, np.ndarray)) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def item_box(item: dict[str, Any]) -> np.ndarray | None:
    try:
        box = np.asarray(item.get("bbox", []), dtype=np.float32).reshape(4, 2)
    except (TypeError, ValueError):
        # PyArrow restores some nested OCR boxes as an object array of points.
        # Convert point-by-point rather than discarding valid layout evidence.
        try:
            box = np.vstack([np.asarray(point, dtype=np.float32).reshape(1, 2) for point in item.get("bbox", [])]).reshape(4, 2)
        except (TypeError, ValueError):
            return None
    return box if np.isfinite(box).all() else None


def detect_boxes(engine: Any, image: np.ndarray) -> list[np.ndarray]:
    result = engine.detector(image)
    boxes = getattr(result, "boxes", None)
    if boxes is None and isinstance(result, (list, tuple)) and result:
        boxes = result[0]
    if boxes is None:
        return []
    output: list[np.ndarray] = []
    for value in boxes:
        try:
            output.append(np.asarray(value, dtype=np.float32).reshape(4, 2))
        except (TypeError, ValueError):
            continue
    return output


def estimate_skew_degrees(image: np.ndarray) -> float:
    """Small text-line deskew; do not invent large rotations from noisy cards."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minimum = max(25, min(image.shape[:2]) // 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=minimum, maxLineGap=14)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        while angle <= -45:
            angle += 90
        while angle > 45:
            angle -= 90
        if abs(angle) <= 12:
            angles.append(angle)
    if len(angles) < 3:
        return 0.0
    angle = float(np.median(angles))
    return angle if abs(angle) >= 0.35 else 0.0


def rotate_keep_size(image: np.ndarray, degrees: float) -> np.ndarray:
    if not degrees:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def transform_box_to_roi(box: np.ndarray, roi: list[int], roi_shape: tuple[int, int], rotation_degrees: float) -> np.ndarray:
    """Map a V3 full-document quadrilateral into the enlarged ROI geometry."""
    left, top, _, _ = roi
    points = box.astype(np.float32).copy()
    points[:, 0] -= float(left)
    points[:, 1] -= float(top)
    if rotation_degrees:
        height, width = roi_shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_degrees, 1.0)
        points = cv2.transform(points.reshape(1, 4, 2), matrix).reshape(4, 2)
    return points * 2.0


def boxes_inside_roi(items: list[dict[str, Any]], roi_meta: dict[str, Any], roi_shape: tuple[int, int], rotation_degrees: float) -> list[np.ndarray]:
    bounds = roi_meta.get("roi", [])
    if len(bounds) != 4:
        return []
    left, top, right, bottom = map(float, bounds)
    output: list[np.ndarray] = []
    for item in items:
        box = item_box(item)
        if box is None:
            continue
        center = box.mean(axis=0)
        if left <= center[0] <= right and top <= center[1] <= bottom:
            output.append(transform_box_to_roi(box, list(map(int, bounds)), roi_shape, rotation_degrees))
    return output


def enhance_variants(roi: np.ndarray) -> dict[str, np.ndarray]:
    """Geometry is identical in every output so detected boxes can be reused."""
    enlarged = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if enlarged.ndim == 3 else enlarged
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    gamma = np.power(gray.astype(np.float32) / 255.0, 0.82).clip(0, 1)
    gamma = (gamma * 255).astype(np.uint8)
    unsharp = cv2.addWeighted(clahe, 1.45, cv2.GaussianBlur(clahe, (0, 0), 1.1), -0.45, 0)
    return {
        # Keep OpenCV BGR layout, matching the V3 detector/recognizer path.
        "raw_2x": enlarged if enlarged.ndim == 3 else cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR),
        "gray_2x": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "clahe_2x": cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        "gamma_unsharp_2x": cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR),
    }


def address_roi(image: np.ndarray, items: list[dict[str, Any]], selected_ids: list[int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Use existing layout evidence; never crop a speculative full document."""
    height, width = image.shape[:2]
    anchors: list[tuple[float, np.ndarray]] = []
    for item in items:
        box = item_box(item)
        if box is None:
            continue
        score = core.anchor_score(core.output_text(item.get("text", "")), "address")
        if score >= ANCHOR_THRESHOLD:
            anchors.append((float(score), box))
    if anchors:
        score, box = max(anchors, key=lambda item: item[0])
        x0, y0 = box.min(axis=0); x1, y1 = box.max(axis=0)
        # Address values can be right of a label or continue for several lines
        # below it.  Retain a compact field region, not the entire card.
        left = max(0, int(x0 - 0.05 * width)); right = min(width, int(max(x1 + 0.68 * width, x0 + 0.62 * width)))
        top = max(0, int(y0 - 0.02 * height)); bottom = min(height, int(y1 + 0.38 * height))
        if right - left >= 32 and bottom - top >= 20:
            return image[top:bottom, left:right].copy(), {"roi_source": "address_anchor", "anchor_score": score, "roi": [left, top, right, bottom]}
    # Many MyKad-like cards have no explicit ADDRESS label.  A street/address
    # cue still gives safe geometry: expand only around that cue and its nearby
    # lines, never around the entire document.
    cue_pattern = re.compile(r"\b(JALAN|JL\.?|LORONG|KAMPUNG|TAMAN|DESA|RT\.?|RW\.?|BLOK|BLK\.?|PT\s*\d|NO\.?\s*\d)", re.I)
    cue_boxes = [box for item in items if cue_pattern.search(core.output_text(item.get("text", ""))) and (box := item_box(item)) is not None]
    if cue_boxes:
        points = np.vstack(cue_boxes); x0, y0 = points.min(axis=0); x1, y1 = points.max(axis=0)
        left = max(0, int(x0 - 0.12 * width)); right = min(width, int(x1 + 0.34 * width))
        top = max(0, int(y0 - 0.09 * height)); bottom = min(height, int(y1 + 0.17 * height))
        if right - left >= 32 and bottom - top >= 20:
            return image[top:bottom, left:right].copy(), {"roi_source": "address_text_cue", "anchor_score": 0.0, "roi": [left, top, right, bottom]}
    nodes = [item_box(items[index]) for index in selected_ids if 0 <= index < len(items) and item_box(items[index]) is not None]
    if nodes:
        points = np.vstack(nodes); x0, y0 = points.min(axis=0); x1, y1 = points.max(axis=0)
        left = max(0, int(x0 - 0.14 * width)); right = min(width, int(x1 + 0.22 * width))
        top = max(0, int(y0 - 0.10 * height)); bottom = min(height, int(y1 + 0.14 * height))
        if right - left >= 32 and bottom - top >= 20:
            return image[top:bottom, left:right].copy(), {"roi_source": "existing_address_nodes", "anchor_score": 0.0, "roi": [left, top, right, bottom]}
    return None, {"roi_source": "no_layout_evidence", "anchor_score": 0.0, "roi": []}


LABEL_TOKENS = ("ADDRESS", "ALAMAT", "NAMA", "NAME", "DATE OF BIRTH", "BIRTH DATE", "NATIONALITY", "TEMPAT LAHIR")


def line_is_address_like(text: str) -> bool:
    normalized = clean(text)
    if len(normalized) < 3 or len(normalized) > 110:
        return False
    if any(token in normalized for token in LABEL_TOKENS):
        return False
    if core.normalize_date(normalized):
        return False
    digits = sum(char.isdigit() for char in normalized)
    if digits >= 9 and digits / max(1, len(normalized)) > 0.45:
        return False
    return sum(char.isalpha() for char in normalized) >= 2


def candidate_from_variant(lines: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [line for line in lines if line_is_address_like(line["text"])]
    usable.sort(key=lambda line: (round(line["y"] / max(1.0, line["height"])), line["x"]))
    if not usable:
        return {"text": "", "score": 0.0, "line_count": 0, "mean_confidence": 0.0}
    # Collapse detector duplicates at the same line position while retaining the
    # higher-confidence recognition.  This prevents merged duplicate text.
    unique: list[dict[str, Any]] = []
    for line in usable:
        match = next((old for old in unique if abs(old["y"] - line["y"]) <= max(7.0, old["height"], line["height"]) and fuzz.ratio(clean(old["text"]), clean(line["text"])) >= 86), None)
        if match is None:
            unique.append(line)
        elif line["confidence"] > match["confidence"]:
            match.update(line)
    text = core.output_text(", ".join(line["text"] for line in unique))
    confidence = float(np.mean([line["confidence"] for line in unique]))
    length_score = min(1.0, len(clean(text)) / 36.0)
    cue = bool(re.search(r"\b(JL|JALAN|RT|RW|KEL|KEC|DESA|KOTA|NO|BLOCK|BLK|LORONG|KAMPUNG)\b", clean(text)))
    score = 0.58 * confidence + 0.24 * length_score + 0.10 * min(1.0, len(unique) / 2) + 0.08 * float(cue)
    return {"text": text, "score": float(score), "line_count": len(unique), "mean_confidence": confidence}


def choose_variant(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    available = [(name, value) for name, value in candidates.items() if value["text"]]
    if not available:
        return {"text": "", "score": 0.0, "variant": "", "agreement": 0.0, "line_count": 0, "mean_confidence": 0.0}
    scored = []
    for name, value in available:
        agreement = max((fuzz.ratio(clean(value["text"]), clean(other["text"])) / 100.0 for other_name, other in available if other_name != name), default=0.0)
        scored.append((value["score"] + 0.15 * agreement, agreement, name, value))
    _, agreement, name, value = max(scored, key=lambda item: (item[0], item[1], item[3]["mean_confidence"], len(clean(item[3]["text"]))))
    return {"text": value["text"], "score": value["score"], "variant": name, "agreement": agreement, "line_count": value["line_count"], "mean_confidence": value["mean_confidence"]}


def cache_path(filename: str) -> Path:
    return CACHE / "roi_ocr" / f"{Path(filename).stem}.json"


def select_ocr_rows(base: pd.DataFrame) -> dict[str, pd.Series]:
    tables = {
        "raw": pd.read_parquet(V3_RESULTS / "ocr_raw.parquet"),
        "gray": pd.read_parquet(V3_RESULTS / "ocr_gray.parquet"),
        "gray_clahe": pd.read_parquet(V3_RESULTS / "ocr_gray_clahe.parquet"),
    }
    index = {name: table.set_index("filename", drop=False) for name, table in tables.items()}
    selected: dict[str, pd.Series] = {}
    for row in base.itertuples(index=False):
        # ``preprocessing_method`` is the adaptive policy label in V3; preserve
        # the actual raw/gray/CLAHE method that produced its OCR boxes.
        method = str(getattr(row, "selected_preprocessing_method", getattr(row, "preprocessing_method", "raw")))
        table = index.get(method, index["raw"])
        if row.filename in table.index:
            selected[str(row.filename)] = table.loc[row.filename]
    return selected


def process_one(row: pd.Series, ocr_row: pd.Series, engine: Any, overwrite: bool) -> dict[str, Any]:
    target = cache_path(str(row.filename))
    if target.exists() and not overwrite:
        try:
            cached = json.loads(target.read_text(encoding="utf-8"))
            if cached.get("filename") == row.filename:
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    started = time.perf_counter()
    items = read_items(ocr_row.get("ocr_items", []))
    input_path = Path(str(ocr_row.get("input_path", "")))
    record: dict[str, Any] = {
        "filename": str(row.filename), "status": "ok", "error": "", "address_candidate": "", "candidate_score": 0.0,
        "candidate_variant": "", "candidate_agreement": 0.0, "candidate_line_count": 0, "candidate_mean_confidence": 0.0,
        "gpu_crop_count": 0, "detected_lines": 0, "deskew_degrees": 0.0,
    }
    try:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(input_path))
        roi, roi_meta = address_roi(image, items, parse_node_ids(row.get("selected_address_nodes", "[]")))
        record.update(roi_meta)
        if roi is None:
            record["status"] = "no_roi"
        else:
            angle = estimate_skew_degrees(roi)
            # A positive Hough angle needs a negative image rotation.  Existing
            # V3 quadrilateral boxes are transformed with the same affine map.
            applied_rotation = -angle
            original_shape = roi.shape[:2]
            roi = rotate_keep_size(roi, applied_rotation)
            record["deskew_degrees"] = float(applied_rotation)
            variants = enhance_variants(roi)
            # Reuse V3's already computed quadrilateral text boxes.  Each box
            # is perspective-cropped from four ROI variants and recognized in a
            # CUDA batch, avoiding a second CPU detector pass per image.
            boxes = boxes_inside_roi(items, roi_meta, original_shape, applied_rotation)
            record["detected_lines"] = len(boxes)
            crops: list[np.ndarray] = []
            metadata: list[tuple[str, np.ndarray]] = []
            for name, variant in variants.items():
                for box in boxes:
                    try:
                        crops.append(core.perspective_crop(variant, box))
                        metadata.append((name, box))
                    except Exception:
                        continue
            record["gpu_crop_count"] = len(crops)
            recognized = engine.recognizer.predict_batch(crops, batch_size=GPU_RECOGNITION_BATCH)
            by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
            for (name, box), prediction in zip(metadata, recognized):
                text = core.output_text(prediction.get("text", ""))
                if not text:
                    continue
                x0, y0 = box.min(axis=0); x1, y1 = box.max(axis=0)
                by_variant[name].append({"text": text, "confidence": float(prediction.get("confidence", 0.0)), "x": float((x0 + x1) / 2), "y": float((y0 + y1) / 2), "height": float(max(1.0, y1 - y0))})
            chosen = choose_variant({name: candidate_from_variant(lines) for name, lines in by_variant.items()})
            record.update({
                "address_candidate": chosen["text"], "candidate_score": chosen["score"], "candidate_variant": chosen["variant"],
                "candidate_agreement": chosen["agreement"], "candidate_line_count": chosen["line_count"], "candidate_mean_confidence": chosen["mean_confidence"],
            })
    except Exception as error:
        record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)
    return record


def address_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    labeled = frame.loc[frame.has_ground_truth.astype(bool)].copy()
    present = labeled.address_present_gt.astype(bool)
    return {
        "labeled_rows": int(len(labeled)),
        "address_exact_match": float(labeled.address_exact.mean()),
        "nonempty_address_rows": int(present.sum()),
        "nonempty_address_exact_match": float(labeled.loc[present, "address_exact"].mean()) if present.any() else None,
        "nonempty_address_mean_similarity": float(labeled.loc[present, "address_similarity"].mean()) if present.any() else None,
    }


def make_candidate(base: pd.DataFrame, roi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = base.copy().merge(roi, on="filename", how="left", validate="one_to_one")
    decisions: list[dict[str, Any]] = []
    for index, row in output.iterrows():
        value = core.output_text(row.get("address_candidate", ""))
        score = float(row.get("candidate_score", 0.0) or 0.0)
        agreement = float(row.get("candidate_agreement", 0.0) or 0.0)
        previous = core.output_text(row.address_pred)
        accepted = bool(value and score >= MIN_VARIANT_SCORE and (agreement >= 0.72 or float(row.get("candidate_mean_confidence", 0.0) or 0.0) >= 0.90) and clean(value) != clean(previous))
        if not accepted:
            continue
        output.at[index, "address_pred"] = value
        decisions.append({
            "filename": row.filename, "data_split": row.data_split, "previous_address": previous, "roi_address": value,
            "candidate_score": score, "variant_agreement": agreement, "variant": row.get("candidate_variant", ""),
            "roi_source": row.get("roi_source", ""), "deskew_degrees": row.get("deskew_degrees", 0.0),
        })
    return core.score_prediction_rows(output), pd.DataFrame(decisions)


def validation_gate(base: pd.DataFrame, candidate: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    labeled = base.loc[base.has_ground_truth.astype(bool)].copy()
    groups = [V3.identity_key(row) for _, row in labeled.iterrows()]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    _, validation_positions = next(splitter.split(labeled, groups=groups))
    filenames = set(labeled.iloc[validation_positions].filename)
    baseline_metrics = address_metrics(base.loc[base.filename.isin(filenames)])
    candidate_metrics = address_metrics(candidate.loc[candidate.filename.isin(filenames)])
    passed = (
        candidate_metrics["nonempty_address_exact_match"] > baseline_metrics["nonempty_address_exact_match"]
        and candidate_metrics["address_exact_match"] >= baseline_metrics["address_exact_match"]
    )
    return bool(passed), {"validation_images": int(len(filenames)), "baseline": baseline_metrics, "candidate": candidate_metrics, "passed": bool(passed)}


def run(max_images: int | None, overwrite: bool) -> pd.DataFrame:
    base = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda column: column.map(V3.natural_key)).reset_index(drop=True)
    selected_ocr = select_ocr_rows(base)
    rows = base if max_images is None else base.head(max_images)
    engine = V3.PPOCRSVTRv2Server()
    records: list[dict[str, Any]] = []
    try:
        for number, (_, row) in enumerate(rows.iterrows(), start=1):
            ocr_row = selected_ocr.get(str(row.filename))
            if ocr_row is None:
                records.append({"filename": row.filename, "status": "missing_selected_ocr", "error": "", "address_candidate": "", "candidate_score": 0.0})
                continue
            records.append(process_one(row, ocr_row, engine, overwrite=overwrite))
            if number % 25 == 0 or number == len(rows):
                print(f"address ROI OCR: {number}/{len(rows)} committed", flush=True)
    finally:
        engine.close()
    roi = pd.DataFrame(records)
    atomic_csv(RESULTS / "address_roi_manifest.csv", roi)
    if max_images is not None:
        return roi
    candidate, decisions = make_candidate(base, roi)
    gate, validation = validation_gate(base, candidate)
    released = candidate if gate else core.score_prediction_rows(base.copy())
    atomic_parquet(RESULTS / "predictions_address_roi_candidate.parquet", candidate)
    atomic_csv(RESULTS / "predictions_address_roi_candidate.csv", candidate.drop(columns=[column for column in candidate if column.endswith("_detail")], errors="ignore"))
    atomic_parquet(RESULTS / "predictions_address_roi_released.parquet", released)
    atomic_csv(RESULTS / "predictions_address_roi_released.csv", released.drop(columns=[column for column in released if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "address_roi_decisions.csv", decisions)
    atomic_json(RESULTS / "validation_gate.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "candidate_changed_addresses": int(len(decisions)), "test_release_gate_passed": bool(gate),
        "official_submission_overwritten": False,
        **{f"released_{key}": value for key, value in address_metrics(released).items()},
        **{f"candidate_{key}": value for key, value in address_metrics(candidate).items()},
    })
    atomic_json(RESULTS / "metrics_address_roi.json", metrics)
    submission = released.loc[released.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, got {len(submission)}")
    atomic_csv(RESULTS / "test_predictions_address_roi.csv", submission)
    return released


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=None, help="Smoke-test only this many sorted images; no final metrics or submission.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute cached ROI records after a pipeline revision.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V7 address ROI -> Hough fine deskew -> 2x variants -> reused V3 boxes + batched CUDA SVTRv2 recognition",
        "gpu_recognition_batch": GPU_RECOGNITION_BATCH,
        "detector_rerun": False,
        "geometry_source": "V3 PP-OCRv6 detector boxes; no conflicting second detector runtime",
        "official_results_untouched": True,
    }, ensure_ascii=False, indent=2), flush=True)
    run(args.max_images, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
