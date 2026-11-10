#!/usr/bin/env python3
"""V26: geometry-only, supervised MyKad name-span ranking.

MyKad names are often split across two OCR lines, while the legacy family
parser selects one line.  Candidate spans use at most two spatially adjacent
lines.  A random-forest ranker sees only geometry, OCR confidence, text shape,
and distance from the card-number line; name text and ground truth are never
features.  A group-disjoint holdout gate is required before release.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V18_PATH = PROJECT / "scripts_v18" / "01_group_consensus_generic_fields.py"
V3_RESULTS = PROJECT / "results_v3_stable"
V25_RESULTS = PROJECT / "results_v25_mykad_address_format_gate"
RESULTS = PROJECT / "results_v26_mykad_name_span_ranker"
SEED = 61
BAD_LABELS = (
    "MALAYSIA", "MYKAD", "KAD PENGENALAN", "WARGANEGARA", "LELAKI",
    "PEREMPUAN", "ISLAM", "CITIZEN", "MALE", "FEMALE",
)
STREET = re.compile(r"\b(JALAN|JL\.?|LORONG|KAMPUNG|TAMAN|DESA|RT\.?|RW\.?|BLOK|BLK\.?|PT\s*\d|NO\.?\s*\d|KOTA|KUALA)\b", re.I)
ID = re.compile(r"\d{6}[- ]?\d{2}[- ]?\d{4}")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V18 = load_module(V18_PATH, "v18_v26_support")
V3, V14, core = V18.V3, V18.V14, V18.core


def text(value: Any) -> str:
    return re.sub(r"\s+", " ", core.output_text(value).upper()).strip()


def item_box(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=float).reshape(4, 2)
    except (TypeError, ValueError):
        try:
            return np.vstack([np.asarray(point, dtype=float).reshape(1, 2) for point in value]).reshape(4, 2)
        except (TypeError, ValueError):
            return None


def ocr_tables() -> dict[str, pd.DataFrame]:
    return {
        method: pd.read_parquet(V3_RESULTS / f"ocr_{method}.parquet").set_index("filename")
        for method in ("raw", "gray", "gray_clahe")
    }


def document_lines(row: Any, tables: dict[str, pd.DataFrame]) -> tuple[list[tuple[float, float, float, float, str, float]], float | None]:
    table = tables.get(str(getattr(row, "selected_preprocessing_method", "raw")), tables["raw"])
    source = table.loc[str(row.filename)]
    items = source.ocr_items
    if isinstance(items, str):
        items = json.loads(items)
    lines: list[tuple[float, float, float, float, str, float]] = []
    id_positions: list[float] = []
    for item in items:
        item = dict(item)
        value = text(item.get("text", ""))
        box = item_box(item.get("bbox"))
        if box is None:
            continue
        x, y = float(box[:, 0].mean()), float(box[:, 1].mean())
        width, height = float(np.ptp(box[:, 0])), float(np.ptp(box[:, 1]))
        if ID.search(value):
            id_positions.append(y)
        alpha = sum(char.isalpha() for char in value)
        alnum = sum(char.isalnum() for char in value)
        if (
            len(value) < 3 or any(label in value for label in BAD_LABELS)
            or ID.search(value) or alpha / max(1, alnum) < 0.60
        ):
            continue
        lines.append((y, x, width, height, value, float(item.get("confidence", 0.0))))
    return sorted(lines), (min(id_positions) if id_positions else None)


def features(part: list[tuple[float, float, float, float, str, float]], id_y: float | None, max_x: float, max_y: float) -> list[float]:
    joined = " ".join(line[4] for line in part)
    letters, digits = sum(char.isalpha() for char in joined), sum(char.isdigit() for char in joined)
    ys, xs = [line[0] for line in part], [line[1] for line in part]
    return [
        float(len(part)), float(np.mean(ys) / max_y), float((max(ys) - min(ys)) / max_y),
        float(np.mean(xs) / max_x), float((max(xs) - min(xs)) / max_x),
        float(np.mean([line[2] for line in part]) / max_x), float(np.mean([line[3] for line in part]) / max_y),
        float(np.mean([line[5] for line in part])), float(min(1.0, len(joined) / 80)),
        float(letters / max(1, letters + digits)), float(digits / max(1, letters + digits)),
        float(min(1.0, len(joined.split()) / 8)), float(bool(STREET.search(joined))),
        float((np.mean(ys) - id_y) / max_y if id_y is not None else 0.0), float(id_y is not None),
    ]


def build_candidates(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in frame.loc[frame.document_family.eq("MYKAD")].itertuples(index=False):
        lines, id_y = document_lines(row, tables)
        max_x = max(1.0, max((line[1] + line[2] for line in lines), default=1.0))
        max_y = max(1.0, max((line[0] + line[3] for line in lines), default=1.0))
        for start in range(len(lines)):
            for count in (1, 2):
                part = lines[start:start + count]
                if len(part) < count or (count == 2 and part[-1][0] - part[0][0] > 0.22 * max_y):
                    continue
                value = text(" ".join(line[4] for line in part))
                record = {
                    "filename": str(row.filename), "candidate_text": value,
                    "target_similarity": ratio(value, text(getattr(row, "name_gt", ""))) / 100
                    if bool(getattr(row, "has_ground_truth", False)) else np.nan,
                }
                record.update({f"feature_{index}": value for index, value in enumerate(features(part, id_y, max_x, max_y))})
                records.append(record)
    return pd.DataFrame(records)


def name_metrics(frame: pd.DataFrame, filenames: set[str]) -> dict[str, Any]:
    subset = frame.loc[frame.filename.isin(filenames) & frame.has_ground_truth.astype(bool)].copy()
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None}
    return {
        "images": int(len(subset)), "exact_match": float(subset.name_pred.eq(subset.name_gt).mean()),
        "mean_similarity": float(sum(ratio(str(prediction), str(truth)) for prediction, truth in zip(subset.name_pred, subset.name_gt)) / len(subset) / 100),
    }


def address_present_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Report address quality only for records whose label has an address."""
    subset = frame.loc[
        frame.has_ground_truth.astype(bool)
        & frame.address_gt.map(lambda value: bool(core.output_text(value)))
    ].copy()
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None}
    return {
        "images": int(len(subset)), "exact_match": float(subset.address_pred.eq(subset.address_gt).mean()),
        "mean_similarity": float(sum(
            ratio(str(prediction), str(truth))
            for prediction, truth in zip(subset.address_pred, subset.address_gt)
        ) / len(subset) / 100),
    }


def choose(candidates: pd.DataFrame, train_files: set[str], prediction_files: set[str]) -> tuple[pd.Series, pd.DataFrame]:
    feature_columns = [f"feature_{index}" for index in range(15)]
    train = candidates.loc[candidates.filename.isin(train_files) & candidates.target_similarity.notna()].copy()
    model = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=2, max_features=0.8,
        n_jobs=-1, random_state=SEED,
    ).fit(train[feature_columns], train.target_similarity)
    predicted = candidates.loc[candidates.filename.isin(prediction_files)].copy()
    predicted["rank_score"] = model.predict(predicted[feature_columns])
    decisions = predicted.sort_values(["filename", "rank_score"], ascending=[True, False]).groupby("filename", as_index=False).head(1)
    return decisions.set_index("filename").candidate_text, decisions


def apply_names(base: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    output = base.copy()
    mask = output.filename.isin(values.index)
    output.loc[mask, "name_pred"] = output.loc[mask, "filename"].map(values).fillna("")
    return core.score_prediction_rows(output)


def save(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"))


def run() -> None:
    base = pd.read_parquet(V25_RESULTS / "predictions_released.parquet").sort_values(
        "filename", key=lambda values: values.map(V3.natural_key)
    ).reset_index(drop=True)
    labelled = base.loc[base.document_family.eq("MYKAD") & base.has_ground_truth.astype(bool)].copy()
    groups = labelled.identity_group.map(lambda value: text(value) or "__ungrouped")
    train_pos, validation_pos = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(labelled, groups=groups))
    train_files, validation_files = set(labelled.iloc[train_pos].filename), set(labelled.iloc[validation_pos].filename)
    tables = ocr_tables()
    candidates = build_candidates(base, tables)
    V14.atomic_csv(RESULTS / "name_span_candidates.csv", candidates)

    validation_values, validation_decisions = choose(candidates, train_files, validation_files)
    validation_candidate = apply_names(base, validation_values)
    validation_baseline = name_metrics(base, validation_files)
    validation_after = name_metrics(validation_candidate, validation_files)
    safe = bool(
        validation_baseline["images"] >= 20
        and validation_after["exact_match"] > validation_baseline["exact_match"]
        and validation_after["mean_similarity"] >= validation_baseline["mean_similarity"] + 0.04
    )

    all_train_files = set(labelled.filename)
    all_mykad_files = set(base.loc[base.document_family.eq("MYKAD"), "filename"])
    values, decisions = choose(candidates, all_train_files, all_mykad_files)
    candidate = apply_names(base, values)
    release = safe
    released = candidate if release else base.copy()
    validation = {
        "scope": "MyKad names; 1-2 OCR-line geometry-only span ranking",
        "train_documents": len(train_files), "validation_documents": len(validation_files),
        "candidate_spans": int(len(candidates)), "validation_baseline": validation_baseline,
        "validation_candidate": validation_after, "span_ranker_safe": safe,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    save("predictions_candidate", candidate)
    save("predictions_released", released)
    V14.atomic_csv(RESULTS / "validation_name_span_decisions.csv", validation_decisions)
    V14.atomic_csv(RESULTS / "name_span_decisions.csv", decisions)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    present_address = address_present_metrics(released)
    metrics.update({
        "mykad_name_span_ranker_safe": safe, "mykad_name_span_candidates": int(len(candidates)),
        "address_present_documents": present_address["images"],
        "address_present_exact_match": present_address["exact_match"],
        "address_present_similarity": present_address["mean_similarity"],
        "gpu_evidence_reused": True, "test_release_gate_passed": release,
        "official_submission_overwritten": False,
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V26 MyKad name span ranker", "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
