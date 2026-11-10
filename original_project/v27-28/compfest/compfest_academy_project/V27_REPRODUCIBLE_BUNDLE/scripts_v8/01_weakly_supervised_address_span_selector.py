#!/usr/bin/env python3
"""V8: weakly supervised address-line selection from existing OCR boxes.

The V7 geometry pass showed that address-like text becomes more similar to the
target, but whole-ROI output often contains extra card metadata.  V8 therefore
learns which *existing OCR lines* belong to the address.  On labelled images,
each line receives a weak label from its fuzzy overlap with the address ground
truth.  The learnt model receives only line text, geometry, OCR confidence and
document-family features; it never receives a ground-truth value at inference.

At inference it scores all OCR lines, chooses a compact spatially consecutive
address span, and leaves name/DOB unchanged.  Train/validation splitting is by
identity, and a held-out metric gate decides whether any test candidate may be
released.  The official submission remains untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
V3_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v8_address_span"

SEED = 1808
LINE_MATCH_THRESHOLD = 0.76
SPAN_MAX_LINES = 4
THRESHOLD_OPTIONS = tuple(np.round(np.arange(0.40, 0.81, 0.05), 2))


def load_v3() -> Any:
    spec = importlib.util.spec_from_file_location("v3_span_selector_support", V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V3 helper: {V3_PATH}")
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


def item_box(item: dict[str, Any]) -> np.ndarray | None:
    try:
        box = np.asarray(item.get("bbox", []), dtype=np.float32).reshape(4, 2)
    except (TypeError, ValueError):
        try:
            box = np.vstack([np.asarray(point, dtype=np.float32).reshape(1, 2) for point in item.get("bbox", [])]).reshape(4, 2)
        except (TypeError, ValueError):
            return None
    return box if np.isfinite(box).all() else None


def select_ocr_rows(base: pd.DataFrame) -> dict[str, pd.Series]:
    tables = {
        "raw": pd.read_parquet(V3_RESULTS / "ocr_raw.parquet"),
        "gray": pd.read_parquet(V3_RESULTS / "ocr_gray.parquet"),
        "gray_clahe": pd.read_parquet(V3_RESULTS / "ocr_gray_clahe.parquet"),
    }
    by_method = {name: table.set_index("filename", drop=False) for name, table in tables.items()}
    selected: dict[str, pd.Series] = {}
    for row in base.itertuples(index=False):
        method = str(getattr(row, "selected_preprocessing_method", getattr(row, "preprocessing_method", "raw")))
        table = by_method.get(method, by_method["raw"])
        if row.filename in table.index:
            selected[str(row.filename)] = table.loc[row.filename]
    return selected


STREET_PATTERN = re.compile(r"\b(JALAN|JL\.?|LORONG|KAMPUNG|TAMAN|DESA|RT\.?|RW\.?|BLOK|BLK\.?|PT\s*\d|NO\.?\s*\d|KOTA|KUALA)\b", re.I)
METADATA_PATTERN = re.compile(r"\b(NAME|NAMA|BIRTH|LAHIR|DATE|TARIKH|NATIONALITY|WARGANEGARA|CITIZEN|AGAMA|RELIGION|ISLAM|LELAKI|PEREMPUAN|MALE|FEMALE|SEX|JANTINA|MYKAD|MALAYSIA|KAD PENGENALAN)\b", re.I)
ID_PATTERN = re.compile(r"\b\d{6}[- ]?\d{2}[- ]?\d{4}\b|\b\d{8,}\b")


def line_match(text: str, target: str) -> float:
    """Weak supervision only: target is used to make train labels, never features."""
    source, truth = clean(text), clean(target)
    if not source or not truth or len(source) < 3:
        return 0.0
    partial = fuzz.partial_ratio(source, truth) / 100.0
    token = fuzz.token_set_ratio(source, truth) / 100.0
    ratio = fuzz.ratio(source, truth) / 100.0
    return float(max(partial * 0.82 + token * 0.18, ratio))


def flatten_lines(base: pd.DataFrame) -> pd.DataFrame:
    ocr = select_ocr_rows(base)
    rows: list[dict[str, Any]] = []
    for doc in base.itertuples(index=False):
        source = ocr.get(str(doc.filename))
        if source is None:
            continue
        items = read_items(source.get("ocr_items", []))
        boxes = [box for item in items if (box := item_box(item)) is not None]
        max_x = max(1.0, max((float(box[:, 0].max()) for box in boxes), default=1.0))
        max_y = max(1.0, max((float(box[:, 1].max()) for box in boxes), default=1.0))
        for ordinal, item in enumerate(items):
            box = item_box(item)
            if box is None:
                continue
            text = core.output_text(item.get("text", ""))
            normalized = clean(text)
            if not normalized:
                continue
            x0, y0 = box.min(axis=0); x1, y1 = box.max(axis=0)
            letters = sum(char.isalpha() for char in normalized)
            digits = sum(char.isdigit() for char in normalized)
            metadata = bool(METADATA_PATTERN.search(normalized))
            identifier = bool(ID_PATTERN.search(normalized))
            date_like = bool(core.normalize_date(normalized))
            match = line_match(text, doc.address_gt) if bool(doc.has_ground_truth) else 0.0
            rows.append({
                "filename": doc.filename, "identity_group": doc.identity_group, "has_ground_truth": bool(doc.has_ground_truth), "data_split": doc.data_split,
                "address_gt": doc.address_gt, "base_address": doc.address_pred, "document_family": doc.document_family, "quality_group": doc.quality_group,
                "ordinal": ordinal, "text": text, "normalized_text": normalized, "ocr_confidence": float(item.get("confidence", 0.0)),
                "x": float((x0 + x1) / 2 / max_x), "y": float((y0 + y1) / 2 / max_y), "width": float((x1 - x0) / max_x), "height": float((y1 - y0) / max_y),
                "text_length": len(normalized), "alpha_ratio": letters / max(1, letters + digits), "digit_ratio": digits / max(1, letters + digits),
                "street_cue": float(bool(STREET_PATTERN.search(normalized))), "metadata": float(metadata), "identifier": float(identifier), "date_like": float(date_like),
                "weak_match": match, "weak_label": int(bool(doc.has_ground_truth) and not metadata and not identifier and not date_like and match >= LINE_MATCH_THRESHOLD),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.weak_label.sum() == 0:
        raise RuntimeError("No weakly matched address lines were found; cannot train a span selector")
    return frame


NUMERIC_COLUMNS = ("ocr_confidence", "x", "y", "width", "height", "text_length", "alpha_ratio", "digit_ratio", "street_cue", "metadata", "identifier", "date_like")


class LineModel:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=12000, sublinear_tf=True)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.model = LogisticRegression(C=0.45, class_weight="balanced", max_iter=1800, random_state=SEED)

    def _features(self, frame: pd.DataFrame, fit: bool) -> sparse.csr_matrix:
        text = self.vectorizer.fit_transform(frame.normalized_text) if fit else self.vectorizer.transform(frame.normalized_text)
        numeric_values = frame.loc[:, NUMERIC_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        numeric = self.scaler.fit_transform(numeric_values) if fit else self.scaler.transform(numeric_values)
        family = frame[["document_family", "quality_group"]].fillna("UNKNOWN").astype(str)
        categorical = self.encoder.fit_transform(family) if fit else self.encoder.transform(family)
        return sparse.hstack([text, sparse.csr_matrix(numeric), categorical], format="csr")

    def fit(self, frame: pd.DataFrame) -> "LineModel":
        if frame.weak_label.nunique() < 2:
            raise RuntimeError("Line-selector training split needs positive and negative weak labels")
        self.model.fit(self._features(frame, fit=True), frame.weak_label.astype(int))
        return self

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._features(frame, fit=False))[:, 1]


def select_span(part: pd.DataFrame, line_threshold: float) -> tuple[str, dict[str, Any]]:
    """Select one compact sequence of non-metadata OCR lines."""
    rows = part.sort_values(["y", "x", "ordinal"]).reset_index(drop=True)
    valid = rows.loc[(rows.metadata < 0.5) & (rows.identifier < 0.5) & (rows.date_like < 0.5)].copy().reset_index(drop=True)
    if valid.empty:
        return "", {"span_score": 0.0, "line_count": 0, "max_probability": 0.0}
    options: list[tuple[float, list[int]]] = []
    for start in range(len(valid)):
        for end in range(start, min(len(valid), start + SPAN_MAX_LINES)):
            candidate = valid.iloc[start:end + 1]
            if candidate.probability.max() < line_threshold:
                continue
            if len(candidate) > 1 and candidate.y.diff().dropna().max() > 0.18:
                continue
            probability = candidate.probability.clip(1e-4, 1 - 1e-4).to_numpy()
            logit_score = float(np.mean(np.log(probability / (1.0 - probability))))
            cue_bonus = 0.25 * float(candidate.street_cue.max())
            length_bonus = 0.06 * min(2, len(candidate) - 1)
            score = logit_score + cue_bonus + length_bonus
            options.append((score, candidate.index.tolist()))
    if not options:
        return "", {"span_score": 0.0, "line_count": 0, "max_probability": float(valid.probability.max())}
    score, indices = max(options, key=lambda item: item[0])
    chosen = valid.iloc[indices].sort_values(["y", "x", "ordinal"])
    return core.output_text(", ".join(chosen.text.tolist())), {
        "span_score": float(score), "line_count": int(len(chosen)), "max_probability": float(chosen.probability.max()),
        "mean_probability": float(chosen.probability.mean()), "has_street_cue": bool(chosen.street_cue.max()),
    }


def build_predictions(base: pd.DataFrame, lines: pd.DataFrame, model: LineModel, line_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = lines.copy()
    scored["probability"] = model.probability(scored)
    choices: list[dict[str, Any]] = []
    for filename, part in scored.groupby("filename", sort=False):
        value, audit = select_span(part, line_threshold)
        choices.append({"filename": filename, "span_address": value, **audit})
    output = base.copy().merge(pd.DataFrame(choices), on="filename", how="left", validate="one_to_one")
    decisions: list[dict[str, Any]] = []
    for index, row in output.iterrows():
        candidate = core.output_text(row.get("span_address", ""))
        previous = core.output_text(row.address_pred)
        if not candidate or clean(candidate) == clean(previous):
            continue
        # Selection is already supervised/calibrated.  The threshold is the
        # only decision criterion; it contains no ground-truth field value.
        output.at[index, "address_pred"] = candidate
        decisions.append({
            "filename": row.filename, "data_split": row.data_split, "previous_address": previous, "span_address": candidate,
            "span_score": row.span_score, "line_count": row.line_count, "max_line_probability": row.max_probability,
            "mean_line_probability": row.mean_probability, "has_street_cue": row.has_street_cue,
        })
    return core.score_prediction_rows(output), pd.DataFrame(decisions)


def address_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    labeled = frame.loc[frame.has_ground_truth.astype(bool)]
    present = labeled.address_present_gt.astype(bool)
    return {
        "labeled_rows": int(len(labeled)), "address_exact_match": float(labeled.address_exact.mean()),
        "nonempty_address_rows": int(present.sum()),
        "nonempty_address_exact_match": float(labeled.loc[present, "address_exact"].mean()) if present.any() else None,
        "nonempty_address_mean_similarity": float(labeled.loc[present, "address_similarity"].mean()) if present.any() else None,
    }


def choose_threshold(base: pd.DataFrame, lines: pd.DataFrame, train_filenames: set[str], groups: dict[str, str]) -> tuple[float, dict[str, Any]]:
    train_docs = sorted(train_filenames)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    inner_train_pos, calibration_pos = next(splitter.split(train_docs, groups=[groups[name] for name in train_docs]))
    inner_train = {train_docs[position] for position in inner_train_pos}
    calibration = {train_docs[position] for position in calibration_pos}
    model = LineModel().fit(lines.loc[lines.filename.isin(inner_train)])
    baseline = address_metrics(base.loc[base.filename.isin(calibration)])
    options = []
    for threshold in THRESHOLD_OPTIONS:
        candidate, decisions = build_predictions(base.loc[base.filename.isin(calibration)].copy(), lines.loc[lines.filename.isin(calibration)], model, float(threshold))
        metrics = address_metrics(candidate)
        safe = metrics["address_exact_match"] >= baseline["address_exact_match"]
        options.append({"threshold": float(threshold), "safe": bool(safe), "changes": int(len(decisions)), **metrics})
    safe = [option for option in options if option["safe"]]
    selected = max(safe, key=lambda option: (option["nonempty_address_exact_match"], option["nonempty_address_mean_similarity"], -option["changes"])) if safe else max(options, key=lambda option: (option["address_exact_match"], option["nonempty_address_exact_match"], option["nonempty_address_mean_similarity"]))
    return float(selected["threshold"]), {"calibration_images": len(calibration), "baseline": baseline, "selected": selected, "options": options}


def run() -> pd.DataFrame:
    base = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda column: column.map(V3.natural_key)).reset_index(drop=True)
    lines = flatten_lines(base)
    labeled = base.loc[base.has_ground_truth.astype(bool)].copy()
    groups = {row.filename: V3.identity_key(row) for _, row in labeled.iterrows()}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_positions, validation_positions = next(splitter.split(labeled, groups=[groups[name] for name in labeled.filename]))
    train_filenames = set(labeled.iloc[train_positions].filename)
    validation_filenames = set(labeled.iloc[validation_positions].filename)
    line_threshold, calibration = choose_threshold(base, lines, train_filenames, groups)
    validation_model = LineModel().fit(lines.loc[lines.filename.isin(train_filenames)])
    validation_base = base.loc[base.filename.isin(validation_filenames)].copy()
    validation_candidate, validation_decisions = build_predictions(validation_base, lines.loc[lines.filename.isin(validation_filenames)], validation_model, line_threshold)
    baseline_metrics = address_metrics(validation_base)
    candidate_metrics = address_metrics(validation_candidate)
    release = candidate_metrics["nonempty_address_exact_match"] > baseline_metrics["nonempty_address_exact_match"] and candidate_metrics["address_exact_match"] >= baseline_metrics["address_exact_match"]
    validation = {
        "train_images": len(train_filenames), "validation_images": len(validation_filenames),
        "line_count": int(len(lines)), "weak_positive_lines": int(lines.weak_label.sum()), "line_probability_threshold": line_threshold,
        "calibration": calibration, "baseline": baseline_metrics, "candidate": candidate_metrics,
        "candidate_changes": int(len(validation_decisions)), "test_release_gate_passed": bool(release),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    final_model = LineModel().fit(lines.loc[lines.has_ground_truth.astype(bool)])
    candidate, decisions = build_predictions(base, lines, final_model, line_threshold)
    released = candidate if release else core.score_prediction_rows(base.copy())
    atomic_parquet(RESULTS / "predictions_address_span_candidate.parquet", candidate)
    atomic_csv(RESULTS / "predictions_address_span_candidate.csv", candidate.drop(columns=[column for column in candidate if column.endswith("_detail")], errors="ignore"))
    atomic_parquet(RESULTS / "predictions_address_span_released.parquet", released)
    atomic_csv(RESULTS / "predictions_address_span_released.csv", released.drop(columns=[column for column in released if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "address_span_decisions.csv", decisions)
    atomic_json(RESULTS / "validation.json", validation)
    joblib.dump({"model": final_model, "line_probability_threshold": line_threshold, "numeric_columns": NUMERIC_COLUMNS}, RESULTS / "address_span_selector.joblib")
    metrics = V3.field_metrics(released)
    metrics.update({
        "candidate_changed_addresses": int(len(decisions)), "test_release_gate_passed": bool(release), "official_submission_overwritten": False,
        **{f"released_{key}": value for key, value in address_metrics(released).items()},
        **{f"candidate_{key}": value for key, value in address_metrics(candidate).items()},
    })
    atomic_json(RESULTS / "metrics_address_span.json", metrics)
    submission = released.loc[released.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, got {len(submission)}")
    atomic_csv(RESULTS / "test_predictions_address_span.csv", submission)
    return released


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V8 weak address-line labels -> text/geometry span selector -> held-out release gate",
        "ocr_rerun": False, "gpu_required": False, "official_results_untouched": True,
    }, ensure_ascii=False, indent=2), flush=True)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
