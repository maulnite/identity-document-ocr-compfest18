#!/usr/bin/env python3
"""V10: supervised, template-scoped ranker for MyKad address spans.

V9 established that the existing OCR lines contain a much better address span
than the parser currently selects.  This experiment learns that *span choice*
directly.  Ground truth only determines the oracle span for labelled training
cards; it is never used as a feature or at inference.  A group-held-out
validation split and a strict metric gate decide whether the candidate can be
released.  The official submission is never overwritten.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
V8_PATH = PROJECT / "scripts_v8" / "01_weakly_supervised_address_span_selector.py"
V3_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v10_mykad_span_ranker"

SEED = 1808
FAMILY = "MYKAD"
MAX_SPAN_LINES = 4
MIN_ORACLE_SIMILARITY = 0.70
THRESHOLDS = tuple(np.round(np.arange(0.20, 0.91, 0.05), 2))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_module(V3_PATH, "v3_v10_support")
V8 = load_module(V8_PATH, "v8_v10_support")
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


def flatten_lines_fast(base: pd.DataFrame) -> pd.DataFrame:
    """Load only ranker features; deliberately avoids V8's fuzzy weak labels."""
    ocr = V8.select_ocr_rows(base)
    rows: list[dict[str, Any]] = []
    for document in base.itertuples(index=False):
        source = ocr.get(str(document.filename))
        if source is None:
            continue
        items = V8.read_items(source.get("ocr_items", []))
        boxes = [box for item in items if (box := V8.item_box(item)) is not None]
        max_x = max(1.0, max((float(box[:, 0].max()) for box in boxes), default=1.0))
        max_y = max(1.0, max((float(box[:, 1].max()) for box in boxes), default=1.0))
        for ordinal, item in enumerate(items):
            box = V8.item_box(item)
            if box is None:
                continue
            text = core.output_text(item.get("text", ""))
            normalized = V8.clean(text)
            if not normalized:
                continue
            x0, y0 = box.min(axis=0); x1, y1 = box.max(axis=0)
            letters = sum(character.isalpha() for character in normalized)
            digits = sum(character.isdigit() for character in normalized)
            rows.append({
                "filename": document.filename, "address_gt": document.address_gt,
                "has_ground_truth": bool(document.has_ground_truth), "address_present_gt": bool(document.address_present_gt),
                "document_family": document.document_family, "quality_group": document.quality_group,
                "ordinal": ordinal, "text": text, "ocr_confidence": float(item.get("confidence", 0.0)),
                "x": float((x0 + x1) / 2 / max_x), "y": float((y0 + y1) / 2 / max_y),
                "width": float((x1 - x0) / max_x), "height": float((y1 - y0) / max_y),
                "alpha_ratio": letters / max(1, letters + digits), "digit_ratio": digits / max(1, letters + digits),
                "street_cue": float(bool(V8.STREET_PATTERN.search(normalized))),
                "metadata": float(bool(V8.METADATA_PATTERN.search(normalized))),
                "identifier": float(bool(V8.ID_PATTERN.search(normalized))),
                "date_like": float(bool(core.normalize_date(normalized))),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No OCR lines found for span ranking")
    return frame


def span_similarity(target: str, candidate: str) -> float:
    """Same normalized Levenshtein similarity as core.text_metrics, in C."""
    truth = core.metric_text(target)
    prediction = core.metric_text(candidate)
    distance = Levenshtein.distance(truth, prediction)
    return float(max(0.0, 1.0 - distance / max(1, len(truth), len(prediction))))


def span_rows(base: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Enumerate compact OCR-line spans without consulting labels."""
    document_info = base.set_index("filename", drop=False)
    output: list[dict[str, Any]] = []
    for filename, all_lines in lines.groupby("filename", sort=False):
        if filename not in document_info.index:
            continue
        document = document_info.loc[filename]
        ordered = all_lines.sort_values(["y", "x", "ordinal"]).reset_index(drop=True)
        valid = ordered.loc[(ordered.metadata < 0.5) & (ordered.identifier < 0.5) & (ordered.date_like < 0.5)].copy().reset_index(drop=True)
        for start in range(len(valid)):
            for end in range(start, min(len(valid), start + MAX_SPAN_LINES)):
                part = valid.iloc[start:end + 1]
                if len(part) > 1 and part.y.diff().dropna().max() > 0.18:
                    continue
                text = core.output_text(", ".join(part.text.tolist()))
                letters = sum(character.isalpha() for character in V8.clean(text))
                digits = sum(character.isdigit() for character in V8.clean(text))
                output.append({
                    "filename": filename,
                    "address_gt": document.address_gt,
                    "has_ground_truth": bool(document.has_ground_truth),
                    "address_present_gt": bool(document.address_present_gt),
                    "document_family": document.document_family,
                    "quality_group": document.quality_group,
                    "span_text": text,
                    "line_ids": [int(value) for value in part.ordinal.tolist()],
                    "line_count": int(len(part)),
                    "first_ordinal": int(part.ordinal.min()),
                    "last_ordinal": int(part.ordinal.max()),
                    "first_y": float(part.y.min()), "last_y": float(part.y.max()),
                    "mean_y": float(part.y.mean()), "mean_x": float(part.x.mean()),
                    "min_x": float(part.x.min()), "max_x": float(part.x.max()),
                    "mean_width": float(part.width.mean()), "mean_height": float(part.height.mean()),
                    "vertical_span": float(part.y.max() - part.y.min()),
                    "mean_confidence": float(part.ocr_confidence.mean()),
                    "min_confidence": float(part.ocr_confidence.min()),
                    "max_confidence": float(part.ocr_confidence.max()),
                    "text_length": int(len(V8.clean(text))),
                    "alpha_ratio": float(letters / max(1, letters + digits)),
                    "digit_ratio": float(digits / max(1, letters + digits)),
                    "street_cue_count": float(part.street_cue.sum()),
                    "has_street_cue": float(part.street_cue.max()),
                })
    spans = pd.DataFrame(output)
    if spans.empty:
        raise RuntimeError("No OCR spans could be constructed")
    labelled = spans.has_ground_truth & spans.address_present_gt
    spans["oracle_similarity"] = 0.0
    for filename, positions in spans.loc[labelled].groupby("filename", sort=False).groups.items():
        target = spans.loc[positions[0], "address_gt"]
        spans.loc[positions, "oracle_similarity"] = [span_similarity(target, text) for text in spans.loc[positions, "span_text"]]
    spans["oracle_target"] = 0
    for filename, positions in spans.loc[labelled].groupby("filename", sort=False).groups.items():
        candidates = spans.loc[positions].copy()
        # The shorter candidate is preferred when text similarity ties: it
        # prevents a tied metadata tail becoming an artificial positive.
        best = candidates.sort_values(["oracle_similarity", "line_count", "text_length"], ascending=[False, True, True]).index[0]
        spans.at[best, "oracle_target"] = 1
    spans["document_oracle_similarity"] = spans.groupby("filename")["oracle_similarity"].transform("max")
    return spans


NUMERIC_COLUMNS = (
    "line_count", "first_ordinal", "last_ordinal", "first_y", "last_y", "mean_y", "mean_x", "min_x", "max_x",
    "mean_width", "mean_height", "vertical_span", "mean_confidence", "min_confidence", "max_confidence",
    "text_length", "alpha_ratio", "digit_ratio", "street_cue_count", "has_street_cue",
)


class SpanRanker:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=6000, sublinear_tf=True)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.model = LogisticRegression(C=0.55, class_weight="balanced", max_iter=3000, random_state=SEED)

    def _features(self, frame: pd.DataFrame, fit: bool) -> sparse.csr_matrix:
        text = self.vectorizer.fit_transform(frame.span_text) if fit else self.vectorizer.transform(frame.span_text)
        values = frame.loc[:, NUMERIC_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        numeric = self.scaler.fit_transform(values) if fit else self.scaler.transform(values)
        categories = frame[["document_family", "quality_group"]].fillna("UNKNOWN").astype(str)
        categorical = self.encoder.fit_transform(categories) if fit else self.encoder.transform(categories)
        return sparse.hstack([text, sparse.csr_matrix(numeric), categorical], format="csr")

    def fit(self, frame: pd.DataFrame) -> "SpanRanker":
        eligible = frame.loc[
            frame.document_family.eq(FAMILY)
            & frame.has_ground_truth.astype(bool)
            & frame.address_present_gt.astype(bool)
            & frame.document_oracle_similarity.ge(MIN_ORACLE_SIMILARITY)
        ].copy()
        if eligible.oracle_target.nunique() < 2:
            raise RuntimeError("The ranker needs both positive and negative span candidates")
        self.model.fit(self._features(eligible, fit=True), eligible.oracle_target.astype(int))
        return self

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._features(frame, fit=False))[:, 1]


def mykad_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    subset = frame.loc[frame.has_ground_truth.astype(bool) & frame.document_family.eq(FAMILY)].copy()
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None, "mean_cer": None}
    return {
        "images": int(len(subset)),
        "exact_match": float(subset.address_exact.mean()),
        "mean_similarity": float(subset.address_similarity.mean()),
        "mean_cer": float(subset.address_cer.mean()),
    }


def build_predictions(base: pd.DataFrame, spans: pd.DataFrame, model: SpanRanker, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = spans.copy()
    scored["probability"] = model.probability(scored)
    updates: list[dict[str, Any]] = []
    for filename, options in scored.loc[scored.document_family.eq(FAMILY)].groupby("filename", sort=False):
        best = options.sort_values(["probability", "has_street_cue", "line_count"], ascending=[False, False, True]).iloc[0]
        if float(best.probability) < threshold:
            continue
        updates.append({
            "filename": filename, "span_address": best.span_text, "span_probability": float(best.probability),
            "span_line_ids": "|".join(map(str, best.line_ids)), "span_line_count": int(best.line_count),
            "span_has_street_cue": bool(best.has_street_cue),
        })
    output = base.copy()
    update_frame = pd.DataFrame(updates)
    if not update_frame.empty:
        output = output.merge(update_frame, on="filename", how="left", validate="one_to_one")
        changed = output.span_address.notna() & output.span_address.map(V8.clean).ne(output.address_pred.map(V8.clean))
        output.loc[changed, "address_pred"] = output.loc[changed, "span_address"]
        decisions = output.loc[changed, ["filename", "data_split", "address_pred", "span_probability", "span_line_ids", "span_line_count", "span_has_street_cue"]].rename(columns={"address_pred": "selected_address"})
    else:
        decisions = pd.DataFrame(columns=["filename", "data_split", "selected_address", "span_probability", "span_line_ids", "span_line_count", "span_has_street_cue"])
    return core.score_prediction_rows(output), decisions


def pick_threshold(base: pd.DataFrame, spans: pd.DataFrame, train_files: set[str], groups: dict[str, str]) -> tuple[float, dict[str, Any]]:
    documents = sorted(train_files)
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    fit_positions, calibration_positions = next(split.split(documents, groups=[groups[name] for name in documents]))
    fit_files = {documents[position] for position in fit_positions}
    calibration_files = {documents[position] for position in calibration_positions}
    model = SpanRanker().fit(spans.loc[spans.filename.isin(fit_files)])
    calibration_base = base.loc[base.filename.isin(calibration_files)].copy()
    baseline = mykad_metrics(calibration_base)
    options: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        candidate, decisions = build_predictions(calibration_base, spans.loc[spans.filename.isin(calibration_files)], model, float(threshold))
        metrics = mykad_metrics(candidate)
        safe = metrics["exact_match"] >= baseline["exact_match"] and metrics["mean_similarity"] >= baseline["mean_similarity"] + 0.03
        options.append({"threshold": float(threshold), "safe": bool(safe), "changes": int(len(decisions)), **metrics})
    safe_options = [option for option in options if option["safe"]]
    selected = max(safe_options, key=lambda option: (option["mean_similarity"], option["exact_match"], -option["changes"])) if safe_options else max(options, key=lambda option: (option["mean_similarity"], option["exact_match"], -option["changes"]))
    return float(selected["threshold"]), {"calibration_images": len(calibration_files), "baseline": baseline, "selected": selected, "options": options}


def run() -> None:
    base = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda value: value.map(V3.natural_key)).reset_index(drop=True)
    # Generic/MRZ cards have no address-positive training examples and this
    # experiment never changes them.  Excluding them here reduces span/TF-IDF
    # work substantially while preserving the full-frame metric afterwards.
    mykad_base = base.loc[base.document_family.eq(FAMILY)].copy()
    print(json.dumps({"stage": "load_mykad", "images": int(len(mykad_base))}, ensure_ascii=False), flush=True)
    lines = flatten_lines_fast(mykad_base)
    print(json.dumps({"stage": "flatten_lines", "lines": int(len(lines))}, ensure_ascii=False), flush=True)
    spans = span_rows(base, lines)
    print(json.dumps({"stage": "enumerate_spans", "spans": int(len(spans))}, ensure_ascii=False), flush=True)
    labelled = base.loc[base.has_ground_truth.astype(bool) & base.document_family.eq(FAMILY) & base.address_present_gt.astype(bool)].copy()
    groups = {row.filename: V3.identity_key(row) for _, row in labelled.iterrows()}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_positions, validation_positions = next(splitter.split(labelled, groups=[groups[name] for name in labelled.filename]))
    train_files = set(labelled.iloc[train_positions].filename)
    validation_files = set(labelled.iloc[validation_positions].filename)
    threshold, calibration = pick_threshold(base, spans, train_files, groups)
    validation_model = SpanRanker().fit(spans.loc[spans.filename.isin(train_files)])
    validation_base = base.loc[base.filename.isin(validation_files)].copy()
    validation_candidate, validation_decisions = build_predictions(validation_base, spans.loc[spans.filename.isin(validation_files)], validation_model, threshold)
    baseline_metrics = mykad_metrics(validation_base)
    candidate_metrics = mykad_metrics(validation_candidate)
    # A threshold must be safe both when calibrated and on the untouched outer
    # split.  Passing only one split is not enough evidence to replace output.
    release = bool(calibration["selected"]["safe"]) and candidate_metrics["exact_match"] >= baseline_metrics["exact_match"] and candidate_metrics["mean_similarity"] >= baseline_metrics["mean_similarity"] + 0.03
    validation = {
        "family": FAMILY, "train_images": len(train_files), "validation_images": len(validation_files),
        "oracle_quality_floor": MIN_ORACLE_SIMILARITY, "selected_probability_threshold": threshold,
        "calibration": calibration, "baseline": baseline_metrics, "candidate": candidate_metrics,
        "candidate_changes": int(len(validation_decisions)), "test_release_gate_passed": bool(release),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    final_model = SpanRanker().fit(spans)
    candidate, decisions = build_predictions(base, spans, final_model, threshold)
    released = candidate if release else core.score_prediction_rows(base.copy())
    atomic_parquet(RESULTS / "predictions_candidate.parquet", candidate)
    atomic_csv(RESULTS / "predictions_candidate.csv", candidate.drop(columns=[column for column in candidate if column.endswith("_detail")], errors="ignore"))
    atomic_parquet(RESULTS / "predictions_released.parquet", released)
    atomic_csv(RESULTS / "predictions_released.csv", released.drop(columns=[column for column in released if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "span_decisions.csv", decisions)
    atomic_csv(RESULTS / "span_candidates.csv", spans)
    atomic_json(RESULTS / "validation.json", validation)
    joblib.dump({"model": final_model, "threshold": threshold, "numeric_columns": NUMERIC_COLUMNS}, RESULTS / "mykad_address_span_ranker.joblib")
    metrics = V3.field_metrics(released)
    metrics.update({
        "candidate_changed_addresses": int(len(decisions)), "test_release_gate_passed": bool(release), "official_submission_overwritten": False,
        **{f"released_mykad_{key}": value for key, value in mykad_metrics(released).items()},
        **{f"candidate_mykad_{key}": value for key, value in mykad_metrics(candidate).items()},
    })
    atomic_json(RESULTS / "metrics.json", metrics)
    submission = released.loc[released.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, got {len(submission)}")
    atomic_csv(RESULTS / "test_predictions.csv", submission)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V10 supervised MyKad OCR-span ranker", "ocr_rerun": False, "gpu_required": False, "official_results_untouched": True}, ensure_ascii=False, indent=2), flush=True)
    run()
