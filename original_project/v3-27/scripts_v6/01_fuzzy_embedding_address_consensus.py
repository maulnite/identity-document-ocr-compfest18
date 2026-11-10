#!/usr/bin/env python3
"""V6: soft identity graph + fuzzy address consensus.

This experiment keeps the completed V3 OCR predictions immutable and does not
write ``results/test_predictions.csv``.  It combines the already computed
OCR, DINOv2, Siamese, and optional ArcFace face embeddings into a supervised
pair-probability model.  The probability is an *edge weight*, not a forced
cluster assignment.  Address votes are then pooled from a document's
high-confidence neighbours with fuzzy text agreement.

Ground-truth identities are used only to fit and validate the pair linker on
labeled rows.  Ground-truth field values are never used as features or votes.
Train/labeled and test documents are always kept in separate graphs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
BASE_RESULTS = PROJECT / "results_v3_stable"
V4_RESULTS = PROJECT / "results_v4_visual_linker"
V5_RESULTS = PROJECT / "results_v5_face"
RESULTS = PROJECT / "results_v6_fuzzy_embedding"

SEED = 1808
PAIR_PRECISION_TARGET = 0.985
# A pair must pass this calibrated probability guard before it can vote.  It is
# deliberately still a soft graph: all retained edges retain their probability
# as a continuous vote weight instead of becoming a hard cluster membership.
MIN_ADDRESS_AGREEMENT = 90.0
MIN_NEIGHBOUR_SUPPORT = 2
MIN_WINNING_SHARE = 0.72
MIN_WINNING_MARGIN = 0.18


def load_v3() -> Any:
    spec = importlib.util.spec_from_file_location("v3_fuzzy_support", V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V3 helper: {V3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_v3()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=V3.core.json_default), encoding="utf-8")
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


def readable_name(value: Any) -> bool:
    return len(clean(value)) >= 6


def valid_date(value: Any) -> bool:
    return V3.valid_date(value)


def pair_is_compatible(left: pd.Series, right: pd.Series) -> bool:
    """Reject only strong contradictory evidence; missing/noisy OCR is allowed."""
    left_dob, right_dob = clean(left.birth_date_pred), clean(right.birth_date_pred)
    if valid_date(left_dob) and valid_date(right_dob) and left_dob != right_dob:
        return False
    left_name, right_name = clean(left.name_pred), clean(right.name_pred)
    if readable_name(left_name) and readable_name(right_name) and fuzz.ratio(left_name, right_name) < 65:
        return False
    return True


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    base_path = BASE_RESULTS / "predictions_base.parquet"
    paths = {
        "v3_link": BASE_RESULTS / "siamese_embeddings.npy",
        "document": V4_RESULTS / "visual_embeddings.npy",
        "v4_link": V4_RESULTS / "visual_fusion_embeddings.npy",
        "v5_link": V5_RESULTS / "face_fusion_embeddings.npy",
        "face": V5_RESULTS / "face_embeddings.npy",
    }
    face_manifest_path = V5_RESULTS / "face_detection_manifest.csv"
    missing = [str(path) for path in [base_path, face_manifest_path, *paths.values()] if not path.exists()]
    if missing:
        raise RuntimeError("V6 requires completed V3/V4/V5 artifacts; missing: " + ", ".join(missing))
    frame = pd.read_parquet(base_path).sort_values("filename", key=lambda column: column.map(V3.natural_key)).reset_index(drop=True)
    embeddings = {name: np.load(path).astype(np.float32) for name, path in paths.items()}
    for name, values in embeddings.items():
        if values.ndim != 2 or values.shape[0] != len(frame):
            raise RuntimeError(f"{name} embedding shape {values.shape} does not align with {len(frame)} V3 rows")
        norms = np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
        embeddings[name] = values / norms
    face_manifest = pd.read_csv(face_manifest_path).sort_values("filename", key=lambda column: column.map(V3.natural_key)).reset_index(drop=True)
    if len(face_manifest) != len(frame) or not face_manifest.filename.equals(frame.filename):
        raise RuntimeError("V5 face manifest does not align with V3 rows")
    face_available = face_manifest.face_available.astype(bool).to_numpy()
    return frame, embeddings, face_available


FEATURE_NAMES = (
    "v3_link_cosine", "v4_link_cosine", "v5_link_cosine", "document_cosine",
    "face_cosine", "face_available_both", "name_fuzzy_ratio", "address_fuzzy_ratio",
    "same_valid_dob", "conflicting_valid_dob", "same_document_family",
    "confidence_min", "confidence_mean",
)


def field_ratio(left: Any, right: Any) -> float:
    a, b = clean(left), clean(right)
    return float(fuzz.ratio(a, b) / 100.0) if a and b else 0.0


def pair_feature_matrix(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    face_available: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Produce features from predictions/embeddings only; never ground truth."""
    if not len(pairs):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    left, right = pairs[:, 0], pairs[:, 1]
    cosine = lambda name: (embeddings[name][left] * embeddings[name][right]).sum(axis=1)
    both_face = face_available[left] & face_available[right]
    face_cos = cosine("face")
    face_cos[~both_face] = 0.0
    name = np.asarray([field_ratio(frame.at[i, "name_pred"], frame.at[j, "name_pred"]) for i, j in pairs], dtype=np.float32)
    address = np.asarray([field_ratio(frame.at[i, "address_pred"], frame.at[j, "address_pred"]) for i, j in pairs], dtype=np.float32)
    left_dob = frame.birth_date_pred.iloc[left].map(clean).to_numpy()
    right_dob = frame.birth_date_pred.iloc[right].map(clean).to_numpy()
    left_valid = np.asarray([valid_date(value) for value in left_dob])
    right_valid = np.asarray([valid_date(value) for value in right_dob])
    same_dob = left_valid & right_valid & (left_dob == right_dob)
    conflict_dob = left_valid & right_valid & (left_dob != right_dob)
    family = frame.document_family.fillna("UNROUTED").astype(str).to_numpy()
    confidence = pd.to_numeric(frame.ocr_confidence, errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float32)
    return np.column_stack([
        cosine("v3_link"), cosine("v4_link"), cosine("v5_link"), cosine("document"),
        face_cos, both_face.astype(np.float32), name, address, same_dob.astype(np.float32),
        conflict_dob.astype(np.float32), (family[left] == family[right]).astype(np.float32),
        np.minimum(confidence[left], confidence[right]), (confidence[left] + confidence[right]) / 2.0,
    ]).astype(np.float32)


def all_pairs(indices: Iterable[int]) -> np.ndarray:
    values = np.asarray(list(indices), dtype=np.int64)
    if len(values) < 2:
        return np.empty((0, 2), dtype=np.int64)
    first, second = np.triu_indices(len(values), k=1)
    return np.column_stack([values[first], values[second]]).astype(np.int64)


def labels_for_pairs(pairs: np.ndarray, groups: list[str]) -> np.ndarray:
    return np.asarray([groups[left] == groups[right] for left, right in pairs], dtype=np.int8)


def fit_pair_model(features: np.ndarray, labels: np.ndarray) -> Pipeline:
    if labels.min() == labels.max():
        raise RuntimeError("The training split needs both positive and negative identity pairs")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.35, class_weight="balanced", max_iter=2500, random_state=SEED)),
    ])
    model.fit(features, labels)
    return model


def calibrate_probability(probability: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Calibrate a high-precision edge guard on held-out identities."""
    choices: list[tuple[bool, float, float, float, int, int]] = []
    for threshold in np.unique(np.quantile(probability, np.linspace(0.50, 0.9998, 500))):
        predicted = probability >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum((~predicted) & (labels == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        choices.append((precision >= PAIR_PRECISION_TARGET, recall, precision, float(threshold), tp, fp))
    safe = [choice for choice in choices if choice[0]]
    _, recall, precision, threshold, tp, fp = max(safe, key=lambda row: (row[1], row[2])) if safe else max(choices, key=lambda row: (row[2], row[1]))
    return threshold, {
        "edge_probability_threshold": threshold,
        "pair_precision": precision,
        "pair_recall": recall,
        "true_positive_edges": tp,
        "false_positive_edges": fp,
        "precision_target": PAIR_PRECISION_TARGET,
        "precision_target_met": bool(safe),
        "pair_auc": float(roc_auc_score(labels, probability)),
        "pair_average_precision": float(average_precision_score(labels, probability)),
    }


def quality_weight(row: pd.Series) -> float:
    return max(0.15, float(pd.to_numeric(row.get("ocr_confidence", 0.0), errors="coerce") or 0.0))


def address_vote(
    frame: pd.DataFrame,
    target: int,
    neighbour_edges: list[tuple[int, float]],
) -> tuple[str, dict[str, Any]]:
    """Weighted fuzzy vote for one address; returns an empty string when unsure."""
    entries: list[dict[str, Any]] = []
    own = frame.loc[target]
    own_value, own_normalized = V3.core.output_text(own.address_pred), clean(own.address_pred)
    if own_normalized:
        entries.append({"index": target, "value": own_value, "normalized": own_normalized, "weight": 0.35 * quality_weight(own), "probability": 1.0, "is_self": True})
    for neighbour, probability in neighbour_edges:
        row = frame.loc[neighbour]
        value, normalized = V3.core.output_text(row.address_pred), clean(row.address_pred)
        if not normalized:
            continue
        entries.append({
            "index": neighbour, "value": value, "normalized": normalized,
            "weight": (float(probability) ** 1.5) * quality_weight(row),
            "probability": float(probability), "is_self": False,
        })
    neighbour_entries = [entry for entry in entries if not entry["is_self"]]
    if not neighbour_entries:
        return "", {"reason": "no_address_from_safe_neighbour"}
    candidates: list[dict[str, Any]] = []
    for seed in entries:
        support = [entry for entry in entries if fuzz.ratio(seed["normalized"], entry["normalized"]) >= MIN_ADDRESS_AGREEMENT]
        neighbour_support = [entry for entry in support if not entry["is_self"]]
        if not neighbour_support:
            continue
        weight = sum(entry["weight"] for entry in support)
        # Prefer a source with a confident OCR reading when fuzzy groups tie.
        representative = max(support, key=lambda entry: (entry["weight"], len(entry["normalized"])))
        candidates.append({
            "value": representative["value"], "normalized": representative["normalized"],
            "weight": float(weight), "neighbour_count": len({entry["index"] for entry in neighbour_support}),
            "mean_probability": float(np.mean([entry["probability"] for entry in neighbour_support])),
        })
    if not candidates:
        return "", {"reason": "no_fuzzy_address_group"}
    candidates.sort(key=lambda item: (item["weight"], item["neighbour_count"], len(item["normalized"])), reverse=True)
    best = candidates[0]
    total_weight = sum(entry["weight"] for entry in entries)
    second_weight = candidates[1]["weight"] if len(candidates) > 1 else 0.0
    share = best["weight"] / max(1e-12, total_weight)
    margin = (best["weight"] - second_weight) / max(1e-12, total_weight)
    audit = {
        "candidate_weight": best["weight"], "candidate_neighbour_count": best["neighbour_count"],
        "candidate_mean_edge_probability": best["mean_probability"], "winning_share": share,
        "winning_margin": margin,
    }
    if clean(best["value"]) == own_normalized:
        return "", {"reason": "base_address_already_wins", **audit}
    # One matching neighbour is only a duplicate reading, not group-level
    # evidence.  Requiring two separate documents prevents a single bad OCR
    # field from being copied into an otherwise blank/uncertain address.
    if best["neighbour_count"] < MIN_NEIGHBOUR_SUPPORT:
        return "", {"reason": "insufficient_independent_neighbours", **audit}
    if share < MIN_WINNING_SHARE or margin < MIN_WINNING_MARGIN:
        return "", {"reason": "ambiguous_soft_vote", **audit}
    return best["value"], {"reason": "accepted", **audit}


def soft_address_consensus(
    frame: pd.DataFrame,
    pairs: np.ndarray,
    probability: np.ndarray,
    edge_threshold: float,
    allowed: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply per-document soft consensus.  No graph clustering/union occurs."""
    output = frame.copy()
    neighbours: dict[int, list[tuple[int, float]]] = {index: [] for index in frame.index}
    edge_rows: list[dict[str, Any]] = []
    for (left, right), score in zip(pairs, probability):
        if score < edge_threshold:
            continue
        if allowed is not None and (int(left) not in allowed or int(right) not in allowed):
            continue
        if output.at[left, "data_split"] != output.at[right, "data_split"]:
            continue
        if not pair_is_compatible(output.loc[left], output.loc[right]):
            continue
        neighbours[int(left)].append((int(right), float(score)))
        neighbours[int(right)].append((int(left), float(score)))
        edge_rows.append({
            "filename": output.at[left, "filename"], "neighbour_filename": output.at[right, "filename"],
            "data_split": output.at[left, "data_split"], "identity_probability": float(score),
        })
    decisions: list[dict[str, Any]] = []
    targets = sorted(allowed) if allowed is not None else list(output.index)
    for target in targets:
        value, audit = address_vote(output, target, neighbours[target])
        if not value:
            continue
        previous = V3.core.output_text(output.at[target, "address_pred"])
        output.at[target, "address_pred"] = value
        decisions.append({
            "filename": output.at[target, "filename"], "data_split": output.at[target, "data_split"],
            "previous_address": previous, "fuzzy_consensus_address": value,
            "safe_neighbour_count": len(neighbours[target]), **audit,
        })
    return V3.core.score_prediction_rows(output), pd.DataFrame(decisions), pd.DataFrame(edge_rows)


def address_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    labeled = frame.loc[frame.has_ground_truth.astype(bool)].copy()
    present = labeled.address_gt.map(clean).ne("")
    return {
        "labeled_rows": int(len(labeled)),
        "address_exact_match": float(labeled.address_exact.mean()),
        "address_present_rows": int(present.sum()),
        "address_present_exact_match": float(labeled.loc[present, "address_exact"].mean()) if present.any() else None,
        "address_present_mean_edit_similarity": float(labeled.loc[present, "address_edit_similarity"].mean()) if present.any() and "address_edit_similarity" in labeled else None,
    }


def run() -> pd.DataFrame:
    frame, embeddings, face_available = load_inputs()
    labeled = frame.has_ground_truth.astype(bool).to_numpy()
    labeled_indices = np.flatnonzero(labeled).tolist()
    groups = [V3.identity_key(row) if bool(row.has_ground_truth) else f"__unlabeled__{row.filename}" for _, row in frame.iterrows()]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_local, validation_local = next(splitter.split(labeled_indices, groups=[groups[index] for index in labeled_indices]))
    train_indices = [labeled_indices[index] for index in train_local]
    validation_indices = [labeled_indices[index] for index in validation_local]

    train_pairs = all_pairs(train_indices)
    validation_pairs = all_pairs(validation_indices)
    train_labels = labels_for_pairs(train_pairs, groups)
    validation_labels = labels_for_pairs(validation_pairs, groups)
    validation_model = fit_pair_model(pair_feature_matrix(frame, embeddings, face_available, train_pairs), train_labels)
    validation_probability = validation_model.predict_proba(pair_feature_matrix(frame, embeddings, face_available, validation_pairs))[:, 1]
    edge_threshold, validation = calibrate_probability(validation_probability, validation_labels)
    validation.update({
        "train_images": len(train_indices), "validation_images": len(validation_indices),
        "train_identity_groups": len({groups[index] for index in train_indices}),
        "validation_identity_groups": len({groups[index] for index in validation_indices}),
        "train_pairs": int(len(train_pairs)), "validation_pairs": int(len(validation_pairs)),
        "face_available_total": int(face_available.sum()),
        "method": "soft supervised pair graph; no hard identity clusters",
    })
    validation_output, validation_decisions, _ = soft_address_consensus(
        frame, validation_pairs, validation_probability, edge_threshold, allowed=set(validation_indices)
    )
    baseline_validation = frame.loc[validation_indices].copy()
    baseline_address = address_metrics(baseline_validation)
    fuzzy_address = address_metrics(validation_output.loc[validation_indices])
    # A test prediction may be released only if a disjoint set of identities
    # shows a strict address-EM improvement.  This makes the experiment safe to
    # keep in the project even when a particular dataset lacks enough repeated,
    # readable address fields to benefit from consensus.
    release_to_test = fuzzy_address["address_exact_match"] > baseline_address["address_exact_match"]
    validation["address_validation_baseline"] = baseline_address
    validation["address_validation_fuzzy_candidate"] = fuzzy_address
    validation["address_validation_changed_rows"] = int(len(validation_decisions))
    validation["test_release_gate_passed"] = bool(release_to_test)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)

    final_pairs = all_pairs(labeled_indices)
    final_labels = labels_for_pairs(final_pairs, groups)
    final_model = fit_pair_model(pair_feature_matrix(frame, embeddings, face_available, final_pairs), final_labels)
    # Score all intra-split pairs.  This includes the held-out test split, but
    # only learned parameters and predictions/embeddings enter its features.
    split_pairs = np.vstack([all_pairs(part.index) for _, part in frame.groupby("data_split", dropna=False)])
    split_probability = final_model.predict_proba(pair_feature_matrix(frame, embeddings, face_available, split_pairs))[:, 1]
    candidate_output, decisions, edges = soft_address_consensus(frame, split_pairs, split_probability, edge_threshold)
    atomic_parquet(RESULTS / "predictions_fuzzy_candidate.parquet", candidate_output)
    atomic_csv(RESULTS / "predictions_fuzzy_candidate.csv", candidate_output.drop(columns=[column for column in candidate_output if column.endswith("_detail")], errors="ignore"))
    output = candidate_output if release_to_test else V3.core.score_prediction_rows(frame.copy())
    atomic_parquet(RESULTS / "predictions_fuzzy_address_consensus.parquet", output)
    atomic_csv(RESULTS / "predictions_fuzzy_address_consensus.csv", output.drop(columns=[column for column in output if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "fuzzy_address_decisions.csv", decisions)
    atomic_csv(RESULTS / "soft_identity_edges.csv", edges)
    joblib.dump({"feature_names": FEATURE_NAMES, "model": final_model, "edge_threshold": edge_threshold}, RESULTS / "soft_pair_linker.joblib")
    metrics = V3.field_metrics(output)
    metrics.update({
        "address_only_consensus": True, "candidate_changed_addresses": int(len(decisions)), "safe_soft_edges": int(len(edges)),
        "test_release_gate_passed": bool(release_to_test), "official_submission_overwritten": False,
        **{f"address_{key}": value for key, value in address_metrics(output).items()},
        **{f"candidate_address_{key}": value for key, value in address_metrics(candidate_output).items()},
    })
    atomic_json(RESULTS / "metrics_fuzzy_address_consensus.json", metrics)
    atomic_json(RESULTS / "linker_validation.json", validation)
    submission = output.loc[output.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, received {len(submission)}")
    atomic_csv(RESULTS / "test_predictions_fuzzy_address_consensus.csv", submission)
    return output


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V6 DINO/face/OCR embeddings -> calibrated soft pair graph -> fuzzy address consensus",
        "hard_cluster_assignment": False,
        "address_only": True,
        "official_results_untouched": True,
    }, ensure_ascii=False, indent=2), flush=True)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
