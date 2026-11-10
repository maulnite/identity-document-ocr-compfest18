#!/usr/bin/env python3
"""V4: vision-first identity grouping with DINOv2 + strict fusion verification.

This experiment consumes V3's OCR-only base predictions.  It never modifies
``results/test_predictions.csv``.  A DINOv2 visual encoder (a vision model, not
a VLM) produces one embedding per rectified document.  Visual nearest
neighbours are only *candidates*; a Siamese model trained on labelled identity
pairs combines the visual embedding with OCR/field features, calibrates a
high-precision threshold on held-out identities, and forms complete-linkage
clusters.  Consensus is applied only inside those strict clusters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
BASE_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v4_visual_linker"
CACHE = PROJECT / "cache_v4_dinov2"

SEED = 1808
VISION_REPOSITORY = "facebookresearch/dinov2"
VISION_MODEL = "dinov2_vits14"
VISION_MODEL_LABEL = "DINOv2 ViT-S/14 (self-supervised vision encoder; not a VLM)"
IMAGE_SIZE = 448
BATCH_SIZE = 24
CANDIDATE_POSITIVE_RECALL_TARGET = 0.995
PAIR_PRECISION_TARGET = 0.985
EMBED_DIM = 64
IMAGE_FEATURE_WEIGHT = 0.70


def load_v3() -> Any:
    if not V3_PATH.exists():
        raise RuntimeError(f"V3 helper script is missing: {V3_PATH}")
    spec = importlib.util.spec_from_file_location("v3_visual_support", V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load V3 helper functions")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_v3()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def natural_key(value: str) -> list[Any]:
    return V3.natural_key(value)


def identity_key(row: pd.Series) -> str:
    return V3.identity_key(row)


def cache_path(filename: str, input_path: str) -> Path:
    signature = f"{filename}|{input_path}|{VISION_MODEL}|{IMAGE_SIZE}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return CACHE / "embeddings" / f"{Path(filename).stem}__{digest}.npy"


def prepare_image(path: Path) -> np.ndarray:
    """Letterbox a rectified document without distorting the card layout."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read visual-linker input: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / max(1, height), IMAGE_SIZE / max(1, width))
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 127, dtype=np.uint8)
    y0 = (IMAGE_SIZE - resized.shape[0]) // 2
    x0 = (IMAGE_SIZE - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor -= np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    tensor /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return tensor


class DINOv2Encoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        started = time.perf_counter()
        # torch.hub pins the official DINOv2 architecture and downloads its
        # public pretrained checkpoint once into the account cache.
        self.model = torch.hub.load(VISION_REPOSITORY, VISION_MODEL, pretrained=True, trust_repo=True).to(self.device).eval()
        self.initialization_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def encode(self, batch: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(batch).to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            values = self.model(inputs)
        values = torch.nn.functional.normalize(values.float(), p=2, dim=1)
        return values.cpu().numpy().astype(np.float32)

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def resolve_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # V3's adaptive table preserves the chosen path as ``pipeline_input_path``;
    # per-method OCR tables use ``input_path``.  Normalize both representations
    # for visual embedding without rerunning completed OCR.
    if "input_path" not in result:
        if "pipeline_input_path" not in result:
            raise RuntimeError("V3 base predictions lack an OCR input path; rerun V3 OCR first")
        result["input_path"] = result["pipeline_input_path"]
    missing = result.input_path.map(lambda value: not Path(str(value)).exists())
    if missing.any():
        missing_names = result.loc[missing, "filename"].tolist()[:5]
        raise RuntimeError(f"{int(missing.sum())} visual inputs are missing, e.g. {missing_names}")
    return result


def build_visual_embeddings(frame: pd.DataFrame, overwrite: bool) -> np.ndarray:
    """Compute all document embeddings once, committing one cache file per image."""
    frame = resolve_inputs(frame).sort_values("filename", key=lambda column: column.map(natural_key)).reset_index(drop=True)
    dimensions: int | None = None
    vectors: list[np.ndarray | None] = [None] * len(frame)
    pending: list[tuple[int, Path]] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        target = cache_path(row.filename, str(row.input_path))
        if target.exists() and not overwrite:
            try:
                cached = np.load(target)
                if cached.ndim == 1 and cached.size > 0 and np.isfinite(cached).all():
                    vectors[position] = cached.astype(np.float32)
                    dimensions = int(cached.size)
                    continue
            except Exception:
                pass
        pending.append((position, Path(row.input_path)))
    if pending:
        encoder = DINOv2Encoder()
        print(json.dumps({
            "vision_encoder": VISION_MODEL_LABEL,
            "device": str(encoder.device),
            "initialization_seconds": round(encoder.initialization_seconds, 2),
            "new_images": len(pending), "cached_images": len(frame) - len(pending),
        }), flush=True)
        try:
            for first in range(0, len(pending), BATCH_SIZE):
                chunk = pending[first:first + BATCH_SIZE]
                batch = np.stack([prepare_image(path) for _, path in chunk])
                encoded = encoder.encode(batch)
                dimensions = int(encoded.shape[1])
                for (position, _), vector in zip(chunk, encoded):
                    row = frame.iloc[position]
                    target = cache_path(str(row.filename), str(row.input_path))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(".tmp.npy")
                    np.save(temp, vector)
                    os.replace(temp, target)
                    vectors[position] = vector
                print(f"visual embedding: {min(first + len(chunk), len(pending))}/{len(pending)} newly committed", flush=True)
        finally:
            encoder.close()
    if dimensions is None or any(vector is None for vector in vectors):
        raise RuntimeError("Visual embedding cache did not produce a complete, valid embedding for every image")
    embeddings = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    np.save(RESULTS / "visual_embeddings.npy", embeddings)
    atomic_json(RESULTS / "visual_embedding_manifest.json", {
        "model": VISION_MODEL_LABEL, "repository": VISION_REPOSITORY, "model_id": VISION_MODEL,
        "image_size": IMAGE_SIZE, "embedding_dimension": int(embeddings.shape[1]),
        "batch_size": BATCH_SIZE, "images": int(len(embeddings)), "device_policy": "cuda when available",
    })
    return embeddings


def combined_similarity(text_retrieval: Any, visual: np.ndarray, indices: list[int]) -> np.ndarray:
    text = (text_retrieval[indices] @ text_retrieval[indices].T).toarray().astype(np.float32)
    image = (visual[indices] @ visual[indices].T).astype(np.float32)
    # Visual similarity proposes candidates; OCR helps expose visually similar
    # cards that belong to different people as hard negatives during training.
    return 0.60 * image + 0.40 * text


def make_pairs(indices: Iterable[int], groups: list[str], text_retrieval: Any, visual: np.ndarray, seed: int, negatives_per_positive: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Same-person pairs plus visually/textually close hard negatives."""
    rng = random.Random(seed)
    available = list(indices)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index in available:
        by_group[groups[index]].append(index)
    positives: list[tuple[int, int]] = []
    for members in by_group.values():
        pairs = list(itertools.combinations(members, 2))
        rng.shuffle(pairs)
        positives.extend(pairs[:100])
    if not positives:
        raise RuntimeError("No repeated labelled identity is available for visual-linker training")
    similarity = combined_similarity(text_retrieval, visual, available)
    local_of = {global_index: local for local, global_index in enumerate(available)}
    negatives: set[tuple[int, int]] = set()
    for left, right in positives:
        for source in (left, right):
            ranking = np.argsort(-similarity[local_of[source]])
            chosen = 0
            for candidate_local in ranking:
                candidate = available[int(candidate_local)]
                if candidate == source or groups[candidate] == groups[source]:
                    continue
                negatives.add(tuple(sorted((source, candidate))))
                chosen += 1
                if chosen >= negatives_per_positive:
                    break
    target = len(positives) * negatives_per_positive
    group_names = list(by_group)
    while len(negatives) < target:
        first, second = rng.sample(group_names, 2)
        negatives.add(tuple(sorted((rng.choice(by_group[first]), rng.choice(by_group[second])))))
    pairs = np.asarray(positives + list(negatives)[:target], dtype=np.int64)
    labels = np.asarray([1] * len(positives) + [0] * min(target, len(negatives)), dtype=np.float32)
    order = list(range(len(labels)))
    rng.shuffle(order)
    return pairs[order], labels[order]


class FusionSiameseLinker(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 384), torch.nn.LayerNorm(384), torch.nn.GELU(), torch.nn.Dropout(0.10),
            torch.nn.Linear(384, 160), torch.nn.GELU(), torch.nn.Dropout(0.05),
            torch.nn.Linear(160, EMBED_DIM),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.encoder(values), p=2, dim=1)


def train_linker(features: np.ndarray, pairs: np.ndarray, labels: np.ndarray, epochs: int, seed: int) -> FusionSiameseLinker:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionSiameseLinker(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    values = torch.from_numpy(features).to(device)
    pair_tensor = torch.from_numpy(pairs).to(device)
    targets = torch.from_numpy(labels).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(labels), generator=generator, device=device)
        losses: list[float] = []
        for batch in order.split(512):
            left, right = model(values[pair_tensor[batch, 0]]), model(values[pair_tensor[batch, 1]])
            cosine = (left * right).sum(dim=1)
            loss = torch.where(targets[batch] > 0.5, (1.0 - cosine).pow(2), torch.relu(cosine - 0.20).pow(2)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % 10 == 0:
            print(f"visual fusion linker epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.5f}", flush=True)
    return model.eval()


@torch.inference_mode()
def embed(model: FusionSiameseLinker, features: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    values = [model(batch.to(device)).cpu().numpy() for batch in torch.from_numpy(features).split(1024)]
    return np.vstack(values).astype(np.float32)


def score_pairs(embeddings: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return (embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]]).sum(axis=1)


def calibrate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, Any]]:
    choices: list[tuple[bool, float, float, float]] = []
    for threshold in np.linspace(0.20, 0.999, 160):
        predicted = scores >= threshold
        tp, fp = int(np.sum(predicted & (labels == 1))), int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum((~predicted) & (labels == 1)))
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        choices.append((precision >= PAIR_PRECISION_TARGET, recall, precision, float(threshold)))
    safe = [choice for choice in choices if choice[0]]
    _, recall, precision, threshold = max(safe, key=lambda value: (value[1], value[2])) if safe else max(choices, key=lambda value: (value[2], value[1]))
    return threshold, {
        "threshold": threshold, "pair_precision": precision, "pair_recall": recall,
        "precision_target": PAIR_PRECISION_TARGET, "precision_target_met": bool(safe),
        "pair_auc": float(roc_auc_score(labels, scores)), "pair_average_precision": float(average_precision_score(labels, scores)),
    }


def positive_visual_scores(visual: np.ndarray, indices: Iterable[int], groups: list[str]) -> np.ndarray:
    """Visual similarity of genuine identity pairs; labels are used only here for calibration."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_group[groups[index]].append(index)
    scores = [float(visual[left] @ visual[right]) for members in by_group.values() for left, right in itertools.combinations(members, 2)]
    if not scores:
        raise RuntimeError("No repeated labelled identities are available to calibrate the visual candidate threshold")
    return np.asarray(scores, dtype=np.float32)


def calibrate_visual_candidate_threshold(visual: np.ndarray, groups: list[str], train_indices: list[int], validation_indices: list[int]) -> tuple[float, dict[str, Any]]:
    """Choose the most selective threshold that retains almost all train positives.

    The held-out positive-pair recall is reported separately.  Test labels are
    never consulted, and this is deliberately a high-recall candidate gate—not
    an identity decision threshold.
    """
    train_scores = positive_visual_scores(visual, train_indices, groups)
    threshold = float(np.quantile(train_scores, 1.0 - CANDIDATE_POSITIVE_RECALL_TARGET))
    validation_scores = positive_visual_scores(visual, validation_indices, groups)
    return threshold, {
        "candidate_threshold": threshold,
        "candidate_positive_recall_target_train": CANDIDATE_POSITIVE_RECALL_TARGET,
        "train_positive_pairs": int(len(train_scores)),
        "train_positive_recall": float(np.mean(train_scores >= threshold)),
        "validation_positive_pairs": int(len(validation_scores)),
        "validation_positive_recall": float(np.mean(validation_scores >= threshold)),
        "train_positive_similarity_min": float(train_scores.min()),
        "train_positive_similarity_median": float(np.median(train_scores)),
    }


def visual_candidate_edges(frame: pd.DataFrame, visual: np.ndarray, threshold: float) -> tuple[set[tuple[int, int]], pd.DataFrame]:
    """Thresholded image-similarity pairs become candidates, never automatic groups."""
    edges: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    labels_known = frame.has_ground_truth.astype(bool).to_numpy()
    groups = [identity_key(row) if bool(row.has_ground_truth) else "" for _, row in frame.iterrows()]
    for split, part in frame.groupby("data_split", dropna=False):
        indices = part.index.to_list()
        scores = visual[indices] @ visual[indices].T
        for left_position, left in enumerate(indices):
            for right_position, right in enumerate(indices[left_position + 1:], start=left_position + 1):
                similarity = float(scores[left_position, right_position])
                if similarity < threshold:
                    continue
                edge = tuple(sorted((left, right)))
                edges.add(edge)
                rows.append({
                    "data_split": split, "filename": frame.at[left, "filename"], "neighbour_filename": frame.at[right, "filename"],
                    "visual_cosine": similarity, "candidate_threshold": threshold,
                    "both_labelled": bool(labels_known[left] and labels_known[right]),
                    "same_identity": bool(labels_known[left] and labels_known[right] and groups[left] == groups[right]),
                })
    report = pd.DataFrame(rows)
    return edges, report


def complete_linkage_clusters(frame: pd.DataFrame, embeddings: np.ndarray, threshold: float, visual_edges: set[tuple[int, int]]) -> list[list[int]]:
    """Seed merges with visual neighbours; require all cross-pairs to pass fusion checks."""
    clusters: list[list[int]] = []
    for split, part in frame.groupby("data_split", dropna=False):
        indices = part.index.to_list()
        local = {index: position for position, index in enumerate(indices)}
        scores = embeddings[indices] @ embeddings[indices].T
        groups: list[set[int]] = [{index} for index in indices]
        group_of = {index: position for position, index in enumerate(indices)}
        candidates: list[tuple[float, int, int]] = []
        for left, right in visual_edges:
            if left not in local or right not in local:
                continue
            score = float(scores[local[left], local[right]])
            if score >= threshold and V3.pair_is_compatible(frame.loc[left], frame.loc[right]):
                candidates.append((score, left, right))
        for _, left, right in sorted(candidates, reverse=True):
            first_index, second_index = group_of[left], group_of[right]
            if first_index == second_index or not groups[first_index] or not groups[second_index]:
                continue
            first, second = groups[first_index], groups[second_index]
            mergeable = all(
                scores[local[source], local[target]] >= threshold and V3.pair_is_compatible(frame.loc[source], frame.loc[target])
                for source in first for target in second
            )
            if mergeable:
                merged = first | second
                groups[first_index], groups[second_index] = merged, set()
                for member in merged:
                    group_of[member] = first_index
        found = [sorted(group) for group in groups if len(group) >= 2]
        clusters.extend(found)
        print(f"{split}: {len(found)} strict visual-seeded clusters", flush=True)
    return clusters


def run_link(overwrite: bool) -> pd.DataFrame:
    base_path = BASE_RESULTS / "predictions_base.parquet"
    if not base_path.exists():
        raise RuntimeError(f"V3 OCR base predictions are not ready: {base_path}")
    frame = pd.read_parquet(base_path).sort_values("filename", key=lambda column: column.map(natural_key)).reset_index(drop=True)
    visual = build_visual_embeddings(frame, overwrite)
    labelled = frame.has_ground_truth.astype(bool).to_numpy()
    labels = [identity_key(row) if bool(row.has_ground_truth) else f"__test__{row.filename}" for _, row in frame.iterrows()]
    text_features, text_retrieval, _, text_processors = V3.build_features(frame)
    vision_scaler = StandardScaler()
    vision_features = vision_scaler.fit_transform(visual).astype(np.float32) * IMAGE_FEATURE_WEIGHT
    features = np.hstack([text_features.astype(np.float32), vision_features]).astype(np.float32)
    labelled_indices = np.flatnonzero(labelled).tolist()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_local, validation_local = next(splitter.split(labelled_indices, groups=[labels[index] for index in labelled_indices]))
    train_indices = [labelled_indices[index] for index in train_local]
    validation_indices = [labelled_indices[index] for index in validation_local]
    candidate_threshold, retrieval_report = calibrate_visual_candidate_threshold(visual, labels, train_indices, validation_indices)
    visual_edges, neighbour_report = visual_candidate_edges(frame, visual, candidate_threshold)
    atomic_csv(RESULTS / "visual_candidates.csv", neighbour_report)
    labelled_candidates = neighbour_report.loc[neighbour_report.both_labelled]
    retrieval_report.update({
        "candidate_edges": len(visual_edges), "labelled_candidate_pairs": int(len(labelled_candidates)),
        "same_identity_rate_among_labelled_candidates": float(labelled_candidates.same_identity.mean()) if len(labelled_candidates) else None,
        "note": "A calibrated visual threshold proposes pairs; fusion verification decides identity groups.",
    })
    atomic_json(RESULTS / "visual_retrieval_report.json", retrieval_report)
    train_pairs, train_targets = make_pairs(train_indices, labels, text_retrieval, visual, SEED)
    validation_pairs, validation_targets = make_pairs(validation_indices, labels, text_retrieval, visual, SEED + 1)
    validation_model = train_linker(features, train_pairs, train_targets, epochs=45, seed=SEED)
    validation_embeddings = embed(validation_model, features)
    threshold, validation = calibrate(score_pairs(validation_embeddings, validation_pairs), validation_targets)
    validation.update({
        "train_images": len(train_indices), "validation_images": len(validation_indices),
        "train_identity_groups": len({labels[index] for index in train_indices}),
        "validation_identity_groups": len({labels[index] for index in validation_indices}),
        "train_pairs": int(len(train_pairs)), "validation_pairs": int(len(validation_pairs)),
        "vision_encoder": VISION_MODEL_LABEL, **retrieval_report,
    })
    print(json.dumps(validation, indent=2), flush=True)
    final_pairs, final_targets = make_pairs(labelled_indices, labels, text_retrieval, visual, SEED + 2)
    final_model = train_linker(features, final_pairs, final_targets, epochs=55, seed=SEED + 2)
    embeddings = embed(final_model, features)
    torch.save({"state_dict": final_model.cpu().state_dict(), "input_dim": int(features.shape[1]), "embed_dim": EMBED_DIM}, RESULTS / "visual_fusion_siamese_linker.pt")
    joblib.dump({"text": text_processors, "vision_scaler": vision_scaler}, RESULTS / "visual_fusion_preprocessors.joblib")
    np.save(RESULTS / "visual_fusion_embeddings.npy", embeddings)
    atomic_json(RESULTS / "linker_validation.json", validation)
    clusters = complete_linkage_clusters(frame, embeddings, threshold, visual_edges)
    output, decisions, cluster_report = V3.apply_group_consensus(frame, clusters)
    for report in (decisions, cluster_report):
        if "cluster_id" in report:
            report["cluster_id"] = report.cluster_id.astype(str).str.replace("_v3_", "_v4_visual_", regex=False)
    atomic_parquet(RESULTS / "predictions_visual_group_consensus.parquet", output)
    atomic_csv(RESULTS / "predictions_visual_group_consensus.csv", output.drop(columns=[column for column in output if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "group_consensus_decisions.csv", decisions)
    atomic_csv(RESULTS / "identity_clusters.csv", cluster_report)
    metrics = V3.field_metrics(output)
    metrics.update({
        "strict_clusters": int(len(clusters)), "changed_predictions": int(len(decisions)),
        "visual_candidate_edges": int(len(visual_edges)), "official_submission_overwritten": False,
    })
    atomic_json(RESULTS / "metrics_visual_group_consensus.json", metrics)
    submission = output.loc[output.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, found {len(submission)}")
    atomic_csv(RESULTS / "test_predictions_dinov2_visual_consensus.csv", submission)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("link", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="Recompute DINOv2 image embeddings instead of using V4 cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V4 DINOv2 visual retrieval -> Siamese fusion verification -> strict group consensus",
        "vision_encoder": VISION_MODEL_LABEL, "is_vlm": False, "base_results": str(BASE_RESULTS),
        "result_dir": str(RESULTS), "official_results_untouched": True,
    }, indent=2), flush=True)
    run_link(args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
