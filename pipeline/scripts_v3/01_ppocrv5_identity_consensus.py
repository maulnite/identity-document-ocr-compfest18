#!/usr/bin/env python3
"""V3: PP-OCRv6 detector + SVTRv2 Server recognizer + identity consensus.

This is deliberately a separate, reproducible pipeline.  It does not modify
``results/test_predictions.csv`` or any V1/V2 artifact.  It uses a conventional
OCR detector/recognizer (PP-OCRv6 Medium detector + SVTRv2 Server recognizer),
never a VLM.  Ground-truth identity
groups train and validate the linker only; no ground-truth value is used as an
inference feature or copied into a prediction.

Stages
------
ocr   Run the PP-OCRv6/SVTRv2 Server OCR ensemble on raw, gray, and gray+CLAHE
      inputs for every image,
      benchmark the labeled rows, and create an image-quality-routed baseline.
link  Train the identity linker, cluster labeled/test images separately, and
      apply quality-weighted field consensus only inside strict clusters.
all   Run both stages (the default).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
# Keep the failed mixed-CUDA attempt intact for auditability.  This stable retry
# uses a fresh cache and results directory, so no partial/error OCR rows can be
# reused as predictions.
RESULTS = PROJECT / "results_v3_stable"
CACHE = PROJECT / "cache_v3_stable"
sys.path.insert(0, str(PROJECT))
import identity_ocr_compfest18 as core


SEED = 1808
METHODS = ("raw", "gray", "gray_clahe")
MODEL_NAME = "ppocrv6_medium_det_cpu__svtrv2_server_rec_gpu"
MODEL_VERSION = "PP-OCRv6 Medium detector (ONNX CPU) + OpenOCR SVTRv2 Server recognizer (PyTorch CUDA)"
EMBED_DIM = 64
PAIR_PRECISION_TARGET = 0.985
NAME_CONSENSUS_RATIO = 92.0
ADDRESS_CONSENSUS_RATIO = 93.0
MIN_GROUP_SUPPORT = 0.60


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", str(value))]


def clean(value: Any) -> str:
    return core.metric_text(core.output_text(value))


def valid_date(value: Any) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", clean(value)))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=core.json_default), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(PROJECT / "results" / "dataset_manifest.csv", keep_default_na=False)
    quality = pd.read_csv(PROJECT / "results" / "image_quality.csv", keep_default_na=False)
    localization = pd.read_csv(PROJECT / "results" / "localization_results.csv", keep_default_na=False)
    for label, frame in (("manifest", manifest), ("quality", quality), ("localization", localization)):
        if len(frame) != len(manifest) or frame.filename.nunique() != len(manifest):
            raise RuntimeError(f"{label} does not contain one row for every dataset image")
    return manifest, quality, localization


def identity_key(row: pd.Series) -> str:
    """Training label only; it is never part of an OCR/linker feature."""
    # Result tables retain ground truth as ``*_gt`` so it cannot be confused
    # with predictions; the source manifest uses un-suffixed column names.
    # Supporting both forms keeps labels available for training only.
    values = []
    for column in ("name", "birth_date", "address"):
        values.append(clean(row.get(column, row.get(f"{column}_gt", ""))))
    return "\x1f".join(values)


@dataclass
class OCRItem:
    text: str
    bbox: list[list[float]]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {"text": core.output_text(self.text), "bbox": self.bbox, "confidence": float(self.confidence)}


class PPOCRSVTRv2Server:
    """Stable hybrid OCR: PP-OCRv6 detector plus SVTRv2 Server recognizer.

    Both components are dedicated OCR networks.  The recognizer is a PyTorch
    SVTRv2 Server checkpoint from OpenOCR, rather than a vision-language model.
    The detector intentionally runs on CPU: mixing its ONNX CUDA allocator with
    PyTorch CUDA caused an illegal-memory-access failure on this A100 host.
    The larger recognizer still uses the A100.
    """

    def __init__(self) -> None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangDet, ModelType, OCRVersion

        # Recognition is intentionally disabled here.  RapidOCR supplies only
        # PP-OCRv6 Medium text boxes; SVTRv2 Server is the sole recognizer.
        self.detector = RapidOCR(params={
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Global.use_cls": False,
            "Global.use_det": True,
            "Global.use_rec": False,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MEDIUM,
        })
        self.recognizer = core.SVTRv2CropRecognizer(core.CFG)
        self.runtime = "rapidocr_onnxruntime_cpu_detector+openocr_torch_cuda_recognizer"
        self.package_version = self.recognizer.model_version

    def predict(self, image: np.ndarray) -> list[OCRItem]:
        detection = self.detector(image)
        boxes = getattr(detection, "boxes", None)
        if boxes is None and isinstance(detection, (tuple, list)) and detection:
            boxes = detection[0]
        if boxes is None or len(boxes) == 0:
            return []
        crops: list[np.ndarray] = []
        valid_boxes: list[list[list[float]]] = []
        for box in boxes:
            try:
                normalized_box = np.asarray(box, dtype=float).reshape(4, 2).tolist()
                crops.append(core.perspective_crop(image, normalized_box))
                valid_boxes.append(normalized_box)
            except Exception:
                continue
        recognized = self.recognizer.predict_batch(crops, batch_size=64)
        output: list[OCRItem] = []
        for box, item in zip(valid_boxes, recognized):
            text = core.output_text(item.get("text", ""))
            if text:
                output.append(OCRItem(text, box, float(item.get("confidence", 0.0))))
        return output

    def close(self) -> None:
        del self.detector
        self.recognizer.close()


def preprocessing_inputs(method: str, manifest: pd.DataFrame, localization: pd.DataFrame) -> pd.DataFrame:
    if method == "raw":
        frame = localization[["filename", "has_ground_truth", "rectified_path"]].rename(columns={"rectified_path": "preprocessed_path"}).copy()
        frame["preprocess_latency_ms"] = 0.0
    else:
        preprocessing = pd.read_parquet(PROJECT / "results" / "preprocessing_manifest.parquet")
        frame = preprocessing.loc[
            (preprocessing.preprocessing_method.eq(method)) & preprocessing.status.isin(["ok", "cached"]),
            ["filename", "has_ground_truth", "preprocessed_path", "preprocess_latency_ms"],
        ].copy()
        missing = manifest.loc[~manifest.filename.isin(frame.filename), ["filename", "has_ground_truth"]].copy()
        if len(missing):
            fallback = localization[["filename", "rectified_path"]].rename(columns={"rectified_path": "preprocessed_path"})
            missing = missing.merge(fallback, on="filename", how="left")
            missing["preprocess_latency_ms"] = 0.0
            frame = pd.concat([frame, missing], ignore_index=True)
    frame["preprocessing_method"] = method
    frame = frame.sort_values("filename", key=lambda series: series.map(natural_key)).reset_index(drop=True)
    if len(frame) != len(manifest) or frame.filename.nunique() != len(manifest):
        raise RuntimeError(f"{method} did not produce one input path per image")
    return frame


def ocr_cache_path(filename: str, method: str) -> Path:
    digest = hashlib.sha1(f"{filename}|{method}|{MODEL_VERSION}".encode("utf-8")).hexdigest()[:16]
    return CACHE / "ocr" / f"{Path(filename).stem}__{method}__{digest}.json"


def empty_ocr_row(row: Any, method: str, error: str) -> dict[str, Any]:
    return {
        "filename": row.filename, "has_ground_truth": bool(row.has_ground_truth),
        "input_path": str(row.preprocessed_path), "preprocessing_method": method,
        "ocr_model": MODEL_NAME, "ocr_runtime": "unavailable", "model_version": MODEL_VERSION,
        "ocr_items": [], "ocr_text": "", "ocr_confidence": 0.0,
        "ocr_latency_ms": np.nan, "detector_latency_ms": np.nan, "recognition_latency_ms": np.nan,
        "ram_peak_mb": np.nan, "vram_peak_mb": np.nan,
        "preprocess_latency_ms": float(getattr(row, "preprocess_latency_ms", 0.0) or 0.0),
        "status": "error", "error": error, "cold_start_ms": np.nan,
        "model_initialization_ms": np.nan, "warmup_latency_ms": np.nan,
    }


def run_ocr(inputs: pd.DataFrame, method: str, overwrite: bool) -> pd.DataFrame:
    """Full-coverage, resumable inference; one JSON cache entry is committed per image."""
    model_started = time.perf_counter()
    try:
        adapter = PPOCRSVTRv2Server()
        initialization_ms = (time.perf_counter() - model_started) * 1000
    except Exception as error:
        details = f"initialization failed: {type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"
        result = pd.DataFrame([empty_ocr_row(row, method, details) for row in inputs.itertuples(index=False)])
        atomic_json(RESULTS / f"{MODEL_NAME}_{method}_initialization_error.json", {"error": details})
        return result

    rows: list[dict[str, Any]] = []
    warmup_ms = np.nan
    cold_start_ms = np.nan
    try:
        first = Path(inputs.iloc[0].preprocessed_path)
        if first.exists():
            start = time.perf_counter()
            adapter.predict(core.read_image(first))
            core.gpu_synchronize()
            warmup_ms = (time.perf_counter() - start) * 1000
            cold_start_ms = initialization_ms + warmup_ms
        for position, row in enumerate(inputs.itertuples(index=False), start=1):
            target = ocr_cache_path(row.filename, method)
            cached: dict[str, Any] | None = None
            if target.exists() and not overwrite:
                try:
                    cached = json.loads(target.read_text(encoding="utf-8"))
                except Exception:
                    cached = None
            if cached is not None:
                cached["input_path"] = str(row.preprocessed_path)
                rows.append(cached)
                continue
            record = empty_ocr_row(row, method, "")
            record.update({
                "ocr_runtime": adapter.runtime,
                "model_version": f"{MODEL_VERSION}; recognizer={adapter.package_version}",
                "status": "ok", "error": "", "cold_start_ms": cold_start_ms,
                "model_initialization_ms": initialization_ms, "warmup_latency_ms": warmup_ms,
            })
            try:
                image = core.read_image(Path(row.preprocessed_path))
                core.gpu_synchronize()
                started = time.perf_counter()
                with core.ResourceMonitor() as monitor:
                    items = adapter.predict(image)
                    monitor.sample()
                core.gpu_synchronize()
                record.update({
                    "ocr_items": [item.as_dict() for item in items],
                    "ocr_text": "\n".join(item.text for item in items),
                    "ocr_confidence": float(np.mean([item.confidence for item in items])) if items else 0.0,
                    "ocr_latency_ms": (time.perf_counter() - started) * 1000,
                    "ram_peak_mb": monitor.ram_peak, "vram_peak_mb": monitor.vram_peak,
                })
            except Exception as error:
                record["status"] = "error"
                record["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=2)}"
            atomic_json(target, record)
            rows.append(record)
            if position % 25 == 0:
                print(f"{method}: {position}/{len(inputs)} images committed", flush=True)
    finally:
        adapter.close()
    output = pd.DataFrame(rows).sort_values("filename", key=lambda series: series.map(natural_key)).reset_index(drop=True)
    if len(output) != len(inputs) or output.filename.nunique() != len(inputs):
        raise RuntimeError(f"OCR {method} lost or duplicated images")
    return output


def field_metrics(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.loc[frame.has_ground_truth.astype(bool)]
    latency = valid.ocr_latency_ms.dropna() if "ocr_latency_ms" in valid else pd.Series(dtype=float)
    return {
        "Name EM": float(valid.name_exact.mean()),
        "DOB EM": float(valid.dob_exact.mean()),
        "Address EM": float(valid.address_exact.mean()),
        "Macro Field EM": float(valid.macro_field_exact.mean()),
        "Full Record Accuracy": float(valid.full_record_correct.mean()),
        "CER": float(valid.record_cer.mean()),
        "Normalized Edit Similarity": float(valid.normalized_edit_similarity.mean()),
        "P50": float(latency.quantile(0.50)) if len(latency) else float("nan"),
        "P95": float(latency.quantile(0.95)) if len(latency) else float("nan"),
        "Labeled N": int(len(valid)),
    }


def choose_quality_policy(predictions: pd.DataFrame, quality: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    labeled = predictions.loc[predictions.has_ground_truth.astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    policy: dict[str, str] = {}
    for group in sorted(quality.quality_group.astype(str).unique()):
        subset = labeled.loc[labeled.quality_group.astype(str).eq(group)]
        scores = []
        for method, part in subset.groupby("preprocessing_method"):
            metric = field_metrics(part)
            scores.append({"quality_group": group, "preprocessing_method": method, **metric})
        if not scores:
            policy[group] = "raw"
            continue
        table = pd.DataFrame(scores)
        best = table.sort_values(["Macro Field EM", "CER", "P95"], ascending=[False, True, True], na_position="last").iloc[0]
        policy[group] = str(best.preprocessing_method)
        rows.extend(scores)
    return policy, pd.DataFrame(rows)


def run_ocr_stage(overwrite: bool) -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest, quality, localization = load_tables()
    prediction_frames: list[pd.DataFrame] = []
    for method in METHODS:
        inputs = preprocessing_inputs(method, manifest, localization)
        ocr = run_ocr(inputs, method, overwrite)
        atomic_parquet(RESULTS / f"ocr_{method}.parquet", ocr)
        structured = core.structure_ocr_results(core.CFG, ocr, manifest, localization, quality, mode="full")
        structured["configuration"] = f"PP-OCRv6 Medium + SVTRv2 Server / {method}"
        atomic_parquet(RESULTS / f"predictions_{method}.parquet", structured)
        atomic_csv(RESULTS / f"predictions_{method}.csv", structured.drop(columns=[column for column in structured if column.endswith("_detail")], errors="ignore"))
        prediction_frames.append(structured)
        print(json.dumps({"method": method, **field_metrics(structured)}, indent=2), flush=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    policy, benchmark = choose_quality_policy(all_predictions, quality)
    atomic_json(RESULTS / "preprocessing_policy.json", {
        "model": MODEL_VERSION, "methods_benchmarked": list(METHODS), "policy": policy,
        "selection_rule": "highest labeled Macro Field EM within each image-quality group; CER breaks ties",
        "test_safety": "The runtime route uses image quality only; no test label is read.",
    })
    atomic_csv(RESULTS / "preprocessing_benchmark.csv", benchmark)
    adaptive_parts = []
    for group, method in policy.items():
        selected = all_predictions.loc[
            all_predictions.quality_group.astype(str).eq(group) & all_predictions.preprocessing_method.eq(method)
        ].copy()
        adaptive_parts.append(selected)
    adaptive = pd.concat(adaptive_parts, ignore_index=True).drop_duplicates("filename")
    if len(adaptive) != len(manifest):
        raise RuntimeError("quality-aware selection did not cover every image exactly once")
    adaptive["selected_preprocessing_method"] = adaptive.preprocessing_method
    adaptive["preprocessing_method"] = "quality_aware_adaptive_v5"
    adaptive["configuration"] = "PP-OCRv6 Medium (CPU detector) + SVTRv2 Server (A100 recognizer) / quality-aware adaptive"
    adaptive = core.score_prediction_rows(adaptive)
    atomic_parquet(RESULTS / "predictions_base.parquet", adaptive)
    atomic_csv(RESULTS / "predictions_base.csv", adaptive.drop(columns=[column for column in adaptive if column.endswith("_detail")], errors="ignore"))
    atomic_json(RESULTS / "metrics_base.json", field_metrics(adaptive))
    return adaptive


def document_text(row: pd.Series) -> str:
    values = [row.get("ocr_text", ""), row.get("name_pred", ""), row.get("birth_date_pred", ""), row.get("address_pred", "")]
    return " ".join(core.output_text(value) for value in values if core.output_text(value)).upper()


def build_features(frame: pd.DataFrame) -> tuple[np.ndarray, Any, np.ndarray, dict[str, Any]]:
    texts = frame.apply(document_text, axis=1).tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=18000, sublinear_tf=True)
    sparse = vectorizer.fit_transform(texts)
    components = min(96, sparse.shape[0] - 1, sparse.shape[1] - 1)
    reducer = TruncatedSVD(n_components=max(2, components), random_state=SEED)
    text_features = reducer.fit_transform(sparse)
    numeric = pd.DataFrame({
        "ocr_confidence": frame.ocr_confidence.fillna(0.0),
        "ocr_item_count": frame.ocr_item_count.fillna(0.0),
        "has_name": frame.name_pred.map(lambda value: float(bool(clean(value)))),
        "has_dob": frame.birth_date_pred.map(lambda value: float(valid_date(value))),
        "has_address": frame.address_pred.map(lambda value: float(bool(clean(value)))),
        "brightness": frame.get("quality_brightness", pd.Series(0.0, index=frame.index)).fillna(0.0),
        "contrast": frame.get("quality_contrast", pd.Series(0.0, index=frame.index)).fillna(0.0),
        "illumination": frame.get("quality_illumination", pd.Series(0.0, index=frame.index)).fillna(0.0),
    })
    for family in sorted(frame.document_family.fillna("UNROUTED").astype(str).unique()):
        numeric[f"family_{family}"] = frame.document_family.fillna("UNROUTED").astype(str).eq(family).astype(float)
    scaler = StandardScaler()
    dense = np.hstack([text_features, scaler.fit_transform(numeric)]).astype(np.float32)
    retrieval = sparse.astype(np.float32)
    metadata = {"vectorizer": vectorizer, "reducer": reducer, "scaler": scaler, "numeric_columns": list(numeric.columns)}
    return dense, retrieval, sparse, metadata


def make_pairs(indices: Iterable[int], groups: list[str], retrieval: Any, seed: int, negatives_per_positive: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Positive pairs plus OCR-similar hard negatives from different identities."""
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
        raise RuntimeError("No repeated labeled identity is available for contrastive training")
    candidate_set = set(available)
    similarity = (retrieval[available] @ retrieval[available].T).toarray()
    local_to_global = {local: global_index for local, global_index in enumerate(available)}
    global_to_local = {global_index: local for local, global_index in enumerate(available)}
    negatives: set[tuple[int, int]] = set()
    for left, right in positives:
        for source in (left, right):
            source_local = global_to_local[source]
            ranking = np.argsort(-similarity[source_local])
            chosen = 0
            for candidate_local in ranking:
                candidate = local_to_global[int(candidate_local)]
                if candidate == source or groups[candidate] == groups[source] or candidate not in candidate_set:
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


class SiameseLinker(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.LayerNorm(256), torch.nn.GELU(), torch.nn.Dropout(0.10),
            torch.nn.Linear(256, 128), torch.nn.GELU(), torch.nn.Dropout(0.05),
            torch.nn.Linear(128, EMBED_DIM),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.encoder(values), p=2, dim=1)


def train_linker(features: np.ndarray, pairs: np.ndarray, labels: np.ndarray, epochs: int, seed: int) -> SiameseLinker:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseLinker(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    all_features = torch.from_numpy(features).to(device)
    pair_tensor = torch.from_numpy(pairs).to(device)
    label_tensor = torch.from_numpy(labels).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(labels), generator=generator, device=device)
        losses: list[float] = []
        for batch in order.split(512):
            left = model(all_features[pair_tensor[batch, 0]])
            right = model(all_features[pair_tensor[batch, 1]])
            cosine = (left * right).sum(dim=1)
            positive = (1.0 - cosine).pow(2)
            negative = torch.relu(cosine - 0.20).pow(2)
            loss = torch.where(label_tensor[batch] > 0.5, positive, negative).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % 10 == 0:
            print(f"linker epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.5f}", flush=True)
    return model.eval()


@torch.no_grad()
def embed(model: SiameseLinker, features: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    result = []
    for batch in torch.from_numpy(features).split(1024):
        result.append(model(batch.to(device)).detach().cpu().numpy())
    return np.vstack(result)


def pair_scores(embeddings: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return (embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]]).sum(axis=1)


def calibrate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, Any]]:
    options: list[tuple[bool, float, float, float]] = []
    for threshold in np.linspace(0.20, 0.999, 160):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum((~predicted) & (labels == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        options.append((precision >= PAIR_PRECISION_TARGET, recall, precision, float(threshold)))
    safe = [option for option in options if option[0]]
    _, recall, precision, threshold = max(safe, key=lambda value: (value[1], value[2])) if safe else max(options, key=lambda value: (value[2], value[1]))
    return threshold, {
        "threshold": threshold, "pair_precision": precision, "pair_recall": recall,
        "precision_target": PAIR_PRECISION_TARGET, "precision_target_met": bool(safe),
        "pair_auc": float(roc_auc_score(labels, scores)), "pair_average_precision": float(average_precision_score(labels, scores)),
    }


def pair_is_compatible(left: pd.Series, right: pd.Series) -> bool:
    left_dob, right_dob = clean(left.birth_date_pred), clean(right.birth_date_pred)
    if valid_date(left_dob) and valid_date(right_dob) and left_dob != right_dob:
        return False
    left_name, right_name = clean(left.name_pred), clean(right.name_pred)
    # A readable but clearly different name is strong contradictory evidence.
    if len(left_name) >= 6 and len(right_name) >= 6 and fuzz.ratio(left_name, right_name) < 65:
        return False
    return True


def complete_linkage_clusters(frame: pd.DataFrame, embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Build cliques, not transitive components: every proposed cross-edge must pass."""
    clusters: list[list[int]] = []
    for split, split_frame in frame.groupby("data_split", dropna=False):
        indices = split_frame.index.to_list()
        local_embeddings = embeddings[indices]
        scores = local_embeddings @ local_embeddings.T
        groups: list[set[int]] = [{index} for index in indices]
        group_of = {index: position for position, index in enumerate(indices)}
        candidates: list[tuple[float, int, int]] = []
        for left_position, left in enumerate(indices):
            for right_position, right in enumerate(indices[left_position + 1:], start=left_position + 1):
                score = float(scores[left_position, right_position])
                if score >= threshold and pair_is_compatible(frame.loc[left], frame.loc[right]):
                    candidates.append((score, left, right))
        for _, left, right in sorted(candidates, reverse=True):
            left_group, right_group = group_of[left], group_of[right]
            if left_group == right_group:
                continue
            first, second = groups[left_group], groups[right_group]
            if not first or not second:
                continue
            mergeable = True
            for source in first:
                for target in second:
                    source_local, target_local = indices.index(source), indices.index(target)
                    if scores[source_local, target_local] < threshold or not pair_is_compatible(frame.loc[source], frame.loc[target]):
                        mergeable = False
                        break
                if not mergeable:
                    break
            if mergeable:
                merged = first | second
                groups[left_group] = merged
                groups[right_group] = set()
                for member in merged:
                    group_of[member] = left_group
        clusters.extend(sorted(group) for group in groups if len(group) >= 2)
        print(f"{split}: {sum(len(group) >= 2 for group in groups)} strict clusters", flush=True)
    return clusters


def consensus_value(cluster: pd.DataFrame, field: str, ratio: float, date_only: bool = False) -> tuple[str, int, float]:
    candidates: list[tuple[int, str, str, float]] = []
    for index, row in cluster.iterrows():
        original = core.output_text(row[field])
        normalized = clean(original)
        if not normalized or (date_only and not valid_date(original)):
            continue
        if field == "name_pred" and len(normalized) < 5:
            continue
        candidates.append((int(index), original, normalized, max(0.01, float(row.ocr_confidence))))
    if len(candidates) < 2:
        return "", 0, 0.0
    required = max(2, math.ceil(MIN_GROUP_SUPPORT * len(cluster)))
    scored: list[tuple[int, float, float, str]] = []
    for _, original, normalized, confidence in candidates:
        support = [item for item in candidates if fuzz.ratio(normalized, item[2]) >= ratio]
        support_count = len({item[0] for item in support})
        weighted = sum(item[3] * fuzz.ratio(normalized, item[2]) / 100.0 for item in support) / max(1e-9, sum(item[3] for item in support))
        mean_confidence = float(np.mean([item[3] for item in support])) if support else confidence
        scored.append((support_count, weighted, mean_confidence, original))
    support_count, agreement, _, value = max(scored, key=lambda item: (item[0], item[1], item[2], len(clean(item[3]))))
    return (value, support_count, agreement) if support_count >= required else ("", support_count, agreement)


def apply_group_consensus(frame: pd.DataFrame, clusters: list[list[int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    decisions: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for number, indices in enumerate(clusters, start=1):
        group = output.loc[indices]
        cluster_id = f"{group.data_split.iloc[0]}_v3_{number:03d}"
        choices = {
            "name_pred": consensus_value(group, "name_pred", NAME_CONSENSUS_RATIO),
            "birth_date_pred": consensus_value(group, "birth_date_pred", 100.0, date_only=True),
            "address_pred": consensus_value(group, "address_pred", ADDRESS_CONSENSUS_RATIO),
        }
        reports.append({
            "cluster_id": cluster_id, "data_split": group.data_split.iloc[0], "member_count": len(group),
            "filenames": "|".join(sorted(group.filename.astype(str))),
            **{f"{field.removesuffix('_pred')}_support": details[1] for field, details in choices.items()},
        })
        for field, (value, support, agreement) in choices.items():
            if not value:
                continue
            for index in indices:
                previous = core.output_text(output.at[index, field])
                if clean(previous) == clean(value):
                    continue
                output.at[index, field] = value
                decisions.append({
                    "filename": output.at[index, "filename"], "cluster_id": cluster_id,
                    "data_split": output.at[index, "data_split"], "field": field.removesuffix("_pred"),
                    "previous_value": previous, "consensus_value": value,
                    "support_images": support, "weighted_agreement": agreement,
                })
    return core.score_prediction_rows(output), pd.DataFrame(decisions), pd.DataFrame(reports)


def run_link_stage(overwrite: bool) -> pd.DataFrame:
    base_path = RESULTS / "predictions_base.parquet"
    if not base_path.exists():
        raise RuntimeError("Run --stage ocr before --stage link; V3 has no base OCR predictions yet.")
    frame = pd.read_parquet(base_path).sort_values("filename", key=lambda series: series.map(natural_key)).reset_index(drop=True)
    labeled = frame.has_ground_truth.astype(bool).to_numpy()
    group_labels = [identity_key(row) if bool(row.has_ground_truth) else f"__test__{row.filename}" for _, row in frame.iterrows()]
    features, retrieval, _, preprocessors = build_features(frame)
    labeled_indices = np.flatnonzero(labeled).tolist()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_indices, validation_indices = next(splitter.split(labeled_indices, groups=[group_labels[index] for index in labeled_indices]))
    train_indices = [labeled_indices[index] for index in train_indices]
    validation_indices = [labeled_indices[index] for index in validation_indices]
    train_pairs, train_labels = make_pairs(train_indices, group_labels, retrieval, SEED)
    validation_pairs, validation_labels = make_pairs(validation_indices, group_labels, retrieval, SEED + 1)
    validation_model = train_linker(features, train_pairs, train_labels, epochs=45, seed=SEED)
    validation_embeddings = embed(validation_model, features)
    threshold, validation_report = calibrate(pair_scores(validation_embeddings, validation_pairs), validation_labels)
    validation_report.update({
        "train_images": len(train_indices), "validation_images": len(validation_indices),
        "train_pairs": int(len(train_pairs)), "validation_pairs": int(len(validation_pairs)),
        "train_identity_groups": len({group_labels[index] for index in train_indices}),
        "validation_identity_groups": len({group_labels[index] for index in validation_indices}),
    })
    print(json.dumps(validation_report, indent=2), flush=True)

    final_pairs, final_labels = make_pairs(labeled_indices, group_labels, retrieval, SEED + 2)
    final_model = train_linker(features, final_pairs, final_labels, epochs=55, seed=SEED + 2)
    embeddings = embed(final_model, features)
    torch.save({"state_dict": final_model.cpu().state_dict(), "input_dim": int(features.shape[1]), "embed_dim": EMBED_DIM}, RESULTS / "siamese_linker.pt")
    joblib.dump(preprocessors, RESULTS / "siamese_linker_preprocessors.joblib")
    np.save(RESULTS / "siamese_embeddings.npy", embeddings)
    atomic_json(RESULTS / "linker_validation.json", validation_report)

    clusters = complete_linkage_clusters(frame, embeddings, threshold)
    output, decisions, cluster_report = apply_group_consensus(frame, clusters)
    atomic_parquet(RESULTS / "predictions_group_consensus.parquet", output)
    atomic_csv(RESULTS / "predictions_group_consensus.csv", output.drop(columns=[column for column in output if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "group_consensus_decisions.csv", decisions)
    atomic_csv(RESULTS / "identity_clusters.csv", cluster_report)
    metrics = field_metrics(output)
    metrics.update({
        "strict_clusters": int(len(clusters)), "changed_predictions": int(len(decisions)),
        "official_submission_overwritten": False,
    })
    atomic_json(RESULTS / "metrics_group_consensus.json", metrics)
    submission = output.loc[output.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, found {len(submission)}")
    atomic_csv(RESULTS / "test_predictions_ppocrv6_svtrv2_group_consensus.csv", submission)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ocr", "link", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="Ignore V3 per-image OCR caches and run OCR again.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V3 PP-OCRv6 detector + SVTRv2 Server recognizer + strict Siamese identity consensus",
        "ocr": MODEL_VERSION, "is_vlm": False, "methods": METHODS,
        "result_dir": str(RESULTS), "official_results_untouched": True,
    }, indent=2), flush=True)
    if args.stage in {"ocr", "all"}:
        run_ocr_stage(args.overwrite)
    if args.stage in {"link", "all"}:
        run_link_stage(args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
