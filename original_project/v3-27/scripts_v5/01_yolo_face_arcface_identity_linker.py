#!/usr/bin/env python3
"""V5: optional YOLO-face + ArcFace evidence for strict document identity linking.

V5 consumes the completed V3 OCR base and V4 DINOv2 document embeddings.  It
does not export a face crop.  YOLOv8-Face detects a card portrait and ArcFace
turns only a sufficiently clear crop into a 512-dimensional vector.  Missing,
tiny, blurry, or low-confidence faces are explicitly marked unavailable and
fall back to the V4 OCR/document feature path.  Face similarity is never an
automatic identity decision: the fusion Siamese model, held-out calibration,
contradiction checks, and complete-linkage clustering still apply.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
V4_PATH = PROJECT / "scripts_v4" / "01_dinov2_visual_identity_linker.py"
BASE_RESULTS = PROJECT / "results_v3_stable"
V4_RESULTS = PROJECT / "results_v4_visual_linker"
RESULTS = PROJECT / "results_v5_face"
CACHE = PROJECT / "cache_v5_face"
MODELS = PROJECT / "models_v5"

SEED = 1808
FACE_MODEL_URL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
FACE_MODEL_SHA256 = "b038ca653b503453a94f6e12d76feca6840b2a97d7a1322b4498c5e922f29832"
FACE_MODEL_PATH = MODELS / "yolov8n-face-lindevs.pt"
ARCFACE_PATH = MODELS / "insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
FACE_DETECT_CONFIDENCE = 0.35
MIN_FACE_SIDE = 48
MIN_FACE_BLUR = 10.0
FACE_DIM = 512
FACE_WEIGHT = 1.00
QUALITY_WEIGHT = 0.35
EMBED_DIM = 64
FACE_BATCH_SIZE = 24


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4 = load_module(V4_PATH, "v4_face_support")
V3 = V4.V3


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


def ensure_yolo_face_weights() -> None:
    """Download and integrity-check the small face detector once."""
    MODELS.mkdir(parents=True, exist_ok=True)
    if FACE_MODEL_PATH.exists():
        digest = hashlib.sha256(FACE_MODEL_PATH.read_bytes()).hexdigest()
        if digest == FACE_MODEL_SHA256:
            return
        FACE_MODEL_PATH.unlink()
    temporary = FACE_MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(FACE_MODEL_URL, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != FACE_MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("YOLOv8-Face checkpoint checksum mismatch")
    os.replace(temporary, FACE_MODEL_PATH)


def ensure_arcface_weights() -> None:
    """Download InsightFace buffalo_l once; use only its ArcFace recognizer."""
    if ARCFACE_PATH.exists():
        return
    from insightface.utils import storage
    storage.ensure_available("models", "buffalo_l", root=str(MODELS / "insightface"))
    if not ARCFACE_PATH.exists():
        raise RuntimeError(f"ArcFace recognition checkpoint was not installed: {ARCFACE_PATH}")


class FaceEncoder:
    def __init__(self) -> None:
        ensure_yolo_face_weights()
        ensure_arcface_weights()
        from ultralytics import YOLO
        from insightface.model_zoo import get_model

        self.detector = YOLO(str(FACE_MODEL_PATH))
        # ArcFace runs on CPU deliberately.  This avoids mixing ONNX Runtime's
        # CUDA allocator with PyTorch CUDA in the same process; YOLO detection
        # still uses the A100 and ArcFace crops are small.
        self.recognizer = get_model(str(ARCFACE_PATH), providers=["CPUExecutionProvider"])
        self.recognizer.prepare(ctx_id=-1)
        self.device = 0 if torch.cuda.is_available() else "cpu"

    @staticmethod
    def unavailable(reason: str, count: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros(FACE_DIM, dtype=np.float32), {"face_available": False, "face_count": count, "face_confidence": 0.0, "face_width": 0, "face_height": 0, "face_blur": 0.0, "face_reason": reason}

    def crop_candidate(self, image: np.ndarray, result: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None, self.unavailable("not_detected", 0)[1]
        candidates: list[tuple[float, float, np.ndarray]] = []
        for bounds, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            left, top, right, bottom = np.round(bounds).astype(int)
            width, height = max(0, right - left), max(0, bottom - top)
            if width == 0 or height == 0:
                continue
            # Confidence is primary.  Area breaks ties and avoids a tiny
            # spurious detector box replacing the portrait crop.
            candidates.append((float(confidence), float(width * height), np.asarray([left, top, right, bottom], dtype=int)))
        if not candidates:
            return None, self.unavailable("invalid_box", int(len(boxes)))[1]
        confidence, _, bounds = max(candidates, key=lambda value: (value[0], value[1]))
        left, top, right, bottom = bounds
        padding_x, padding_y = int((right - left) * 0.18), int((bottom - top) * 0.18)
        left, top = max(0, left - padding_x), max(0, top - padding_y)
        right, bottom = min(image.shape[1], right + padding_x), min(image.shape[0], bottom + padding_y)
        crop = image[top:bottom, left:right]
        height, width = crop.shape[:2] if crop.size else (0, 0)
        blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0
        available = confidence >= FACE_DETECT_CONFIDENCE and min(width, height) >= MIN_FACE_SIDE and blur >= MIN_FACE_BLUR
        metadata = {
            "face_available": bool(available), "face_count": int(len(boxes)), "face_confidence": float(confidence),
            "face_width": int(width), "face_height": int(height), "face_blur": blur,
            "face_reason": "ok" if available else "low_quality",
        }
        if not available:
            return None, metadata
        return crop, metadata

    def extract_many(self, images: list[np.ndarray | None]) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Batch GPU detection and batch ArcFace crops; no raw crop is persisted."""
        output: list[tuple[np.ndarray, dict[str, Any]] | None] = [None] * len(images)
        valid_positions = [position for position, image in enumerate(images) if image is not None]
        valid_images = [images[position] for position in valid_positions]
        if valid_images:
            results = self.detector.predict(valid_images, imgsz=960, conf=0.20, device=self.device, verbose=False, max_det=10, batch=len(valid_images))
            crops: list[np.ndarray] = []
            crop_positions: list[int] = []
            for position, image, result in zip(valid_positions, valid_images, results):
                crop, metadata = self.crop_candidate(image, result)
                if crop is None:
                    output[position] = (np.zeros(FACE_DIM, dtype=np.float32), metadata)
                else:
                    output[position] = (np.zeros(FACE_DIM, dtype=np.float32), metadata)
                    crops.append(crop)
                    crop_positions.append(position)
            if crops:
                vectors = self.recognizer.get_feat(crops).astype(np.float32)
                for position, vector in zip(crop_positions, vectors):
                    metadata = output[position][1]
                    vector = vector.reshape(-1)
                    if vector.size != FACE_DIM or not np.isfinite(vector).all():
                        metadata.update({"face_available": False, "face_reason": "embedding_invalid"})
                        output[position] = (np.zeros(FACE_DIM, dtype=np.float32), metadata)
                    else:
                        vector /= np.linalg.norm(vector).clip(min=1e-12)
                        output[position] = (vector, metadata)
        for position, image in enumerate(images):
            if image is None:
                output[position] = self.unavailable("image_unreadable", 0)
        return [item for item in output if item is not None]


def face_cache_path(filename: str, input_path: str) -> Path:
    signature = f"{filename}|{input_path}|yolov8n-face-lindevs|buffalo_l-w600k-r50|v1"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return CACHE / "embeddings" / f"{Path(filename).stem}__{digest}.npz"


def load_face_embeddings(frame: pd.DataFrame, overwrite: bool) -> tuple[np.ndarray, pd.DataFrame]:
    frame = V4.resolve_inputs(frame).sort_values("filename", key=lambda column: column.map(V3.natural_key)).reset_index(drop=True)
    vectors: list[np.ndarray | None] = [None] * len(frame)
    details: list[dict[str, Any] | None] = [None] * len(frame)
    pending: list[tuple[int, Path]] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        target = face_cache_path(row.filename, str(row.input_path))
        if target.exists() and not overwrite:
            try:
                cache = np.load(target, allow_pickle=False)
                vector = cache["embedding"].astype(np.float32)
                if vector.shape == (FACE_DIM,) and np.isfinite(vector).all():
                    vectors[position] = vector
                    details[position] = {
                        "filename": row.filename, "face_available": bool(cache["available"].item()), "face_count": int(cache["count"].item()),
                        "face_confidence": float(cache["confidence"].item()), "face_width": int(cache["width"].item()),
                        "face_height": int(cache["height"].item()), "face_blur": float(cache["blur"].item()), "face_reason": str(cache["reason"].item()),
                    }
                    continue
            except Exception:
                pass
        pending.append((position, Path(row.input_path)))
    if pending:
        started = time.perf_counter()
        encoder = FaceEncoder()
        print(json.dumps({"new_face_inputs": len(pending), "cached_face_inputs": len(frame) - len(pending), "yolo_device": encoder.device, "arcface_device": "CPU"}), flush=True)
        for first in range(0, len(pending), FACE_BATCH_SIZE):
            chunk = pending[first:first + FACE_BATCH_SIZE]
            extracted = encoder.extract_many([cv2.imread(str(path), cv2.IMREAD_COLOR) for _, path in chunk])
            if len(extracted) != len(chunk):
                raise RuntimeError("Face batch detector returned an unexpected number of results")
            for (position, _), (vector, metadata) in zip(chunk, extracted):
                row = frame.iloc[position]
                target = face_cache_path(str(row.filename), str(row.input_path))
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, embedding=vector, available=int(metadata["face_available"]), count=int(metadata["face_count"]), confidence=float(metadata["face_confidence"]), width=int(metadata["face_width"]), height=int(metadata["face_height"]), blur=float(metadata["face_blur"]), reason=str(metadata["face_reason"]))
                os.replace(temporary, target)
                vectors[position] = vector
                details[position] = {"filename": row.filename, **metadata}
            print(f"face embedding: {min(first + len(chunk), len(pending))}/{len(pending)} newly committed", flush=True)
        print(f"face embedding seconds: {time.perf_counter() - started:.1f}", flush=True)
    if any(vector is None for vector in vectors) or any(detail is None for detail in details):
        raise RuntimeError("Face cache did not produce one record per image")
    embeddings = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    metadata = pd.DataFrame([dict(detail) for detail in details])
    np.save(RESULTS / "face_embeddings.npy", embeddings)
    atomic_csv(RESULTS / "face_detection_manifest.csv", metadata)
    atomic_json(RESULTS / "face_embedding_summary.json", {
        "detector": "YOLOv8n-Face", "recognizer": "ArcFace ResNet50 WebFace600K (buffalo_l)",
        "face_available": int(metadata.face_available.sum()), "face_unavailable": int((~metadata.face_available).sum()),
        "fallback": "V4 OCR + document embedding for every face_unavailable row; no raw face crop is saved.",
        "minimum_confidence": FACE_DETECT_CONFIDENCE, "minimum_face_side": MIN_FACE_SIDE, "minimum_blur": MIN_FACE_BLUR,
    })
    return embeddings, metadata


class FaceFusionLinker(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 448), torch.nn.LayerNorm(448), torch.nn.GELU(), torch.nn.Dropout(0.12),
            torch.nn.Linear(448, 192), torch.nn.GELU(), torch.nn.Dropout(0.06), torch.nn.Linear(192, EMBED_DIM),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.encoder(values), p=2, dim=1)


def train_linker(features: np.ndarray, pairs: np.ndarray, labels: np.ndarray, epochs: int, seed: int) -> FaceFusionLinker:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceFusionLinker(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=1e-4)
    values, pair_tensor, targets = torch.from_numpy(features).to(device), torch.from_numpy(pairs).to(device), torch.from_numpy(labels).to(device)
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
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % 10 == 0:
            print(f"face fusion linker epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.5f}", flush=True)
    return model.eval()


@torch.inference_mode()
def embed(model: FaceFusionLinker, features: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    return np.vstack([model(batch.to(device)).cpu().numpy() for batch in torch.from_numpy(features).split(1024)]).astype(np.float32)


def run(overwrite: bool) -> pd.DataFrame:
    base_path, visual_path = BASE_RESULTS / "predictions_base.parquet", V4_RESULTS / "visual_embeddings.npy"
    if not base_path.exists() or not visual_path.exists():
        raise RuntimeError("V5 needs completed V3 base predictions and V4 visual_embeddings.npy")
    frame = pd.read_parquet(base_path).sort_values("filename", key=lambda column: column.map(V3.natural_key)).reset_index(drop=True)
    visual = np.load(visual_path).astype(np.float32)
    if visual.shape[0] != len(frame):
        raise RuntimeError("V4 visual embedding count does not match V3 base predictions")
    face, face_meta = load_face_embeddings(frame, overwrite)
    labelled = frame.has_ground_truth.astype(bool).to_numpy()
    groups = [V3.identity_key(row) if bool(row.has_ground_truth) else f"__test__{row.filename}" for _, row in frame.iterrows()]
    text_features, text_retrieval, _, text_processors = V3.build_features(frame)
    document_features = StandardScaler().fit_transform(visual).astype(np.float32) * 0.70
    available = face_meta.face_available.to_numpy(dtype=bool)
    face_features = np.zeros_like(face, dtype=np.float32)
    face_scaler: Any = None
    if available.sum() >= 2:
        face_scaler = StandardScaler().fit(face[available])
        face_features[available] = face_scaler.transform(face[available]).astype(np.float32) * FACE_WEIGHT
    quality = face_meta[["face_available", "face_confidence", "face_width", "face_height", "face_blur"]].astype(float).to_numpy()
    quality_features = StandardScaler().fit_transform(quality).astype(np.float32) * QUALITY_WEIGHT
    features = np.hstack([text_features.astype(np.float32), document_features, face_features, quality_features]).astype(np.float32)
    labelled_indices = np.flatnonzero(labelled).tolist()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_local, validation_local = next(splitter.split(labelled_indices, groups=[groups[index] for index in labelled_indices]))
    train_indices = [labelled_indices[index] for index in train_local]
    validation_indices = [labelled_indices[index] for index in validation_local]
    candidate_threshold, candidate_report = V4.calibrate_visual_candidate_threshold(visual, groups, train_indices, validation_indices)
    visual_edges, _ = V4.visual_candidate_edges(frame, visual, candidate_threshold)
    train_pairs, train_labels = V4.make_pairs(train_indices, groups, text_retrieval, visual, SEED)
    validation_pairs, validation_labels = V4.make_pairs(validation_indices, groups, text_retrieval, visual, SEED + 1)
    validation_model = train_linker(features, train_pairs, train_labels, epochs=45, seed=SEED)
    validation_embeddings = embed(validation_model, features)
    threshold, validation = V4.calibrate(V4.score_pairs(validation_embeddings, validation_pairs), validation_labels)
    validation.update({
        "train_images": len(train_indices), "validation_images": len(validation_indices), "train_pairs": int(len(train_pairs)), "validation_pairs": int(len(validation_pairs)),
        "train_identity_groups": len({groups[index] for index in train_indices}), "validation_identity_groups": len({groups[index] for index in validation_indices}),
        "face_available_total": int(available.sum()), "face_available_train": int(available[train_indices].sum()), "face_available_validation": int(available[validation_indices].sum()),
        **candidate_report,
    })
    print(json.dumps(validation, indent=2), flush=True)
    final_pairs, final_labels = V4.make_pairs(labelled_indices, groups, text_retrieval, visual, SEED + 2)
    final_model = train_linker(features, final_pairs, final_labels, epochs=55, seed=SEED + 2)
    embeddings = embed(final_model, features)
    torch.save({"state_dict": final_model.cpu().state_dict(), "input_dim": int(features.shape[1]), "embed_dim": EMBED_DIM}, RESULTS / "face_fusion_siamese_linker.pt")
    joblib.dump({"text": text_processors, "document_scaler": "StandardScaler", "face_scaler": face_scaler, "fallback_mask": "face_available"}, RESULTS / "face_fusion_preprocessors.joblib")
    np.save(RESULTS / "face_fusion_embeddings.npy", embeddings)
    atomic_json(RESULTS / "linker_validation.json", validation)
    clusters = V4.complete_linkage_clusters(frame, embeddings, threshold, visual_edges)
    output, decisions, clusters_report = V3.apply_group_consensus(frame, clusters)
    for report in (decisions, clusters_report):
        if "cluster_id" in report:
            report["cluster_id"] = report.cluster_id.astype(str).str.replace("_v3_", "_v5_face_", regex=False)
    atomic_parquet(RESULTS / "predictions_face_group_consensus.parquet", output)
    atomic_csv(RESULTS / "predictions_face_group_consensus.csv", output.drop(columns=[column for column in output if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "group_consensus_decisions.csv", decisions)
    atomic_csv(RESULTS / "identity_clusters.csv", clusters_report)
    metrics = V3.field_metrics(output)
    metrics.update({"strict_clusters": int(len(clusters)), "changed_predictions": int(len(decisions)), "face_available": int(available.sum()), "official_submission_overwritten": False})
    atomic_json(RESULTS / "metrics_face_group_consensus.json", metrics)
    submission = output.loc[output.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, found {len(submission)}")
    atomic_csv(RESULTS / "test_predictions_yolo_face_arcface_consensus.csv", submission)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Recompute per-document face vectors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "pipeline": "V5 YOLO-face -> ArcFace optional embedding -> OCR/document/face fusion -> strict consensus",
        "face_fallback": "Rows without a quality-approved face use V4 OCR + document features only.",
        "raw_face_crop_saved": False, "official_results_untouched": True,
    }, indent=2), flush=True)
    run(args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
