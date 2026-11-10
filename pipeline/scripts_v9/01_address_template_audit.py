#!/usr/bin/env python3
"""V9: ground-truth audit of address-span feasibility and review pack.

This is an analysis tool, not an inference pipeline.  It enumerates compact
OCR-line spans on labelled cards and computes the *oracle* span—the closest
span possible from text that V3 already detected.  That separates a parser
problem (oracle is good) from a recognition problem (oracle is poor).

It also writes a stratified review pack of 40 non-empty-address failures:
numbered OCR-box overlay images plus a CSV template for human line annotations.
The overlays contain no ground-truth text; it remains in the accompanying
project-local CSV for the reviewer.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
V8_PATH = PROJECT / "scripts_v8" / "01_weakly_supervised_address_span_selector.py"
V3_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v9_address_template_audit"
MAX_SPAN_LINES = 4
REVIEW_COUNT = 40


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_module(V3_PATH, "v3_template_audit_support")
V8 = load_module(V8_PATH, "v8_template_audit_support")
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


def candidate_spans(lines: pd.DataFrame) -> list[dict[str, Any]]:
    """All short, spatially coherent OCR-line spans; labels are not consulted."""
    ordered = lines.sort_values(["y", "x", "ordinal"]).reset_index(drop=True)
    usable = ordered.loc[(ordered.metadata < 0.5) & (ordered.identifier < 0.5) & (ordered.date_like < 0.5)].copy().reset_index(drop=True)
    output: list[dict[str, Any]] = []
    for start in range(len(usable)):
        for end in range(start, min(len(usable), start + MAX_SPAN_LINES)):
            span = usable.iloc[start:end + 1]
            if len(span) > 1 and span.y.diff().dropna().max() > 0.18:
                continue
            output.append({
                "text": core.output_text(", ".join(span.text.tolist())),
                "line_ids": [int(value) for value in span.ordinal.tolist()],
                "line_count": int(len(span)), "has_street_cue": bool(span.street_cue.max()),
            })
    return output


def oracle_for_document(part: pd.DataFrame) -> dict[str, Any]:
    target = core.output_text(part.address_gt.iloc[0])
    choices = candidate_spans(part)
    if not choices:
        return {"oracle_text": "", "oracle_line_ids": [], "oracle_similarity": 0.0, "oracle_exact": False, "oracle_line_count": 0, "oracle_has_street_cue": False}
    best = max(choices, key=lambda item: core.text_metrics(target, item["text"])[1])
    _, similarity, exact = core.text_metrics(target, best["text"])
    return {
        "oracle_text": best["text"], "oracle_line_ids": best["line_ids"], "oracle_similarity": float(similarity), "oracle_exact": bool(exact),
        "oracle_line_count": best["line_count"], "oracle_has_street_cue": best["has_street_cue"],
    }


def oracle_table(lines: pd.DataFrame) -> pd.DataFrame:
    labelled = lines.loc[lines.has_ground_truth.astype(bool)].copy()
    labelled = labelled.loc[labelled.address_gt.map(V8.clean).ne("")]
    rows: list[dict[str, Any]] = []
    for filename, part in labelled.groupby("filename", sort=False):
        first = part.iloc[0]
        base_cer, base_similarity, base_exact = core.text_metrics(first.address_gt, first.base_address)
        oracle = oracle_for_document(part)
        rows.append({
            "filename": filename, "document_family": first.document_family, "quality_group": first.quality_group,
            "base_exact": bool(base_exact), "base_similarity": float(base_similarity), "base_cer": float(base_cer),
            **oracle,
        })
    return pd.DataFrame(rows)


def render_overlay(filename: str, source: pd.Series, destination: Path) -> None:
    image = cv2.imread(str(source.input_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    items = V8.read_items(source.ocr_items)
    for ordinal, item in enumerate(items):
        box = V8.item_box(item)
        if box is None:
            continue
        box_int = np.asarray(box, dtype=np.int32)
        center = box_int.mean(axis=0).astype(int)
        cv2.polylines(image, [box_int], True, (0, 180, 255), 2)
        cv2.putText(image, str(ordinal), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    maximum = 1800
    if image.shape[1] > maximum:
        scale = maximum / image.shape[1]
        image = cv2.resize(image, (maximum, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)


def choose_review_rows(table: pd.DataFrame) -> pd.DataFrame:
    failures = table.loc[~table.base_exact].copy()
    # Include both high-oracle parser opportunities and low-oracle OCR-hard
    # cases, stratified by detected document family.
    selected: list[pd.DataFrame] = []
    families = list(failures.document_family.fillna("UNKNOWN").value_counts().index)
    per_family = max(2, REVIEW_COUNT // max(1, len(families)))
    for family in families:
        part = failures.loc[failures.document_family.fillna("UNKNOWN").eq(family)]
        high = part.sort_values("oracle_similarity", ascending=False).head(per_family // 2 + 1)
        low = part.sort_values("oracle_similarity", ascending=True).head(per_family // 2 + 1)
        selected.append(pd.concat([high, low], ignore_index=True))
    combined = pd.concat(selected, ignore_index=True).drop_duplicates("filename") if selected else failures.head(0)
    if len(combined) < REVIEW_COUNT:
        remainder = failures.loc[~failures.filename.isin(combined.filename)].sort_values("oracle_similarity", ascending=False)
        combined = pd.concat([combined, remainder.head(REVIEW_COUNT - len(combined))], ignore_index=True)
    return combined.head(REVIEW_COUNT).copy()


def run() -> None:
    base = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda column: column.map(V3.natural_key)).reset_index(drop=True)
    lines = V8.flatten_lines(base)
    table = oracle_table(lines)
    if table.empty:
        raise RuntimeError("No labelled non-empty-address rows found for template audit")
    family = table.groupby("document_family", dropna=False).agg(
        images=("filename", "count"), base_exact=("base_exact", "mean"), base_similarity=("base_similarity", "mean"),
        oracle_exact=("oracle_exact", "mean"), oracle_similarity=("oracle_similarity", "mean"), oracle_street_cue=("oracle_has_street_cue", "mean"),
    ).reset_index().sort_values("images", ascending=False)
    summary = {
        "nonempty_labelled_images": int(len(table)),
        "base_exact_match": float(table.base_exact.mean()), "base_mean_similarity": float(table.base_similarity.mean()),
        "oracle_exact_match": float(table.oracle_exact.mean()), "oracle_mean_similarity": float(table.oracle_similarity.mean()),
        "interpretation": "High oracle similarity/exactness means existing OCR lines contain the answer and a layout/span rule can help; low oracle similarity means recognition/capture information is the limiting factor.",
    }
    review = choose_review_rows(table)
    source_rows = V8.select_ocr_rows(base)
    detailed_rows: list[dict[str, Any]] = []
    for row in review.itertuples(index=False):
        source = source_rows.get(str(row.filename))
        if source is None:
            continue
        render_overlay(str(row.filename), source, RESULTS / "overlays" / f"{Path(str(row.filename)).stem}_lines.png")
        part = lines.loc[lines.filename.eq(row.filename)].sort_values("ordinal")
        detailed_rows.append({
            "filename": row.filename, "document_family": row.document_family, "quality_group": row.quality_group,
            "ground_truth_address": part.address_gt.iloc[0], "base_address": part.base_address.iloc[0],
            "oracle_line_ids": "|".join(map(str, row.oracle_line_ids)), "oracle_text": row.oracle_text,
            "oracle_similarity": row.oracle_similarity,
            "ocr_lines": " || ".join(f"{int(item.ordinal)}:{item.text}" for item in part.itertuples(index=False)),
            "review_selected_line_ids": "", "review_notes": "",
        })
    atomic_csv(RESULTS / "oracle_span_audit.csv", table)
    atomic_csv(RESULTS / "template_oracle_summary.csv", family)
    atomic_csv(RESULTS / "review_tasks.csv", pd.DataFrame(detailed_rows))
    atomic_json(RESULTS / "audit_summary.json", summary)
    print(json.dumps({**summary, "review_tasks": int(len(detailed_rows))}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    run()
