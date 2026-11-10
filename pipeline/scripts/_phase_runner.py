#!/usr/bin/env python3
"""Shared dispatcher for the ordered COMPFEST 18 Python phase scripts.

The numbered files in this directory are deliberately small entry points.  The
tested implementations live in ``identity_ocr_compfest18.py`` so every phase
uses the same configuration, cache keys, adapters, validators, and metrics.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path("/home/nafisnaufal1426/adit/compfest_academy_project")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import identity_ocr_compfest18 as ocr


def config() -> ocr.Config:
    cfg = ocr.Config(project_dir=PROJECT_DIR)
    cfg.make_dirs()
    ocr.set_determinism(cfg.seed)
    return cfg


def audited(cfg: ocr.Config):
    return ocr.audit_dataset(cfg)


def core_tables(cfg: ocr.Config):
    _, manifest, _ = audited(cfg)
    quality = ocr._load_or_run_quality(cfg, manifest)
    localization = ocr._load_or_run_loc(cfg, manifest, True)
    return manifest, quality, localization


def selected_preprocessing(cfg: ocr.Config) -> str:
    path = cfg.output_dir / "preprocessing_summary.csv"
    if not path.exists():
        return "raw"
    return ocr.select_best_preprocessing(pd.read_csv(path), allow_adaptive=False)


def selected_ocr(cfg: ocr.Config, localization: pd.DataFrame):
    method = selected_preprocessing(cfg)
    inputs = ocr.inputs_for_method(cfg, method, localization)
    output = ocr.run_ocr_dataset(
        cfg, inputs, "ppocrv6_medium_rapidocr", method
    )
    return method, output


def save_named_predictions(cfg: ocr.Config, name: str, frame: pd.DataFrame) -> None:
    ocr.atomic_parquet(cfg.output_dir / f"{name}.parquet", frame)
    ocr.atomic_csv(cfg.output_dir / f"{name}.csv", frame)


def run_phase(number: int) -> None:
    cfg = config()
    if number == 1:
        ocr.inspect_environment(cfg)
        return
    gt, manifest, _ = audited(cfg)
    if number == 2:
        ocr.ground_truth_eda(cfg, gt)
        return
    quality = ocr._load_or_run_quality(cfg, manifest)
    if number == 3:
        return
    localization = ocr._load_or_run_loc(cfg, manifest, True)
    if number == 4:
        return
    if number == 5:
        base = ocr.geometry_manifest(cfg, manifest, localization, None)
        learned = ocr.run_d2dewarp(cfg, base)
        ocr.geometry_manifest(cfg, manifest, localization, learned)
        return
    if number == 6:
        ocr.materialize_classical_preprocessing(cfg, localization)
        ocr.run_docres(cfg, localization)
        return
    if number == 7:
        method = selected_preprocessing(cfg)
        ocr.run_backbone_ablation(cfg, manifest, quality, localization, method)
        return
    method, medium = selected_ocr(cfg, localization)
    if number == 8:
        svtr = ocr.run_svtrv2_recognition_benchmark(cfg, medium, localization)
        ocr.compare_identical_crop_recognition(cfg, medium, svtr, manifest)
        predictions = ocr.structure_ocr_results(
            cfg, svtr, manifest, localization, quality, "full"
        )
        save_named_predictions(cfg, "svtrv2_structured_predictions", predictions)
        return
    if number == 9:
        frames = []
        for model in ("ppocrv6_small_rapidocr", "ppocrv6_small_paddle"):
            result = ocr.run_ocr_dataset(
                cfg, ocr.inputs_for_method(cfg, method, localization), model, method
            )
            prediction = ocr.structure_ocr_results(
                cfg, result, manifest, localization, quality, "full"
            )
            prediction["configuration"] = model
            frames.append(prediction)
        combined = pd.concat(frames, ignore_index=True)
        save_named_predictions(cfg, "rapidocr_runtime_predictions", combined)
        summary = []
        for key, part in combined.groupby("configuration"):
            summary.append({"Model": key, **ocr.aggregate_metrics(part)})
        ocr.atomic_csv(cfg.output_dir / "rapidocr_runtime_summary.csv", pd.DataFrame(summary))
        return
    mode_by_phase = {10: "family", 11: "family_parser", 12: "graph", 13: "full"}
    if number in mode_by_phase:
        mode = mode_by_phase[number]
        prediction = ocr.structure_mode_results(
            cfg, medium, manifest, localization, quality, mode
        )
        prediction["configuration"] = mode
        save_named_predictions(cfg, f"phase_{number:02d}_{mode}_predictions", prediction)
        return
    if number == 14:
        preprocessing, summary = ocr.run_preprocessing_ablation(
            cfg, manifest, quality, localization, include_docres=True
        )
        ocr.run_resolution_ablation(cfg, manifest, quality, localization)
        best_fixed = ocr.select_best_preprocessing(summary, allow_adaptive=False)
        inputs = ocr.inputs_for_method(cfg, best_fixed, localization)
        selected = ocr.run_ocr_dataset(
            cfg, inputs, "ppocrv6_medium_rapidocr", best_fixed
        )
        ocr.save_observed_anchor_inventory(cfg, selected)
        fixed_predictions = ocr.structure_ocr_results(
            cfg, selected, manifest, localization, quality, "full"
        )
        fixed_predictions["configuration"] = f"FULL / {best_fixed}"
        best_overall = ocr.select_best_preprocessing(summary, allow_adaptive=True)
        if best_overall == "quality_aware_adaptive":
            primary = preprocessing.loc[
                preprocessing.preprocessing_method.eq(best_overall)
            ].copy()
        else:
            primary = fixed_predictions
        primary["configuration"] = f"FULL / {best_overall}"
        ocr.save_structured_artifacts(cfg, primary, "structured_predictions")
        return
    if number == 15:
        structured, _ = ocr.run_structured_ablation(
            cfg, medium, manifest, quality, localization
        )
        geometry_predictions, _ = ocr.run_geometry_ablation(
            cfg, manifest, quality, localization
        )
        preprocessing = pd.read_parquet(cfg.output_dir / "preprocessing_results.parquet")
        primary = pd.read_parquet(cfg.output_dir / "structured_predictions.parquet")
        ocr.final_system_ablation(
            cfg, manifest, quality, localization, geometry_predictions,
            preprocessing, structured, medium, method, primary,
        )
        return
    primary = pd.read_parquet(cfg.output_dir / "structured_predictions.parquet")
    if number == 16:
        ocr.evaluate_robustness(cfg, primary)
        return
    if number == 17:
        ocr.build_final_report(cfg)
        return
    if number == 18:
        ocr.inspect_environment(cfg)
        ocr.build_final_report(cfg)
        report = ocr.validate_final_artifacts(cfg)
        if not bool(report.coverage_ok.all()):
            raise RuntimeError("Final notebook refused: full-data coverage failed")
        notebook = ocr.build_notebook(cfg)
        subprocess.run(
            [str(cfg.ocr_python), str(cfg.project_dir / "tools" / "notebooks" / "execute_identity_notebook.py")],
            cwd=cfg.project_dir,
            check=True,
        )
        print(json.dumps({"notebook": str(notebook), "coverage_rows": len(report)}, indent=2))
        return
    raise ValueError(f"Unknown phase number: {number}")
