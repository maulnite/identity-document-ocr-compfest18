#!/usr/bin/env python3
"""V11: conservative MyKad address reformatting after V10 span selection.

V10 showed that the main remaining error in many high-similarity predictions is
missing whitespace (for example ``KUALALUMPUR``).  This experiment learns a
word lexicon from the training split only, then performs deterministic word
boundary restoration.  It never substitutes OCR characters or copies an
address value.  Both calibration and group-held-out validation must pass
before the formatted candidate is released.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V3_PATH = PROJECT / "scripts_v3" / "01_ppocrv5_identity_consensus.py"
V10_PATH = PROJECT / "scripts_v10" / "01_supervised_mykad_address_span_ranker.py"
V3_RESULTS = PROJECT / "results_v3_stable"
RESULTS = PROJECT / "results_v11_mykad_format_consensus"

SEED = 1808
FAMILY = "MYKAD"

# Public structural vocabulary.  Training vocabulary extends this list but is
# never built from the validation/test field values.
FIXED_WORDS = frozenset("""
JALAN JLN LORONG KAMPUNG KG TAMAN DESA BANDAR SEKSYEN BLOK BLK MUKIM FELDA
PANGSAPURI APARTMENT RUMAH KUALA LUMPUR TERENGGANU PULAU PINANG PERSEKUTUAN
SELANGOR KEDAH JOHOR KELANTAN PERAK PAHANG SABAH SARAWAK MELAKA NEGERI
SEMBILAN SHAH ALAM KOTA BHARU KOTA TINGGI BATU CAVES PARIT SULONG WAKAF
BHARU KEPALA BATAS SUNGAI MACHANG DUNGUN KEMAMAN NILAI JASIN IPOH PEKAN
SENGGARANG TUMPAT MIRI MULU RASA SEPANG SKUDAI KLUANG KLANG KUCHING
PUTRAJAYA JITRA
""".split())


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_module(V3_PATH, "v3_v11_support")
V10 = load_module(V10_PATH, "v10_v11_support")
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


class AddressFormatter:
    def __init__(self, addresses: pd.Series) -> None:
        words = set(FIXED_WORDS)
        for address in addresses.dropna().astype(str):
            words.update(token for token in re.findall(r"[A-Z]{3,}", address.upper()) if len(token) >= 3)
        self.words = tuple(sorted(words, key=lambda value: (-len(value), value)))

    def segment(self, value: str) -> str:
        """Split a fused uppercase run only when it begins with known text."""
        parts: list[str] = []
        index = 0
        known_first = False
        while index < len(value):
            match = next((word for word in self.words if value.startswith(word, index)), None)
            if match is not None:
                if index == 0:
                    known_first = True
                parts.append(match)
                index += len(match)
                continue
            end = index + 1
            while end < len(value) and not any(value.startswith(word, end) for word in self.words):
                end += 1
            parts.append(value[index:end])
            index = end
        # Starting with known address vocabulary is a conservative guard: it
        # avoids splitting OCR gibberish merely because it contains a short
        # substring found in the training vocabulary.
        return " ".join(parts) if known_first and len(parts) > 1 else value

    def format(self, value: str) -> str:
        text = core.output_text(value).upper().strip()
        if not text:
            return ""
        # Prefixes and postcodes have unambiguous address formatting.  The
        # patterns deliberately avoid generic letter/digit splitting, which
        # would corrupt OCR forms such as ``U2/1E``.
        text = re.sub(r"\b(NO|LOT|PT|TL|H)\s*(\d+)(?=[A-Z])", r"\1 \2 ", text)
        text = re.sub(r"\b(MUKIM|BATU)\s*(\d+)(?=[A-Z])", r"\1 \2 ", text)
        text = re.sub(r"\b(JALAN|JLN|SEKSYEN)\s*(\d+(?:/\d+[A-Z]?)?)(?=[,\s]|$)", r"\1 \2", text)
        text = re.sub(r"(\d{4,5})(?=[A-Z])", r"\1 ", text)
        text = re.sub(r"(-\d{2})(?=[A-Z])", r"\1 ", text)
        text = re.sub(r"(?<=\.)(?=[A-Z])", " ", text)
        text = re.sub(r"[A-Z]{3,}", lambda match: self.segment(match.group(0)), text)
        return re.sub(r"\s+", " ", text).strip()


def apply_formatter(frame: pd.DataFrame, formatter: AddressFormatter) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    selected = output.document_family.eq(FAMILY) & output.get("span_address", pd.Series(index=output.index, dtype=object)).notna()
    old_values = output.loc[selected, "address_pred"].copy()
    output.loc[selected, "address_pred"] = old_values.map(formatter.format)
    changed = selected & output.address_pred.map(V10.V8.clean).ne(frame.address_pred.map(V10.V8.clean))
    decisions = output.loc[changed, ["filename", "data_split", "address_pred", "span_address", "span_probability", "span_line_ids"]].rename(columns={"address_pred": "formatted_address", "span_address": "unformatted_span"})
    return core.score_prediction_rows(output), decisions


def split_files(train_files: set[str], groups: dict[str, str]) -> tuple[set[str], set[str]]:
    documents = sorted(train_files)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    fit_positions, calibration_positions = next(splitter.split(documents, groups=[groups[name] for name in documents]))
    return ({documents[position] for position in fit_positions}, {documents[position] for position in calibration_positions})


def select_threshold(base: pd.DataFrame, spans: pd.DataFrame, train_files: set[str], groups: dict[str, str]) -> tuple[float, dict[str, Any]]:
    fit_files, calibration_files = split_files(train_files, groups)
    model = V10.SpanRanker().fit(spans.loc[spans.filename.isin(fit_files)])
    formatter = AddressFormatter(base.loc[base.filename.isin(fit_files), "address_gt"])
    baseline = V10.mykad_metrics(base.loc[base.filename.isin(calibration_files)].copy())
    options: list[dict[str, Any]] = []
    for threshold in V10.THRESHOLDS:
        span_candidate, span_decisions = V10.build_predictions(base.loc[base.filename.isin(calibration_files)].copy(), spans.loc[spans.filename.isin(calibration_files)], model, float(threshold))
        candidate, decisions = apply_formatter(span_candidate, formatter)
        metrics = V10.mykad_metrics(candidate)
        safe = metrics["exact_match"] >= baseline["exact_match"] and metrics["mean_similarity"] >= baseline["mean_similarity"] + 0.03
        options.append({"threshold": float(threshold), "safe": bool(safe), "span_changes": int(len(span_decisions)), "format_changes": int(len(decisions)), **metrics})
    safe_options = [option for option in options if option["safe"]]
    selected = max(safe_options, key=lambda option: (option["exact_match"], option["mean_similarity"], -option["format_changes"])) if safe_options else max(options, key=lambda option: (option["exact_match"], option["mean_similarity"], -option["format_changes"]))
    return float(selected["threshold"]), {"fit_images": len(fit_files), "calibration_images": len(calibration_files), "baseline": baseline, "selected": selected, "options": options}


def run() -> None:
    base = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda value: value.map(V3.natural_key)).reset_index(drop=True)
    mykad = base.loc[base.document_family.eq(FAMILY)].copy()
    lines = V10.flatten_lines_fast(mykad)
    spans = V10.span_rows(base, lines)
    labelled = mykad.loc[mykad.has_ground_truth.astype(bool) & mykad.address_present_gt.astype(bool)].copy()
    groups = {row.filename: V3.identity_key(row) for _, row in labelled.iterrows()}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_positions, validation_positions = next(splitter.split(labelled, groups=[groups[name] for name in labelled.filename]))
    train_files = set(labelled.iloc[train_positions].filename)
    validation_files = set(labelled.iloc[validation_positions].filename)
    threshold, calibration = select_threshold(base, spans, train_files, groups)
    model = V10.SpanRanker().fit(spans.loc[spans.filename.isin(train_files)])
    formatter = AddressFormatter(base.loc[base.filename.isin(train_files), "address_gt"])
    validation_base = base.loc[base.filename.isin(validation_files)].copy()
    validation_span, span_decisions = V10.build_predictions(validation_base, spans.loc[spans.filename.isin(validation_files)], model, threshold)
    validation_candidate, format_decisions = apply_formatter(validation_span, formatter)
    baseline_metrics = V10.mykad_metrics(validation_base)
    span_metrics = V10.mykad_metrics(validation_span)
    candidate_metrics = V10.mykad_metrics(validation_candidate)
    release = bool(calibration["selected"]["safe"]) and candidate_metrics["exact_match"] >= baseline_metrics["exact_match"] and candidate_metrics["mean_similarity"] >= baseline_metrics["mean_similarity"] + 0.03
    validation = {
        "family": FAMILY, "train_images": len(train_files), "validation_images": len(validation_files),
        "threshold": threshold, "calibration": calibration, "baseline": baseline_metrics,
        "span_candidate": span_metrics, "formatted_candidate": candidate_metrics,
        "span_changes": int(len(span_decisions)), "format_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    final_model = V10.SpanRanker().fit(spans)
    final_formatter = AddressFormatter(base.loc[base.has_ground_truth.astype(bool) & base.document_family.eq(FAMILY), "address_gt"])
    span_candidate, span_decisions = V10.build_predictions(base, spans, final_model, threshold)
    candidate, format_decisions = apply_formatter(span_candidate, final_formatter)
    released = candidate if release else core.score_prediction_rows(base.copy())
    atomic_parquet(RESULTS / "predictions_candidate.parquet", candidate)
    atomic_csv(RESULTS / "predictions_candidate.csv", candidate.drop(columns=[column for column in candidate if column.endswith("_detail")], errors="ignore"))
    atomic_parquet(RESULTS / "predictions_released.parquet", released)
    atomic_csv(RESULTS / "predictions_released.csv", released.drop(columns=[column for column in released if column.endswith("_detail")], errors="ignore"))
    atomic_csv(RESULTS / "span_decisions.csv", span_decisions)
    atomic_csv(RESULTS / "format_decisions.csv", format_decisions)
    atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "span_candidate_changes": int(len(span_decisions)), "format_candidate_changes": int(len(format_decisions)),
        "test_release_gate_passed": bool(release), "official_submission_overwritten": False,
        **{f"released_mykad_{key}": value for key, value in V10.mykad_metrics(released).items()},
        **{f"candidate_mykad_{key}": value for key, value in V10.mykad_metrics(candidate).items()},
    })
    atomic_json(RESULTS / "metrics.json", metrics)
    submission = released.loc[released.data_split.eq("test"), ["filename", "name_pred", "birth_date_pred", "address_pred"]].rename(columns={"name_pred": "name", "birth_date_pred": "birth_date", "address_pred": "address"})
    if len(submission) != 100:
        raise RuntimeError(f"Expected 100 test rows, got {len(submission)}")
    atomic_csv(RESULTS / "test_predictions.csv", submission)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V11 V10 span ranker + conservative train-only MyKad spacing restoration", "ocr_rerun": False, "gpu_required": False, "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
