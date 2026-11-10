"""Minimal compatibility core for downstream V24--V27 reproduction only.

Raw-image OCR is intentionally unsupported in this portable bundle.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def output_text(value: Any) -> str:
    value = unicodedata.normalize("NFC", "" if value is None else str(value))
    return re.sub(r"\s+", " ", value).strip()


def metric_text(value: Any) -> str:
    value = unicodedata.normalize("NFC", output_text(value)).upper()
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s*([,;/.-])\s*", r"\1", value)
    return re.sub(r"\s+", " ", value).strip(" ,")


def comparison_text(value: Any) -> str:
    value = unicodedata.normalize("NFKD", "" if value is None else str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.upper().replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def levenshtein_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def text_metrics(truth: str, prediction: str) -> tuple[float, float, bool]:
    truth, prediction = metric_text(truth), metric_text(prediction)
    distance = levenshtein_distance(truth, prediction)
    return float(distance / max(1, len(truth))), float(max(0, 1 - distance / max(1, len(truth), len(prediction)))), truth == prediction


def score_prediction_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for truth_column, prediction_column, prefix in (("name_gt", "name_pred", "name"), ("birth_date_gt", "birth_date_pred", "dob"), ("address_gt", "address_pred", "address")):
        metrics = [text_metrics(truth, prediction) for truth, prediction in zip(output[truth_column].fillna(""), output[prediction_column].fillna(""))]
        output[f"{prefix}_cer"] = [item[0] for item in metrics]
        output[f"{prefix}_similarity"] = [item[1] for item in metrics]
        output[f"{prefix}_exact"] = [item[2] for item in metrics]
    output["macro_field_exact"] = (output.name_exact.astype(float) + output.dob_exact.astype(float) + output.address_exact.astype(float)) / 3
    output["full_record_correct"] = output.name_exact & output.dob_exact & output.address_exact
    output["address_present_gt"] = output.address_gt.fillna("").map(lambda value: bool(metric_text(value)))
    output["address_present_pred"] = output.address_pred.fillna("").map(lambda value: bool(metric_text(value)))
    output["record_cer"] = (output.name_cer + output.dob_cer + output.address_cer) / 3
    output["normalized_edit_similarity"] = (output.name_similarity + output.dob_similarity + output.address_similarity) / 3
    return output


MONTHS = {month: number for number, names in enumerate([
    ("JAN", "JANUARY"), ("FEB", "FEBRUARY"), ("MAR", "MARCH"), ("APR", "APRIL"), ("MAY",), ("JUN", "JUNE"),
    ("JUL", "JULY"), ("AUG", "AUGUST"), ("SEP", "SEPT", "SEPTEMBER"), ("OCT", "OCTOBER"), ("NOV", "NOVEMBER"), ("DEC", "DECEMBER"),
], 1) for month in names}
MONTH_RE = "|".join(MONTHS)
DATE_PATTERNS = [
    re.compile(r"\b(\d{4})[-/. ](\d{1,2})[-/. ](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{4})\b"),
    re.compile(rf"\b(\d{{1,2}})[ .-]+({MONTH_RE})[ .-]+(\d{{4}})\b", re.I),
    re.compile(r"\b(\d{6})\b"), re.compile(r"\b((?:19|20)\d{2})\b"),
]


def normalize_date(value: Any, reference_year: int = 2026) -> str:
    value = comparison_text(value)
    for index, pattern in enumerate(DATE_PATTERNS):
        match = pattern.search(value)
        if not match:
            continue
        try:
            if index == 0:
                year, month, day = map(int, match.groups())
            elif index == 1:
                day, month, year = map(int, match.groups())
            elif index == 2:
                day, month_name, year = match.groups(); day, month, year = int(day), MONTHS[month_name.upper()], int(year)
            elif index == 3:
                digits = match.group(1); year, month, day = int(digits[:2]), int(digits[2:4]), int(digits[4:6]); year = 1900 + year if year > reference_year % 100 else 2000 + year
            else:
                return match.group(1)
            return date(year, month, day).isoformat()
        except (ValueError, KeyError):
            return ""
    return ""


def json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.bool_): return bool(value)
    if pd.isna(value): return None
    raise TypeError(f"Cannot serialize {type(value)}")


def _raw_ocr_unavailable(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("This portable bundle reproduces V24--V27 from frozen OCR tables; raw-image OCR is not included.")


perspective_crop = _raw_ocr_unavailable
read_image = _raw_ocr_unavailable
extract_fields = _raw_ocr_unavailable
structure_ocr_results = _raw_ocr_unavailable
gpu_synchronize = _raw_ocr_unavailable


class ResourceMonitor:
    def __enter__(self) -> "ResourceMonitor": return self
    def __exit__(self, *_args: Any) -> None: return None


class _UnavailableRecognizer:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None: _raw_ocr_unavailable()


SVTRv2CropRecognizer = _UnavailableRecognizer
CFG = object()