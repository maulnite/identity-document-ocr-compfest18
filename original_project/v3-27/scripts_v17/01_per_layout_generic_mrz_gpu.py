#!/usr/bin/env python3
"""V17: per-layout GPU field extraction for international IDs and MRZ passports.

V16 handles the multi-line address layout shared by Malaysian identity cards.
This stage handles the other document types without inventing an address where
none exists: it routes known Generic-ID layouts to small deterministic field
parsers (European driving licences, Czech/Turkish/Chilean/Spanish IDs) and
routes passports to the existing ICAO MRZ parser.  It re-recognises only the
field-bearing Generic lines and lower MRZ lines with the A100 SVTRv2 Server
recogniser.  GPU text is accepted only when it has strong variant consensus
and is valid for its field; each parser-field route must also improve an inner
group-held-out calibration set before it can reach the outer holdout.

Unknown Generic-ID layouts, MyKad/MyPR fields, and every address outside V16's
already validated Malaysian-card scope stay untouched.  The existing official
test submission is copied unchanged: it was produced by a different, stronger
baseline route and has no labels for a safe cross-route replacement.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parents[1]
V16_PATH = PROJECT / "scripts_v16" / "01_all_malaysian_address_cards_gpu.py"
V3_RESULTS = PROJECT / "results_v3_stable"
V16_RESULTS = PROJECT / "results_v16_all_malaysian_address_cards"
RESULTS = PROJECT / "results_v17_per_layout_generic_mrz"
SEED = 1808
GENERIC = "GENERIC_ID"
MRZ = "MRZ_DOCUMENT"
# The shared A100 can have less than half of its VRAM free.  SVTRv2 attention
# grows quickly with batch size, so this remains GPU-efficient while avoiding
# a 256-crop allocation that can exceed the remaining capacity.
GPU_BATCH = 64


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V16 = load_module(V16_PATH, "v16_v17_support")
V15 = V16.V15
V14 = V16.V14
V3 = V16.V3
V10 = V16.V10
core = V16.core


def text(value: Any) -> str:
    return core.output_text(value)


def compact(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", text(value).upper())


def clean_name(value: Any) -> str:
    value = re.sub(r"[^A-Za-zÀ-ÿ' -]", " ", text(value))
    return core.output_text(value)


def alpha_text(value: Any) -> bool:
    value = clean_name(value)
    letters = sum(char.isalpha() for char in value)
    digits = sum(char.isdigit() for char in text(value))
    return bool(letters >= 3 and digits == 0 and len(value) <= 80)


def normalize_date_extended(value: Any) -> str:
    """Parse field dates including European DD.MM.YY and compact DDMMYYYY."""
    raw = text(value)
    iso = re.search(r"(?<!\d)(\d{4})[./-](\d{1,2})[./-](\d{1,2})(?!\d)", raw)
    if iso:
        year, month, day = map(int, iso.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return ""
    match = re.search(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})(?!\d)", raw)
    if match:
        day, month, year = map(int, match.groups())
        if year < 100:
            year = 1900 + year if year >= 27 else 2000 + year
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return ""
    month_tokens = {
        "JAN": 1, "ENE": 1, "FEB": 2, "FEV": 2, "MAR": 3, "APR": 4,
        "ABR": 4, "MAY": 5, "MAI": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "AGO": 8, "SEP": 9, "SET": 9, "OCT": 10, "OKT": 10, "NOV": 11,
        "DEC": 12, "DIC": 12, "DEZ": 12,
    }
    textual = re.search(r"(?<![A-Z0-9])(\d{1,2})\s*([A-Z]{3})\s*(\d{2,4})(?![A-Z0-9])", raw.upper())
    if textual and textual.group(2) in month_tokens:
        day, token, year = textual.groups()
        year = int(year)
        if year < 100:
            year = 1900 + year if year >= 27 else 2000 + year
        try:
            return date(year, month_tokens[token], int(day)).isoformat()
        except ValueError:
            return ""
    for digits in re.findall(r"(?<!\d)(\d{8})(?!\d)", raw):
        day, month, year = int(digits[:2]), int(digits[2:4]), int(digits[4:])
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return core.normalize_date(raw)


def lines_from_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"ordinal": ordinal, "text": text(item.get("text", ""))} for ordinal, item in enumerate(items)]


def numbered_value(lines: list[dict[str, Any]], number: str) -> tuple[str, list[int]]:
    """Read an EU driving-licence numbered value, possibly on the next line."""
    # Detectors often merge the numbered label and its value (``2Anne`` or
    # ``328.02.64``) into one text line.  Accept that compact form as well as
    # the conventional ``2. Anne`` spelling; field-specific validation below
    # still rejects a non-name/non-date result.
    pattern = re.compile(rf"^\s*{re.escape(number)}\s*\.?\s*(.*)$", re.I)
    for position, line in enumerate(lines):
        match = pattern.match(line["text"])
        if not match:
            continue
        value = core.output_text(match.group(1).strip(" .:-"))
        if value:
            return value, [int(line["ordinal"])]
        for successor in lines[position + 1: position + 3]:
            successor_text = text(successor["text"])
            if successor_text:
                return successor_text, [int(successor["ordinal"])]
    return "", []


def next_text_after(lines: list[dict[str, Any]], token: str) -> tuple[str, list[int]]:
    token = compact(token)
    for position, line in enumerate(lines):
        if token not in compact(line["text"]):
            continue
        for successor in lines[position + 1: position + 4]:
            candidate = clean_name(successor["text"])
            candidate_key = compact(candidate)
            if alpha_text(candidate) and not any(bad in candidate_key for bad in ("DATE", "BIRTH", "NAMES", "SURNAME", "NATIONAL", "REPUBLIC", "IDENTITY", "CARD")):
                return candidate, [int(successor["ordinal"])]
    return "", []


def first_date(lines: list[dict[str, Any]], anchor: str | None = None) -> tuple[str, list[int]]:
    anchor_key = compact(anchor) if anchor else ""
    for position, line in enumerate(lines):
        if anchor_key and anchor_key not in compact(line["text"]):
            continue
        options = lines[position: position + 4] if anchor_key else [line]
        for option in options:
            value = normalize_date_extended(option["text"])
            if value:
                return value, [int(option["ordinal"])]
    return "", []


def person_value_lines(lines: list[dict[str, Any]], banned_tokens: tuple[str, ...]) -> list[tuple[str, int]]:
    """Return plausible multi-word personal-name lines, in document order."""
    values: list[tuple[str, int]] = []
    for line in lines:
        value = clean_name(line["text"])
        key = compact(value)
        words = value.split()
        if not (alpha_text(value) and 2 <= len(words) <= 5):
            continue
        if any(token in key for token in banned_tokens):
            continue
        values.append((value, int(line["ordinal"])))
    return values


def template_name(items: list[dict[str, Any]], family: str) -> str:
    if family == MRZ:
        return "MRZ_ICAO"
    joined = "\n".join(text(item.get("text", "")).upper() for item in items)
    if re.search(r"PRAWO|VOZACKA|FUHRER|FOHRER|AJOKORT|KORKORT|FORERKORT|DRIVING\s+LICEN[CS]E", joined):
        return "EU_DRIVING_LICENSE"
    if re.search(r"CARTAO\s+DE\s+CIDAD|CITIZEN\s+CARD", joined):
        return "PORTUGAL_CITIZEN_CARD"
    if re.search(r"\u6e2f\u6fb3\u5c45\u6c11|TRAVEL\s*TO\s*THE\s*MAINLAND", joined):
        return "HK_MAINLAND_PERMIT"
    if re.search(r"UKRAINE|YKPAIHA", joined):
        return "UKRAINE_PASSPORT"
    if re.search(r"CESKA|CZECH REPUBLIC|OBCANSKY", joined):
        return "CZECH_ID"
    if "TURKIYE" in joined:
        return "TURKEY_ID"
    if re.search(r"CHILE|CEDULA", joined):
        return "CHILE_ID"
    if re.search(r"ESPANA|DOCUMENTO NACIONAL", joined):
        return "SPAIN_ID"
    return "OTHER_GENERIC"


def parse_generic_template(items: list[dict[str, Any]]) -> dict[str, Any]:
    lines = lines_from_items(items)
    joined = "\n".join(line["text"].upper() for line in lines)
    template = template_name(items, GENERIC)
    result = {"template": template, "name": "", "birth_date": "", "name_ids": [], "dob_ids": []}
    if template == "EU_DRIVING_LICENSE":
        surname, surname_ids = numbered_value(lines, "1")
        given, given_ids = numbered_value(lines, "2")
        dob_raw, dob_ids = numbered_value(lines, "3")
        if not (alpha_text(surname) and alpha_text(given)):
            person_lines = person_value_lines(lines, (
                "DRIVING", "LICENCE", "LICENSE", "PERMIS", "CONDUIRE", "AJOKORT",
                "KORKORT", "FORERKORT", "AUTOILIJA", "VIRANOMAINEN", "STATENS",
                "VEGVESEN", "SUOMI", "FINLAND", "NORGE", "SVERIGE",
            ))
            if person_lines:
                combined, ordinal = person_lines[0]
                surname, given = "", combined
                surname_ids, given_ids = [], [ordinal]
        dob = normalize_date_extended(dob_raw)
        if not dob:
            dob, dob_ids = first_date(lines)
        result.update({
            "name": core.output_text(f"{clean_name(given)} {clean_name(surname)}"),
            "birth_date": dob,
            "name_ids": given_ids + surname_ids,
            "dob_ids": dob_ids,
        })
    elif template == "PORTUGAL_CITIZEN_CARD":
        person_lines = person_value_lines(lines, (
            "CARTAO", "CITIZEN", "PORTUGAL", "CARD", "NACIONAL", "ASSINATURA",
        ))
        surname, surname_id = (person_lines[0] if person_lines else ("", -1))
        given, given_id = (person_lines[1] if len(person_lines) > 1 else ("", -1))
        dob, dob_ids = first_date(lines)
        result.update({
            "name": core.output_text(f"{given} {surname}"), "birth_date": dob,
            "name_ids": [ordinal for ordinal in (given_id, surname_id) if ordinal >= 0], "dob_ids": dob_ids,
        })
    elif template == "HK_MAINLAND_PERMIT":
        name, name_ids = "", []
        for line in lines:
            match = re.fullmatch(r"\s*([A-Z]{2,6})\s*,?\s*([A-Z]{5,12})\s*", line["text"].upper())
            if not match:
                continue
            surname, given = match.groups()
            given = f"{given[:3]} {given[3:]}" if len(given) == 6 else given
            name, name_ids = core.output_text(f"{surname}, {given}"), [int(line["ordinal"])]
            break
        dob, dob_ids = first_date(lines)
        result.update({"name": name, "birth_date": dob, "name_ids": name_ids, "dob_ids": dob_ids})
    elif template == "UKRAINE_PASSPORT":
        surname, surname_ids, given, given_ids = "", [], "", []
        for position, line in enumerate(lines):
            if "NAME" not in compact(line["text"]):
                continue
            for previous in reversed(lines[max(0, position - 3):position]):
                value = clean_name(previous["text"])
                if alpha_text(value) and len(value.split()) == 1 and 4 <= len(value) <= 20:
                    surname, surname_ids = value, [int(previous["ordinal"])]
                    break
            # Ukrainian passports print each personal-name field twice: first
            # in Cyrillic, then in its Latin transliteration.  The recognizer
            # commonly renders Cyrillic lookalikes as Latin characters (for
            # example, ``MAP'RHA``), while the next line holds ``MARIANA``.
            # Retain all short one-word candidates in the small structural
            # window and deliberately select the later transliteration rather
            # than the visually similar Cyrillic rendition.
            following_values: list[tuple[str, int]] = []
            for following in lines[position + 1: position + 4]:
                value = clean_name(following["text"])
                if alpha_text(value) and len(value.split()) == 1 and 3 <= len(value) <= 20:
                    following_values.append((value, int(following["ordinal"])))
            if following_values:
                given, given_ordinal = following_values[-1]
                given_ids = [given_ordinal]
            break
        dob, dob_ids = first_date(lines, "BIRTH")
        if not dob:
            dob, dob_ids = first_date(lines)
        result.update({"name": core.output_text(f"{given} {surname}"), "birth_date": dob, "name_ids": given_ids + surname_ids, "dob_ids": dob_ids})
    elif template == "CZECH_ID":
        surname, surname_ids = next_text_after(lines, "PRUMENI")
        if not surname:
            surname, surname_ids = next_text_after(lines, "SURNAME")
        given, given_ids = next_text_after(lines, "JMENO")
        # Some cards hold the given name after the English label on the same row.
        for line in lines:
            match = re.search(r"GIVEN\s*NAMES?\s+(.+)", line["text"], re.I)
            inline = clean_name(match.group(1)) if match else ""
            if alpha_text(inline):
                given, given_ids = inline, [int(line["ordinal"])]
                break
        dob, dob_ids = first_date(lines, "NAROZEN")
        if not dob:
            dob, dob_ids = first_date(lines, "DATE OF BIRTH")
        result.update({"name": core.output_text(f"{given} {surname}"), "birth_date": dob, "name_ids": given_ids + surname_ids, "dob_ids": dob_ids})
    elif template == "TURKEY_ID":
        start = 0
        for position, line in enumerate(lines):
            if re.search(r"\d{11}", re.sub(r"\D", "", line["text"])):
                start = position + 1
                break
        values: list[tuple[str, int]] = []
        for line in lines[start:]:
            value = clean_name(line["text"])
            value_key = compact(value)
            if alpha_text(value) and value_key not in {"TURKIYECUMHURIYETIKIMLIKKARTI", "REPUBLICOFTURKEYIDENTITYCARD"}:
                values.append((value, int(line["ordinal"])))
        dob, dob_ids = first_date(lines)
        if len(values) >= 2:
            surname, surname_id = values[0]
            given, given_id = values[1]
            result.update({"name": core.output_text(f"{given} {surname}"), "name_ids": [given_id, surname_id]})
        result.update({"birth_date": dob, "dob_ids": dob_ids})
    elif template == "CHILE_ID":
        surnames: list[tuple[str, int]] = []
        for position, line in enumerate(lines):
            if "APELLID" in compact(line["text"]):
                for successor in lines[position + 1: position + 3]:
                    value = clean_name(successor["text"])
                    if alpha_text(value):
                        surnames.append((value, int(successor["ordinal"])))
                break
        dob, dob_ids = first_date(lines, "FECHA")
        result.update({"name": core.output_text(" ".join(value for value, _ in surnames)), "birth_date": dob, "name_ids": [ordinal for _, ordinal in surnames], "dob_ids": dob_ids})
    elif template == "SPAIN_ID":
        surnames: list[tuple[str, int]] = []
        given, given_ids = "", []
        for position, line in enumerate(lines):
            token = compact(line["text"])
            if "APELLID" in token:
                for successor in lines[position + 1: position + 3]:
                    value = clean_name(successor["text"])
                    if alpha_text(value):
                        surnames.append((value, int(successor["ordinal"])))
            if "NOMBR" in token and position + 1 < len(lines) and not given:
                value = clean_name(lines[position + 1]["text"])
                if alpha_text(value):
                    given, given_ids = value, [int(lines[position + 1]["ordinal"])]
        # Common Spanish-DNI OCR failure: ``NOMBRE`` becomes e.g. ``NOKERE``.
        # The layout still places the given name immediately after the two
        # APELLIDOS lines, so use that constrained structural fallback rather
        # than accepting an unconstrained text line elsewhere on the card.
        if not given and surnames:
            last_surname = max(ordinal for _, ordinal in surnames)
            after_surnames = [line for line in lines if int(line["ordinal"]) > last_surname]
            for index, line in enumerate(after_surnames):
                value = clean_name(line["text"])
                key = compact(value)
                if not alpha_text(value):
                    continue
                # Ignore isolated non-text detector noise between the label,
                # its value, and NACIONALIDAD (e.g. a single symbol box).
                later_alpha = [
                    (candidate, clean_name(candidate["text"]))
                    for candidate in after_surnames[index + 1:]
                    if alpha_text(clean_name(candidate["text"]))
                ]
                following, following_value = later_alpha[0] if later_alpha else (None, "")
                following_key = compact(following_value)
                after_following_key = compact(later_alpha[1][1]) if len(later_alpha) > 1 else ""
                label_like = len(key) <= 8 and key.startswith("NO") and key.endswith("RE")
                # Even heavily corrupted NAME labels remain a short word
                # immediately followed by the actual name and then the
                # NACIONALIDAD label.  This distinguishes them from a real
                # short first name when the label itself was missed.
                structural_label = bool(
                    len(key) <= 8 and alpha_text(following_value) and "NACI" in after_following_key
                )
                if (label_like or structural_label) and alpha_text(following_value) and following_key not in {"NACIONALIDAD", "ESP"}:
                    given, given_ids = following_value, [int(following["ordinal"])]
                    break
                if not label_like and key not in {"NACIONALIDAD", "ESP"}:
                    given, given_ids = value, [int(line["ordinal"])]
                    break
        dob, dob_ids = first_date(lines, "FECHA")
        if not dob:
            # The first compact date in a DNI reading is the birth date; the
            # later date is the document-expiry field.
            dob, dob_ids = first_date(lines)
        result.update({"name": core.output_text(" ".join([given] + [value for value, _ in surnames])), "birth_date": dob, "name_ids": given_ids + [ordinal for _, ordinal in surnames], "dob_ids": dob_ids})
    return result


def image_size(row: Any) -> tuple[int, int]:
    image = cv2.imread(str(getattr(row, "input_path", "")), cv2.IMREAD_COLOR)
    if image is None:
        return 1, 1
    height, width = image.shape[:2]
    return width, height


def initial_targets(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        family = str(row.document_family)
        if family == GENERIC:
            parsed = parse_generic_template(V10.V8.read_items(row.ocr_items))
            if parsed["template"] != "OTHER_GENERIC":
                targets[str(row.filename)] = parsed
        elif family == MRZ:
            targets[str(row.filename)] = {"template": "MRZ_ICAO", "name_ids": [], "dob_ids": []}
    return targets


def collect_gpu_jobs(frame: pd.DataFrame, targets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    source: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        filename = str(row.filename)
        target = targets.get(filename)
        if target is None:
            continue
        image = cv2.imread(str(getattr(row, "input_path", "")), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        name_ids = set(target.get("name_ids", []))
        dob_ids = set(target.get("dob_ids", []))
        for ordinal, item in enumerate(V10.V8.read_items(row.ocr_items)):
            box = V10.V8.item_box(item)
            if box is None:
                continue
            role = ""
            if str(row.document_family) == GENERIC:
                if ordinal in name_ids:
                    role = "generic_name"
                elif ordinal in dob_ids:
                    role = "generic_dob"
            else:
                center_y = float(box.mean(axis=0)[1]) / max(1, height)
                item_text = text(item.get("text", ""))
                if item_text.count("<") >= 2 or center_y >= 0.58:
                    role = "mrz_line"
            if not role:
                continue
            try:
                crop = core.perspective_crop(image, box)
            except Exception:
                continue
            if min(crop.shape[:2]) < 4:
                continue
            key = (filename, ordinal)
            source[key] = {
                "filename": filename,
                "ordinal": ordinal,
                "role": role,
                "template": target["template"],
                "old_text": text(item.get("text", "")),
                "old_confidence": float(item.get("confidence", 0.0)),
            }
            for variant, enhanced in V14.crop_variants(crop).items():
                jobs.append({"filename": filename, "ordinal": ordinal, "variant": variant, "crop": enhanced})
    return jobs, source


def role_valid(role: str, value: str) -> bool:
    if role == "generic_name":
        return alpha_text(value)
    if role == "generic_dob":
        return bool(normalize_date_extended(value))
    if role == "mrz_line":
        normalized = re.sub(r"[^A-Z0-9<]", "", value.upper())
        return bool(normalized.count("<") >= 2 and len(normalized) >= 25)
    return False


def gpu_reocr(frame: pd.DataFrame, targets: dict[str, dict[str, Any]]) -> pd.DataFrame:
    jobs, source = collect_gpu_jobs(frame, targets)
    print(json.dumps({"stage": "gpu_field_reocr_prepared", "lines": len(source), "crops": len(jobs), "batch": GPU_BATCH}, ensure_ascii=False), flush=True)
    if not jobs:
        return pd.DataFrame(columns=["filename", "ordinal", "role", "template", "old_text", "old_confidence", "new_text", "new_confidence", "chosen_variant", "variant_agreement", "raw_accepted", "accepted"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    engine = V3.PPOCRSVTRv2Server()
    started = time.perf_counter()
    try:
        for start in range(0, len(jobs), GPU_BATCH):
            chunk = jobs[start:start + GPU_BATCH]
            predictions = engine.recognizer.predict_batch([job["crop"] for job in chunk], batch_size=GPU_BATCH)
            for job, prediction in zip(chunk, predictions):
                grouped[(job["filename"], job["ordinal"])].append({"variant": job["variant"], "text": text(prediction.get("text", "")), "confidence": float(prediction.get("confidence", 0.0))})
            print(f"GPU per-layout re-OCR crops: {min(start + len(chunk), len(jobs))}/{len(jobs)}", flush=True)
    finally:
        engine.close()
    elapsed = time.perf_counter() - started
    records: list[dict[str, Any]] = []
    for key, original in source.items():
        chosen = V14.choose_text(original["old_text"], original["old_confidence"], grouped.get(key, []))
        raw_accepted = bool(chosen["accepted"])
        spacing_safe = V15.safe_spacing_repair(original["old_text"], chosen["text"])
        semantic_safe = role_valid(original["role"], chosen["text"])
        records.append({
            **original,
            "new_text": chosen["text"], "new_confidence": chosen["confidence"],
            "chosen_variant": chosen["variant"], "variant_agreement": chosen["agreement"],
            "raw_accepted": raw_accepted, "spacing_safe": spacing_safe, "semantic_safe": semantic_safe,
            "accepted": raw_accepted and (spacing_safe or semantic_safe), "gpu_elapsed_seconds": elapsed,
        })
    return pd.DataFrame(records)


def missing_generic_targets(frame: pd.DataFrame, targets: dict[str, dict[str, Any]], manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return newly-selected Generic field lines not already represented in cache."""
    existing = {
        (str(row.filename), int(row.ordinal))
        for row in manifest[["filename", "ordinal"]].itertuples(index=False)
    }
    families = frame.set_index("filename").document_family.to_dict()
    output: dict[str, dict[str, Any]] = {}
    for filename, target in targets.items():
        if str(families.get(filename, "")) != GENERIC:
            continue
        name_ids = [int(ordinal) for ordinal in target.get("name_ids", []) if (filename, int(ordinal)) not in existing]
        dob_ids = [int(ordinal) for ordinal in target.get("dob_ids", []) if (filename, int(ordinal)) not in existing]
        if name_ids or dob_ids:
            output[filename] = {**target, "name_ids": name_ids, "dob_ids": dob_ids}
    return output


def updated_items(items: list[dict[str, Any]], filename: str, manifest: pd.DataFrame) -> list[dict[str, Any]]:
    replacements = manifest.loc[manifest.accepted.astype(bool)].set_index(["filename", "ordinal"])
    output: list[dict[str, Any]] = []
    for ordinal, item in enumerate(items):
        updated = dict(item)
        key = (filename, ordinal)
        if key in replacements.index:
            replacement = replacements.loc[key]
            updated["text"] = text(replacement.new_text)
            updated["confidence"] = float(replacement.new_confidence)
        output.append(updated)
    return output


def canonical_box_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert cached Arrow/NumPy quadrilaterals to core's plain 4x2 format.

    The selected V3 OCR cache contains a small number of nested Arrow arrays.
    V8's box reader already handles them for cropping, but the legacy MRZ graph
    parser expects a homogeneous Python list.  Canonicalising here preserves
    exactly the same geometry and simply drops malformed boxes.
    """
    output: list[dict[str, Any]] = []
    for item in items:
        box = V10.V8.item_box(item)
        if box is None:
            continue
        canonical = dict(item)
        canonical["bbox"] = np.asarray(box, dtype=float).reshape(4, 2).tolist()
        output.append(canonical)
    return output


def build_proposals(frame: pd.DataFrame, targets: dict[str, dict[str, Any]], manifest: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        filename = str(row.filename)
        target = targets.get(filename)
        if target is None:
            continue
        items = updated_items(V10.V8.read_items(row.ocr_items), filename, manifest)
        if str(row.document_family) == GENERIC:
            parsed = parse_generic_template(items)
            records.append({"filename": filename, "route": parsed["template"], "name_candidate": parsed["name"], "dob_candidate": parsed["birth_date"]})
        elif str(row.document_family) == MRZ:
            width, height = image_size(row)
            parsed = core.extract_fields(canonical_box_items(items), width, height, mode="full")
            checksum = bool(parsed.get("mrz_checksum_success", False))
            records.append({
                "filename": filename, "route": "MRZ_ICAO", "name_candidate": parsed.get("name", "") if checksum else "",
                "dob_candidate": parsed.get("birth_date", "") if checksum else "",
                "mrz_checksum_success_candidate": checksum,
            })
    return pd.DataFrame(records)


def field_metrics(frame: pd.DataFrame, filenames: set[str], field: str) -> dict[str, Any]:
    subset = frame.loc[frame.filename.isin(filenames) & frame.has_ground_truth.astype(bool)].copy()
    if subset.empty:
        return {"images": 0, "exact_match": None, "mean_similarity": None}
    prefix = "name" if field == "name" else "dob"
    return {"images": int(len(subset)), "exact_match": float(subset[f"{prefix}_exact"].mean()), "mean_similarity": float(subset[f"{prefix}_similarity"].mean())}


def derive_gates(base: pd.DataFrame, proposals: pd.DataFrame, calibration_files: set[str]) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    for route in sorted(proposals.route.unique()):
        route_rows = proposals.loc[proposals.route.eq(route)]
        for field, candidate_column in (("name", "name_candidate"), ("dob", "dob_candidate")):
            filenames = set(route_rows.loc[route_rows.filename.isin(calibration_files) & route_rows[candidate_column].map(lambda value: bool(text(value))), "filename"])
            before = field_metrics(base, filenames, field)
            trial = base.copy()
            values = route_rows.set_index("filename")[candidate_column]
            prediction_column = "name_pred" if field == "name" else "birth_date_pred"
            target = trial.filename.isin(filenames)
            trial.loc[target, prediction_column] = trial.loc[target, "filename"].map(values).fillna("")
            trial = core.score_prediction_rows(trial)
            after = field_metrics(trial, filenames, field)
            safe = bool(
                before["images"] >= 3
                and after["exact_match"] >= before["exact_match"]
                and after["mean_similarity"] >= before["mean_similarity"] + 0.04
            )
            gates[f"{route}:{field}"] = {"route": route, "field": field, "safe": safe, "baseline": before, "candidate": after}
    return gates


def apply_gates(base: pd.DataFrame, proposals: pd.DataFrame, gates: dict[str, dict[str, Any]]) -> pd.DataFrame:
    output = base.copy()
    indexed = proposals.set_index("filename")
    for field, candidate_column, prediction_column in (("name", "name_candidate", "name_pred"), ("dob", "dob_candidate", "birth_date_pred")):
        allowed_routes = {gate["route"] for gate in gates.values() if gate["field"] == field and gate["safe"]}
        permitted = proposals.loc[proposals.route.isin(allowed_routes) & proposals[candidate_column].map(lambda value: bool(text(value)))]
        if permitted.empty:
            continue
        values = permitted.set_index("filename")[candidate_column]
        mask = output.filename.isin(values.index)
        output.loc[mask, prediction_column] = output.loc[mask, "filename"].map(values).fillna("")
    return core.score_prediction_rows(output)


def family_summary(frame: pd.DataFrame, filenames: set[str]) -> dict[str, Any]:
    return {"name": field_metrics(frame, filenames, "name"), "dob": field_metrics(frame, filenames, "dob")}


def save_predictions(name: str, frame: pd.DataFrame) -> None:
    V14.atomic_parquet(RESULTS / f"{name}.parquet", frame)
    V14.atomic_csv(RESULTS / f"{name}.csv", frame.drop(columns=[column for column in frame if column.endswith("_detail")], errors="ignore"))


def run() -> None:
    source = pd.read_parquet(V3_RESULTS / "predictions_base.parquet").sort_values("filename", key=lambda value: value.map(V3.natural_key)).reset_index(drop=True)
    base = pd.read_parquet(V16_RESULTS / "predictions_released.parquet").sort_values("filename", key=lambda value: value.map(V3.natural_key)).reset_index(drop=True)
    if not source.filename.equals(base.filename):
        raise RuntimeError("V3/V16 filename ordering mismatch")
    # ``predictions_base`` intentionally stores only selected text, while the
    # detector quadrilaterals live in V3's selected OCR records.  Reattach the
    # exact records so every crop still uses the same PP-OCRv6 detector boxes.
    selected_ocr = V10.V8.select_ocr_rows(source)
    source = source.copy()
    source["ocr_items"] = source.filename.map(lambda filename: selected_ocr.get(str(filename), {}).get("ocr_items", []))
    source["input_path"] = source.filename.map(lambda filename: selected_ocr.get(str(filename), {}).get("input_path", ""))
    # Some parquet deserializations expose OCR rows as NumPy arrays, whose
    # direct truth value is deliberately ambiguous.  Size is stable for lists,
    # arrays, and JSON-like values alike.
    if not source.ocr_items.map(lambda items: len(items) if hasattr(items, "__len__") else 0).gt(0).any():
        raise RuntimeError("Could not recover selected V3 OCR items")
    targets = initial_targets(source)
    template_manifest = pd.DataFrame([{"filename": filename, "route": target["template"], "name_line_count": len(target.get("name_ids", [])), "dob_line_count": len(target.get("dob_ids", []))} for filename, target in targets.items()])
    V14.atomic_csv(RESULTS / "template_manifest.csv", template_manifest)
    print(json.dumps({"stage": "layout_routing", "documents": len(targets), "routes": template_manifest.route.value_counts().to_dict()}, ensure_ascii=False), flush=True)

    gpu_cache = RESULTS / "gpu_field_lines.csv"
    if gpu_cache.exists():
        gpu_manifest = pd.read_csv(gpu_cache)
        print(json.dumps({"stage": "gpu_field_reocr_cache", "lines": int(len(gpu_manifest))}, ensure_ascii=False), flush=True)
        incremental_targets = missing_generic_targets(source, targets, gpu_manifest)
        if incremental_targets:
            incremental = gpu_reocr(source, incremental_targets)
            if len(incremental):
                gpu_manifest = pd.concat([gpu_manifest, incremental], ignore_index=True).drop_duplicates(["filename", "ordinal"], keep="last")
                V14.atomic_csv(gpu_cache, gpu_manifest)
            print(json.dumps({"stage": "gpu_field_reocr_incremental", "documents": len(incremental_targets), "new_lines": int(len(incremental))}, ensure_ascii=False), flush=True)
    else:
        gpu_manifest = gpu_reocr(source, targets)
        V14.atomic_csv(gpu_cache, gpu_manifest)
    print(json.dumps({"stage": "gpu_field_guard", "raw_accepted": int(gpu_manifest.raw_accepted.astype(bool).sum()), "accepted": int(gpu_manifest.accepted.astype(bool).sum())}, ensure_ascii=False), flush=True)
    proposals = build_proposals(source, targets, gpu_manifest)
    V14.atomic_csv(RESULTS / "field_proposals.csv", proposals)

    labelled = base.loc[base.has_ground_truth.astype(bool) & base.document_family.isin([GENERIC, MRZ])].copy()
    groups = {row.filename: V3.identity_key(row) for _, row in labelled.iterrows()}
    outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_positions, validation_positions = next(outer.split(labelled, groups=[groups[name] for name in labelled.filename]))
    train_files = set(labelled.iloc[train_positions].filename)
    validation_files = set(labelled.iloc[validation_positions].filename)
    ordered_train = sorted(train_files)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    _, calibration_positions = next(inner.split(ordered_train, groups=[groups[name] for name in ordered_train]))
    calibration_files = {ordered_train[position] for position in calibration_positions}
    gates = derive_gates(base, proposals, calibration_files)
    candidate = apply_gates(base, proposals, gates)
    V14.atomic_json(RESULTS / "template_gates.json", gates)

    validation_baseline = family_summary(base, validation_files)
    validation_candidate = family_summary(candidate, validation_files)
    generic_validation = set(base.loc[base.filename.isin(validation_files) & base.document_family.eq(GENERIC), "filename"])
    mrz_validation = set(base.loc[base.filename.isin(validation_files) & base.document_family.eq(MRZ), "filename"])
    def non_regressing(before: dict[str, Any], after: dict[str, Any]) -> bool:
        return all(
            after[field]["exact_match"] >= before[field]["exact_match"] and after[field]["mean_similarity"] >= before[field]["mean_similarity"]
            for field in ("name", "dob") if before[field]["images"]
        )
    generic_before = family_summary(base, generic_validation)
    generic_after = family_summary(candidate, generic_validation)
    mrz_before = family_summary(base, mrz_validation)
    mrz_after = family_summary(candidate, mrz_validation)
    generic_gain = (generic_after["name"]["mean_similarity"] - generic_before["name"]["mean_similarity"]) + (generic_after["dob"]["mean_similarity"] - generic_before["dob"]["mean_similarity"])
    release = bool(non_regressing(generic_before, generic_after) and non_regressing(mrz_before, mrz_after) and generic_gain >= 0.04)
    validation = {
        "scope": "Generic-ID template fields and checksum-validated MRZ fields; V16 address output retained",
        "train_images": len(train_files), "calibration_images": len(calibration_files), "validation_images": len(validation_files),
        "template_gates": gates,
        "overall_baseline": validation_baseline, "overall_candidate": validation_candidate,
        "generic_baseline": generic_before, "generic_candidate": generic_after,
        "mrz_baseline": mrz_before, "mrz_candidate": mrz_after,
        "test_release_gate_passed": release,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    released = candidate if release else base.copy()
    save_predictions("predictions_candidate", candidate)
    save_predictions("predictions_released", released)
    V14.atomic_json(RESULTS / "validation.json", validation)
    metrics = V3.field_metrics(released)
    metrics.update({
        "routed_documents": int(len(targets)),
        "gpu_line_candidates": int(len(gpu_manifest)),
        "gpu_accepted_lines": int(gpu_manifest.accepted.astype(bool).sum()),
        "safe_template_field_routes": int(sum(gate["safe"] for gate in gates.values())),
        "test_release_gate_passed": release,
        "official_submission_overwritten": False,
        "released_generic": family_summary(released, set(released.loc[released.document_family.eq(GENERIC), "filename"])),
        "candidate_generic": family_summary(candidate, set(candidate.loc[candidate.document_family.eq(GENERIC), "filename"])),
        "released_mrz": family_summary(released, set(released.loc[released.document_family.eq(MRZ), "filename"])),
        "candidate_mrz": family_summary(candidate, set(candidate.loc[candidate.document_family.eq(MRZ), "filename"])),
    })
    V14.atomic_json(RESULTS / "metrics.json", metrics)
    # Do not cross-replace test rows produced by the official baseline with V3
    # provenance.  This experiment publishes its benchmark candidate separately.
    shutil.copy2(PROJECT / "results" / "test_predictions.csv", RESULTS / "test_predictions.csv")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pipeline": "V17 per-layout Generic/MRZ GPU field re-OCR", "gpu_required": True, "official_results_untouched": True}, ensure_ascii=False), flush=True)
    run()
