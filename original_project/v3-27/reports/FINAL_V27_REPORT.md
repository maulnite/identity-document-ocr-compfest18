# Final OCR Report — V27

## Scope

This is the best validated pipeline checkpoint for the COMPfest Academy identity-document OCR project. It evaluates 632 labelled records. The official test submission is deliberately retained unchanged because it has different provenance from the benchmark pipeline.

## Final metrics

| Metric | V27 result |
|---|---:|
| Name exact match | 46.20% |
| Date-of-birth exact match | 69.62% |
| Address exact match (all records) | 83.70% |
| Macro field exact match | 66.51% |
| Full-record accuracy | 37.66% |
| CER (lower is better) | 1.3581 |
| Normalized edit similarity | 79.70% |

The all-record address score includes documents whose ground truth has no address. The operational address-only metric is therefore:

| Address subset | Images | Exact match | Similarity |
|---|---:|---:|---:|
| All records with an address | 112 | 25.0% | 86.93% |
| MyKad addresses | 100 | 28.0% | 89.5% |
| Generic-ID addresses | 12 | 0.0% | 31.4% |

## Results by document route

| Route | Labelled N | Name EM | Name similarity | DOB EM |
|---|---:|---:|---:|---:|
| MyKad | 100 | 63.0% | 95.5% | 100% |
| MRZ / passport | 182 | 46.7% | 70.2% | 71.4% |
| EU driving licence | 92 | 87.0% | 98.4% | 81.5% |
| Generic, unknown layout | 147 | 0.0% | 19.1% | 16.3% |
| Ukraine passport | 20 | 100% | 100% | 100% |
| HK mainland permit | 16 | 100% | 100% | 100% |
| Turkey ID | 9 | 100% | 100% | 100% |
| Czech ID | 20 | 95.0% | 96.1% | 100% |
| Chile ID | 20 | 0.0% | 91.7% | 100% |
| Spain ID | 9 | 0.0% | 91.7% | 100% |
| Portugal citizen card | 17 | 0.0% | 56.2% | 100% |

Exact-match scoring applies the project's standard output normalization, including casing and limited separator cleanup.

## Validated improvements included in V27

1. GPU OCR and peer consensus for Ukrainian passports (V24)
   - PP-OCRv6 Medium detector plus OpenOCR SVTRv2 Server recognizer on the A100.
   - The parser selects the Latin transliteration instead of the preceding Cyrillic OCR rendition.
   - Leave-one-out Ukraine-name validation: 0% to 100% exact match on 7 holdout images.

2. MyKad address-format gate (V25)
   - Repairs only the initial label/house-number separator, such as `NO.12` to `NO 12`; no alphanumeric OCR character is changed.
   - Group-disjoint holdout address exact match: 8% to 40% on 25 images.

3. MyKad name-span ranker (V26)
   - Scores candidate spans of one or two OCR lines using geometry, text shape, OCR confidence, and distance from the card-number line. Name text and ground truth are never inference features.
   - 1,244 candidate spans; group-disjoint holdout Name EM: 15.4% to 46.2%; similarity: 61.8% to 95.2%.

4. Fuzzy leave-one-out peer medoids (V27)
   - Applies only to repeated Portugal and Spain identities. A target scan never votes for itself.
   - Portugal holdout similarity: 52.0% to 64.3%.
   - Spain holdout similarity: 56.2% to 91.7%.
   - These improve character fidelity but not exact match, so the global EM is unchanged while normalized similarity rises from 79.34% to 79.70%.

## Deliberately rejected experiments

- Direct V17 MRZ candidate replacement was worse than the existing MRZ baseline.
- EU driving-licence fuzzy medoids provided no gain.
- The remaining MyKad address-span oracle had only 5.5% exact match after formatting, below the existing 28%; the residual errors are mainly OCR character errors, not span selection.
- A static Malaysian-place-name splitter gained only one record and was not released.

## Main remaining bottleneck

The 147 unknown `GENERIC_ID` documents are the primary limitation: Name EM is 0% and DOB EM is 16.3%. Their visual layouts lack a stable parser route. Further work should focus on template discovery/routing and targeted GPU re-OCR for this family, rather than changing the validated routes above.

## Final artifact and integrity

- Best result directory: `results_v27_fuzzy_peer_medoid_names/`
- Released benchmark file: `predictions_released.parquet` and `predictions_released.csv`
- Official test file: `results/test_predictions.csv`
- Test SHA-256: `d30030f546124d12c6c7e514b8dc702b3631dcf209f8b00a91532400aac60624`
- The copied V27 test file is byte-identical to the official test file.
