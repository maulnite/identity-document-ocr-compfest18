# V28 Generic-ID Bottleneck Experiment

## Outcome

V28 is a validated downstream improvement over V27 for documents that remain `GENERIC_ID` with no V17 route. It uses only existing visual-identity groups and prior OCR predictions; no ground-truth text is used during inference and no image/OCR model is rerun.

| Metric | V27 | V28 | Change |
|---|---:|---:|---:|
| Name exact match | 46.20% | 46.20% | 0.00 pp |
| DOB exact match | 69.62% | 70.41% | +0.79 pp |
| Macro Field EM | 66.51% | 66.77% | +0.26 pp |
| Full-record accuracy | 37.66% | 37.66% | 0.00 pp |
| CER | 1.3581 | 1.3495 | lower by 0.0086 |
| Normalized edit similarity | 79.70% | 81.28% | +1.58 pp |

For the 147 labelled route-unknown Generic-ID documents: name similarity increases from 19.05% to 28.22%; DOB exact match rises from 16.33% to 19.73%; DOB similarity rises from 29.93% to 41.16%.

## Method

- Build candidates only inside the pre-existing visual identity group.
- Exclude the target document from its peer set (leave-one-out).
- Name: fuzzy medoid of peer name predictions, requiring at least two non-empty peers.
- DOB: normalized-date medoid of peer DOB predictions, requiring at least three valid peers.
- Address is intentionally unchanged: only 12 labelled unknown Generic-ID records have an address, and peer evidence was insufficient.

## Validation gate

The 147 labelled unknown Generic-ID records use a deterministic document split: 49 calibration and 98 holdout records.

- Name gate: calibration similarity 19.62% -> 29.70%; holdout 18.77% -> 27.48%; exact match does not regress.
- DOB gate: calibration EM 12.24% -> 16.33%, similarity 26.12% -> 36.73%; holdout EM 18.37% -> 21.43%, similarity 31.84% -> 43.37%.

Both gates pass. The produced `test_predictions.csv` is only a byte-identical copy of the official V27 test release; the official submission was not overwritten.

## Limitation and next requirement

The line-level OCR oracle finds an exact name in only 1 of 147 unknown Generic-ID documents. This is primarily a template-routing and recognition problem, not a text-consensus problem. A larger jump requires recovering raw images and an OCR runtime, then clustering visual templates and performing targeted field OCR with deskew/CLAHE/re-recognition.