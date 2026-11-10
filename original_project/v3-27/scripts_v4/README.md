# V4: DINOv2 visual identity grouping

`01_dinov2_visual_identity_linker.py` starts from the completed V3 OCR base
predictions. It computes a DINOv2 visual embedding for each rectified document,
uses a high-recall visual-similarity threshold calibrated on labelled training
identities only as grouping candidates, and trains a
Siamese verifier on held-out labelled identity groups. The verifier fuses image
embedding and OCR/field features; complete linkage, contradiction checks, and
multi-image consensus prevent a visually similar card template from becoming an
automatic identity match.

DINOv2 is a self-supervised vision encoder, not a VLM. Its initial checkpoint
download is cached on the A100. V4 writes only to `results_v4_visual_linker/`
and will not replace `results/test_predictions.csv`.

Run after `results_v3_stable/predictions_base.parquet` exists:

```bash
cd /home/nafisnaufal1426/adit/compfest_academy_project
PYTHONUNBUFFERED=1 .venv_ocr/bin/python scripts_v4/01_dinov2_visual_identity_linker.py \
  > results_v4_visual_linker/v4_run.log 2>&1 &
```
