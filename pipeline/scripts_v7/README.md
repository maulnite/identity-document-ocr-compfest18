# V7 — address ROI geometry recovery

V7 does not rerun full-document OCR. It uses the V3-selected OCR layout to
identify an address region, fine-deskews it, enlarges it 2×, and produces raw,
grayscale, CLAHE, and gamma/unsharp variants. It reuses V3's already detected
text quadrilaterals, transforms them into the ROI, and recognizes their crops
across all variants together with CUDA SVTRv2 batches of 256.

This avoids a slow second detector pass and dedicates new work to the GPU
recognizer without rerunning full-document OCR. It writes a candidate output only. A disjoint identity
hold-out must improve non-empty address EM before candidate addresses are
released for test output.

```bash
.venv_ocr/bin/python scripts_v7/01_address_roi_geometry_ocr.py
```
