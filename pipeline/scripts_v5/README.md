# V5: YOLO-face + ArcFace optional identity evidence

V5 is a separate experiment that augments completed V3/V4 artifacts. It uses
YOLOv8n-Face to detect and crop the portrait on a rectified identity card, then
uses ArcFace (`buffalo_l` recognition model) to produce a face vector. The crop
is held only in memory and is never exported. A face must meet confidence,
minimum-size, and blur checks; otherwise V5 sets `face_available=false` and
falls back to V4 OCR/document embeddings.

Face vectors are an additional feature for a held-out calibrated Siamese linker,
not an automatic match. Final grouping still requires fusion threshold,
contradiction checks, and complete linkage. Outputs remain in
`results_v5_face/` and never overwrite `results/test_predictions.csv`.

The InsightFace `buffalo_l` model pack is designated for non-commercial
research by its publisher. Confirm its license before any use beyond this
research experiment.
