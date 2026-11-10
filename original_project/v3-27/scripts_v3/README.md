# V3: SVTRv2 Server identity-group pipeline

`01_ppocrv5_identity_consensus.py` is a clean experiment separate from V1/V2.
It uses **PP-OCRv6 Medium** for CPU text detection and the larger **OpenOCR SVTRv2
Server** model for recognition on `gpu:0`; it does not load a VLM. Keeping the
detector on CPU avoids the ONNX Runtime/PyTorch CUDA allocator conflict observed
on this host, while the larger recognition model still uses the A100. It benchmarks
three already-materialized inputs (`raw`, `gray`, and `gray_clahe`) on all 732
images, routes each quality group using labeled benchmark results, then trains a
Siamese contrastive linker from repeated labeled identities.

At inference, test documents are clustered only with other test documents.
Ground-truth fields are never read as features or copied to predictions.  The
clusterer uses complete linkage rather than transitive connected components, so
every cross-pair in a group must meet the calibrated threshold and contradiction
checks.  Field consensus requires support from at least two images and 60% of a
cluster.

Run on the A100:

```bash
cd /home/nafisnaufal1426/adit/compfest_academy_project
.venv_ocr/bin/python scripts_v3/01_ppocrv5_identity_consensus.py --stage all \
  > results_v3_stable/v3_run.log 2>&1 &
echo $! > results_v3_stable/v3_run.pid
```

Results stay under `results_v3_stable/`.  In particular,
`test_predictions_ppocrv6_svtrv2_group_consensus.csv` is a candidate submission; the
pipeline never replaces `results/test_predictions.csv` automatically.
