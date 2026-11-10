# Panduan seluruh isi arsip

ZIP utama: `FINAL PROJECT DSA COMPFEST 18.zip` (386.952.264 bytes). Arsip berisi 748 entries termasuk directory entries, dengan 395.502.559 bytes uncompressed sebelum nested archives diekstrak.

## Struktur sumber

| Lokasi asli | Isi | Peran |
|---|---|---|
| `dataset/images/` | 732 JPG | Raw captures untuk OCR dan image-quality analysis |
| `dataset/ground_truth.csv` | 632 rows | Label tiga target fields; mempunyai masalah CSV quoting |
| `dataset/ground_truth.xlsx` | Satu worksheet, 632 data rows | Spreadsheet raw labels; digunakan notebook EDA |
| `GUIDELINE FINAL PROJECT DSA COMPFEST 18.pdf` | 4 halaman | Brief proyek, deliverables, dan rubrik penilaian |
| `additional/Business_Analysis_ketolak pacil_COMPFEST18.pdf` | 3 halaman | Business understanding, impact, objective, insights, recommendations |
| `additional/Ketolak_Pacil_Final_Pipeline_Driven_EDA_NaturalVisual_EXECUTED.ipynb` | 80 cells | EDA raw data dan pipeline-assisted diagnostics |
| `additional/Ketolak_Pacil_OCR_Identity_Similarity_Ablation_PPvsSVTR.ipynb` | 45 cells | Diagnostic comparison hybrid OCR vs PP-OCRv6 end-to-end |
| `output/Notebook_ketolak pacil_COMPFEST18.ipynb` | 114 cells | Final narrative, frozen evidence, metrics, test release |
| `output/PPT_ketolak pacil_COMPFEST18.pdf` | 25 halaman | 12 slide utama dan appendix mulai halaman 13 |
| `v3-27.zip` | 99 files | Historical scripts, reports, dan Python bytecode |
| `v27-28.zip` | 54 files | Portable reproduction bundle, frozen inputs, V27/V28 results |
| `[MENTEE] Data Science Academy.png` | Satu PNG | Visual academy; tidak dipakai sebagai branding repository |

## Notebook final dan arsip tersembunyi

Satu code cell notebook final menyimpan ZIP sebagai string `_BUNDLE_B64`. Bundle ini memuat **33 files**, dengan 10.005.812 bytes uncompressed: prediction tables, OCR tables, selected source, reports, development timeline, source inventory, tujuh visual examples, dan `MANIFEST.json` berisi hash untuk 32 artifacts.

Notebook yang tampak mandiri tersebut sebenarnya melakukan artifact restoration dan analysis. Data tidak hilang hanya dengan menghapus output notebook: base64 di source juga perlu dipisahkan sebelum publikasi. Salinan GitHub membaca ZIP eksternal dari `data/private/embedded_evidence.zip`.

## `v3-27.zip`

Menyimpan 18 historical phase wrappers di `scripts/` beserta `_phase_runner.py`, lalu experiments `scripts_v3/` sampai `scripts_v27/`. Source mencakup OCR + Siamese linking, DINOv2 visual linking, face-based linking, address candidate selection, GPU re-OCR, specialist layout parsers, dan peer consensus.

Historical wrappers bergantung pada full implementation `identity_ocr_compfest18.py` dari workspace development. Portable core yang tersedia dalam ZIP kedua hanya mendukung final downstream chain. Menganggap seluruh phase wrappers langsung runnable akan menghasilkan panduan yang keliru.

Python bytecode dan `.ipynb_checkpoints` adalah artifacts runtime; keduanya diinventarisasi tetapi tidak disalin ke GitHub.

## `v27-28.zip`

Root bundle di dalam ZIP:

```text
compfest/compfest_academy_project/V27_REPRODUCIBLE_BUNDLE/
```

Private inputs yang digunakan final chain:

- V3: `ocr_raw.parquet`, `ocr_gray.parquet`, `ocr_gray_clahe.parquet`.
- V17: `field_proposals.csv`.
- V22: `predictions_released.parquet` sebagai frozen predecessor.
- V27: retained reference di `reference_outputs/`.
- Official test: `results/test_predictions.csv`.

Bundle juga menyimpan V28 released/candidate outputs, validation summaries, audit CSVs, reports, pinned requirements, dan integrity manifest `SHA256SUMS.txt`. Historical cleanup reports menyebut assets dari workspace A100; daftar itu bukan jaminan bahwa semua assets tersebut ada di ZIP.

## Presentasi dan business analysis

Presentasi menghubungkan business problem, EDA, core architecture, contoh field traceability, validation, bottleneck, dan deployment recommendations. Appendix memuat label distribution, quality audit, routing, preprocessing, OCR transcript, split diagnostics, metric formulas, runtime configuration, Unknown Route, serta pemilihan OCR engines. Sebagian appendix berupa gambar; text extraction saja tidak mewakili seluruh visual isinya.

PDF dan original executed notebooks tetap tersedia dalam ekstraksi lokal. Repository public memuat ringkasan dan source yang dibersihkan karena dokumen tersebut membawa contoh identitas dan/atau OCR transcript.

## Penelusuran file

- [archive_inventory.csv](archive_inventory.csv): path, ukuran, SHA-256, dan kategori untuk setiap file hasil ekstraksi, termasuk nested ZIP contents.
- [source_provenance.csv](source_provenance.csv): pemetaan public source ke lokasi aslinya beserta SHA-256.

Inventory memuat metadata file, tanpa raw labels atau identity records. File image telah diperiksa integritasnya; seluruh 732 JPG dapat dibuka, dengan 62 kombinasi dimensi dan tanpa byte-identical duplicates. Pemeriksaan seluruh image bukan pemeriksaan manual identitas satu per satu.
