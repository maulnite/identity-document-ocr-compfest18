# Evaluation dan batas interpretasi

## Sumber dan unit evaluasi

Metrik final berasal dari released prediction tables yang memuat 732 records. Hanya **632 rows dengan `has_ground_truth=True`** yang masuk evaluation. Sebanyak 100 test rows memiliki provenance release terpisah; tidak tersedia ground truth atau leaderboard score untuk mereka di arsip.

Public copies tersedia di [results/aggregate](../results/aggregate/). Nilai per-person, OCR transcripts, dan predictions tetap lokal.

## Normalisasi dan formula

Implementation acuan adalah `pipeline/identity_ocr_compfest18.py`, khususnya `metric_text()`, `text_metrics()`, dan `score_prediction_rows()`.

`metric_text()` melakukan Unicode NFC normalization, uppercase, whitespace collapse, penyeragaman apostrophe/dash, serta cleanup whitespace di sekitar separator tertentu. Empty labels menjadi string kosong.

Untuk ground truth `g`, prediction `p`, dan Levenshtein distance `d` setelah normalisasi:

| Metrik | Formula |
|---|---|
| Field EM | `1[g == p]` |
| Field CER | `d / max(1, len(g))` |
| Field normalized similarity | `max(0, 1 - d / max(1, len(g), len(p)))` |
| Macro Field EM per record | Mean Name EM, DOB EM, Address EM |
| Full Record EM | Ketiga field exact sekaligus |
| Record CER | Mean CER ketiga fields |
| Record normalized similarity | Mean similarity ketiga fields |

Dataset metrics adalah mean pada labeled records. CER tidak dibatasi 1: prediction panjang terhadap blank/short ground truth dapat menghasilkan CER besar. Nilai CER di sini bukan pooled character-weighted CER atas seluruh corpus.

Final notebook memakai uppercase/whitespace normalization untuk perhitungan ulang EM, sedangkan frozen evaluation core memiliki tambahan separator normalization. Keduanya cocok pada retained release; downstream verification memakai core agar definisinya eksplisit.

## Hasil final

| Metrik | V27 | V28 | Perubahan |
|---|---:|---:|---:|
| Name EM | 46,2025% | 46,2025% | 0 pp |
| DOB EM | 69,6203% | 70,4114% | +0,7911 pp |
| Address EM | 83,7025% | 83,7025% | 0 pp |
| Macro Field EM | 66,5084% | 66,7722% | +0,2637 pp |
| Full Record EM | 37,6582% | 37,6582% | 0 pp |
| Normalized Edit Similarity | 79,7021% | 81,2829% | +1,5808 pp |
| Record CER | 1,358073 | 1,349483 | −0,008591 |

V28 menaikkan DOB exact match sebanyak lima labeled rows secara neto. Name EM tetap, sehingga peningkatan similarity tidak boleh dilaporkan sebagai peningkatan Name exact extraction.

## Address caveat

All-row Address EM 83,70% menyertakan 520 blank ground truths. Pada 112 address-present rows, hanya 28 records exact: **25,00% EM**, dengan V27 similarity **86,93%**. V28 tidak mengubah Address.

Dalam V27 report, MyKad address subset mempunyai 100 images, 28% EM, dan sekitar 89,5% similarity; 12 Generic-ID address rows mempunyai 0% EM dan sekitar 31,4% similarity. False positives pada blank-address records tetap dihukum dalam all-row metric.

## Validation scopes

| Pertanyaan | Protocol yang tersedia | Batas |
|---|---|---|
| Apakah supervised component membantu pada held-out groups? | Group-disjoint holdout pada components tertentu, misalnya V25/V26 | Tidak seluruh end-to-end pipeline diuji ulang pada unseen identities |
| Apakah known peer evidence membantu target capture? | Leave-one-out peer analysis V27/V28 | Group membership sudah diketahui; bukan independent identity linking |
| Apakah V28 gate memberi non-regression? | Deterministic document split: 49 calibration, 98 validation dari 147 unknown labeled rows | Bukan identity-disjoint split; peer pool dapat mencakup capture dari split lain |
| Apakah final metrics konsisten? | Recompute dari frozen tables dan bandingkan aggregate JSON | Memverifikasi artifacts, bukan raw OCR models |
| Apakah OCR text membedakan identity pairs? | TF-IDF/SVD diagnostic dan all-pair ablation | Pair dependence, representation fit, dan threshold selection membatasi klaim generalisasi |

V28 candidates dibangun dari 237 unknown-route records across all splits sebelum labeled gate dievaluasi. Target tidak memilih dirinya sendiri sebagai peer, tetapi leave-one-out saja tidak menghilangkan leakage akibat known label-derived grouping atau transductive peer access. Selain itu, holdout non-regression digunakan sebagai release gate sehingga holdout tersebut tidak dapat dianggap final untouched test set.

## Remaining bottleneck

Unknown Route mencakup 147 labeled images. V28 Name EM tetap 0%; Name similarity sekitar 28,22%; DOB EM sekitar 19,73%, dengan DOB similarity 41,16%. Laporan audit menyatakan exact ground-truth Name ditemukan dalam retained line-level OCR hanya pada **1 dari 147** documents.

Ini menunjukkan perlunya routing/layout discovery dan recognition yang lebih baik. Temuan oracle tersebut berasal dari retained report/notebook audit; tidak berarti setiap kegagalan telah dibuktikan memiliki penyebab tunggal.

## Latency dan official test

Aggregate JSON menyimpan inherited OCR `P50 = 1261,82` dan `P95 = 2955,65` dalam milidetik. Angka ini berasal dari retained `ocr_latency_ms`, bukan pengukuran end-to-end V28 pada mesin sekarang. Tidak dilakukan throughput atau memory benchmark baru.

Official test CSV memiliki 100 unique filenames dan empat output columns. SHA-256 release yang dipertahankan:

```text
d30030f546124d12c6c7e514b8dc702b3631dcf209f8b00a91532400aac60624
```

V27/V28 hanya menyalin release tersebut. Tidak ada competition submission yang dilakukan dalam pekerjaan dokumentasi ini, dan metric V28 tidak menjelaskan kualitas test release tersebut.
