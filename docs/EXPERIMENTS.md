# Perjalanan eksperimen V3–V28

Tabel ini memetakan seluruh versioned source dalam arsip. Deskripsi berasal dari implementation dan reports yang disertakan. Historical versions yang hanya mempunyai source tidak diberi skor rekaan.

| Versi | Source directory | Fokus | Bukti yang tersedia |
|---|---|---|---|
| V3 | `pipeline/scripts_v3/` | Hybrid OCR, quality selection, Siamese identity linking, conservative consensus | Source + tiga frozen OCR tables |
| V4 | `pipeline/scripts_v4/` | DINOv2 visual identity linker | Source + historical README |
| V5 | `pipeline/scripts_v5/` | YOLO-face dan ArcFace identity evidence | Source + historical README |
| V6 | `pipeline/scripts_v6/` | Fuzzy/embedding address consensus | Source + historical README |
| V7 | `pipeline/scripts_v7/` | Address ROI geometry dan targeted OCR | Source + historical README |
| V8 | `pipeline/scripts_v8/` | Weakly supervised address-span selector | Source + historical README; dependency final chain |
| V9 | `pipeline/scripts_v9/` | Address template audit | Source + historical README |
| V10 | `pipeline/scripts_v10/` | Supervised MyKad address-span ranker | Source; dependency final chain |
| V11 | `pipeline/scripts_v11/` | MyKad address format consensus | Source; dependency final chain |
| V12 | `pipeline/scripts_v12/` | Visual-error guard untuk MyKad addresses | Source; dependency final chain |
| V13 | `pipeline/scripts_v13/` | Guarded format consensus | Source; dependency final chain |
| V14 | `pipeline/scripts_v14/` | GPU targeted address re-OCR | Source; historical class definitions |
| V15 | `pipeline/scripts_v15/` | Format-only guard | Source; dependency final chain |
| V16 | `pipeline/scripts_v16/` | GPU re-OCR pada Malaysian address cards | Source; dependency final chain |
| V17 | `pipeline/scripts_v17/` | Per-layout generic/MRZ parsing dan targeted OCR | Source + frozen field proposals |
| V18 | `pipeline/scripts_v18/` | Group-consensus generic fields | Source; dependency final chain |
| V19 | `pipeline/scripts_v19/` | Spain name component consensus | Source |
| V20 | `pipeline/scripts_v20/` | Chile DOB consensus | Source |
| V21 | `pipeline/scripts_v21/` | Extended generic group consensus | Source |
| V22 | `pipeline/scripts_v22/` | Final permit/Ukraine consensus | Source + frozen predecessor predictions |
| V23 | `pipeline/scripts_v23/` | Ukraine Latin consensus experiment | Source |
| V24 | `pipeline/scripts_v24/` | Ukraine Latin-route gate | Reproduced dalam V27 chain |
| V25 | `pipeline/scripts_v25/` | MyKad address-format gate | Reproduced dalam V27 chain |
| V26 | `pipeline/scripts_v26/` | MyKad Name span ranker | Reproduced dalam V27 chain |
| V27 | `pipeline/scripts_v27/` | Portugal/Spain fuzzy leave-one-out medoids | Retained reference + metrics + validation |
| V28 | `pipeline/scripts_v28/` | Unknown Generic-ID peer consensus | Retained reference + metrics + validation |

## Improvements yang dilaporkan source reports

| Komponen | Validation scope | Before → after |
|---|---|---|
| V24 Ukraine Name | 7 holdout images, peer/route analysis | Name EM 0% → 100% |
| V25 MyKad Address | 25 images, group-disjoint holdout | Address EM 8% → 40% |
| V26 MyKad Name | Group-disjoint holdout; 1.244 candidate spans dalam keseluruhan candidate set | Name EM 15,4% → 46,2%; similarity 61,8% → 95,2% |
| V27 Portugal Name | 11 validation documents, known peers | Similarity 51,99% → 64,29% |
| V27 Spain Name | 6 validation documents, known peers | Similarity 56,23% → 91,67% |
| V28 Unknown Name | 98 validation documents, known peers | Similarity 18,77% → 27,48%; EM tetap 0% |
| V28 Unknown DOB | 98 validation documents, known peers | EM 18,37% → 21,43%; similarity 31,84% → 43,37% |

Angka component holdout tidak setara dengan global metric dan beberapa subsets sangat kecil. Hasil V24–V26 di atas merupakan laporan development; reproduksi final output tidak otomatis merekonstruksi seluruh historical research process.

## Experiments yang tidak dipertahankan sebagai release improvement

Final V27 report mencatat direct V17 MRZ replacement lebih buruk daripada baseline; EU driving-licence fuzzy medoids tidak memberi gain; static Malaysian place-name splitter hanya membantu satu record dan tidak dirilis. Address-span oracle yang tersisa hanya mencapai 5,5% EM setelah formatting, di bawah retained MyKad address EM 28%. Ini mengarah pada OCR character errors sebagai remaining address bottleneck.

Historical cleanup menghapus banyak intermediate outputs dan cache sekitar 27,29 GiB di workspace asal. Source yang dipulihkan dari ZIP pertama tetap berguna untuk menelusuri pendekatan, tetapi tidak menyediakan seluruh checkpoint, models, atau scores intermediate.

## Additional OCR identity-signal ablation

Notebook tambahan membandingkan hybrid PP-OCRv6 detector + SVTRv2 recognizer dengan PaddleOCR PP-OCRv6 end-to-end pada **raw rectified input**, bukan adaptive preprocessing output.

Protocol menggunakan shared character TF-IDF/SVD representation, 632 images, 152 groups, dan seluruh **199.396 pairs**: 5.553 positive dan 193.843 negative pairs. Runtime output asal melaporkan PaddlePaddle 3.3.0 dan PaddleOCR 3.7.0.

| Metric, retained executed notebook | Hybrid | PP-OCRv6 end-to-end |
|---|---:|---:|
| Identity-pair ROC-AUC | 0,9357 | **0,9782** |
| Average Precision | 0,7201 | **0,7937** |
| Same-identity median cosine | 0,8843 | **0,9301** |
| Different-identity median cosine | 0,0574 | 0,0659 |
| Best Balanced Accuracy | 0,9205 | **0,9632** |

Pada 300 repeated subsamples yang mempertahankan 80% groups, mean ΔAUC = +0,0427; kuantil 2,5% dan 97,5% masing-masing +0,0268 dan +0,0572. Semua subsamples mempunyai ΔAUC positif. Ini stability diagnostic, bukan formal confidence interval.

Hasil tersebut mendukung lebih kuatnya identity signal PP-OCRv6 end-to-end dalam protocol ini. Hasil ini **tidak mengukur field extraction accuracy**, bukan pure recognizer-only ablation, dan belum membuktikan bahwa mengganti OCR backbone menaikkan V28 EM. Best threshold dipilih pada evaluated pairs sehingga balanced accuracy merupakan descriptive in-sample diagnostic.

Angka ablation ditranskrip dari retained HTML table outputs notebook asal; GPU experiment belum dijalankan ulang dalam pekerjaan dokumentasi. Exported ablation ZIP yang disebut notebook tidak ada sebagai file terpisah dalam arsip utama.
