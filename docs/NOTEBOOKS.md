# Notebook guide

## Pemetaan dan urutan membaca

| Public notebook | Original notebook | Fungsi |
|---|---|---|
| [01_final_project.ipynb](../notebooks/01_final_project.ipynb) | `output/Notebook_ketolak pacil_COMPFEST18.ipynb` | Narasi final: business problem, EDA, architecture, validation, metrics, recommendations |
| [02_pipeline_driven_eda.ipynb](../notebooks/02_pipeline_driven_eda.ipynb) | `additional/Ketolak_Pacil_Final_Pipeline_Driven_EDA_NaturalVisual_EXECUTED.ipynb` | EDA detail dari raw images, spreadsheet, dan frozen pipeline artifacts |
| [03_ocr_identity_ablation.ipynb](../notebooks/03_ocr_identity_ablation.ipynb) | `additional/Ketolak_Pacil_OCR_Identity_Similarity_Ablation_PPvsSVTR.ipynb` | Hybrid vs end-to-end OCR identity-signal diagnostic |

Notebook final adalah pintu masuk terbaik. EDA memperluas hubungan data evidence dengan design decisions. Ablation membahas pertanyaan berbeda dari field extraction accuracy.

## Isi notebook final

Notebook final mencakup provenance/hash checks, business understanding, field availability, repeated-group diagnostics, image quality, route distribution, name complexity, TF-IDF identity signal, adaptive preprocessing, hybrid OCR, specialist extraction, development timeline, validation scopes, V27/V28 metrics, route errors, traceability, dan retained test export.

Visual examples berasal dari frozen development evidence. `Run All` memulihkan private artifacts dan menghitung analysis/metrics; tidak melakukan full raw-image OCR atau retraining Siamese model.

## Isi EDA detail

EDA menemukan dataset dan pipeline bundle, memeriksa normalized/raw labels, menghitung raw image-quality features, mengeksplorasi routes dan field availability, mempelajari repeated-capture variation, menampilkan identity similarity diagnostics, lalu menghasilkan slide-ready summaries dan evidence-to-architecture map.

Sebagian narrative historical menyebut “final linker” atau peer consensus secara umum. Interpretasi yang dipakai repository mengikuti final notebook: Siamese adalah development experiment, sedangkan downstream known-group peer analysis mempunyai validation scope terpisah.

## Isi ablation

Tahapnya adalah environment check, evidence discovery, 632-image alignment, perspective rectification, PP-OCRv6 inference/cache, shared TF-IDF/SVD, all-pair metrics, visual diagnostics, group subsampling, qualitative examples, serta experiment export.

Input utama adalah raw rectified cards. Kedua corpora dipakai bersama untuk unsupervised representation fit. Pair labels dipakai sesudah embedding untuk evaluasi; diagnostic ini belum menyediakan independent test set untuk production linking threshold.

## Perubahan pada public copies

1. Seluruh execution outputs, execution counts, attachments, dan runtime metadata dihapus. Satu pembuka menjelaskan status salinan GitHub.
2. Embedded `_BUNDLE_B64` di notebook final diganti private ZIP loader dengan path containment check. Binary data tidak tersimpan dalam public source.
3. Notebook final menyimpan run artifacts di `data/private/final_project_run/`.
4. Dataset discovery EDA/ablation dibatasi ke repository dan explicit `OCR_DATASET_DIR`, sehingga tidak perlu memindai home directory atau project lain.
5. EDA bundle-root resolution disesuaikan untuk folder `pipeline/`; tidak lagi mensyaratkan nama literal `V27_REPRODUCIBLE_BUNDLE`.
6. Typo ablation `OUTPUT_DIR = WORK_DIR / "outputs"x` diperbaiki. Typo ini ada pada source yang dibagikan meskipun retained outputs menunjukkan execution sebelumnya.
7. Automatic package uninstall/upgrade pada ablation diganti version check; implicit pyarrow installation dihapus. Environment harus disiapkan secara terpisah.

Source methodology dan experiments dipertahankan, termasuk explanatory markdown asal yang sebagian menggunakan Bahasa Inggris. Documentation baru menggunakan Bahasa Indonesia. Original notebooks tetap tersedia dalam arsip lokal dan tidak dimodifikasi.

## Re-run dan publikasi

Menggunakan `Run All` dapat menghasilkan ulang PII dan identity image outputs pada notebook. Jangan menambahkan kembali executed notebooks tersebut ke public repository. Simpan execution copies di `data/private/` dan gunakan aggregate summaries untuk reporting. Public copy harus tetap memiliki `outputs=[]` dan `execution_count=null` pada setiap code cell.
