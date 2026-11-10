# Verifikasi lokal

Verifikasi dilakukan pada **10 November 2026** dalam virtual environment proyek yang terpisah, menggunakan Python 3.10.4. Original archive dan extracted source tidak dimodifikasi.

| Pemeriksaan | Hasil |
|---|---|
| Integrity seluruh 732 raw JPG images | PASS; 62 kombinasi dimensi, tidak ada byte-identical duplicates |
| Original bundle `SHA256SUMS.txt` | PASS; 51 file hashes cocok |
| Final notebook embedded evidence manifest | PASS; 32 tracked hashes cocok |
| V27 downstream runner | PASS; released prediction/evaluation columns dan metric dictionary cocok dengan retained reference |
| V28 downstream runner | PASS; seluruh released table (732 × 100 columns), metrics JSON, dan validation JSON cocok dengan retained V28 release |
| Recompute metrics dengan portable scoring core | PASS untuk V27 dan V28 pada 632 labeled rows |
| Address-present subset | PASS; 112 rows, 25% EM |
| Official test schema, row count, dan hash | PASS; 100 unique filenames, empat columns, SHA-256 cocok |
| Final public notebook, seluruh 58 code cells | PASS; dijalankan dari external private evidence ZIP |
| Public notebook format | PASS; ketiga notebook valid menurut nbformat |
| Source syntax | PASS; historical Python source dan code cells ketiga public notebooks dapat di-parse |
| Public notebook outputs | Final execution serta retained EDA/ablation analytical outputs disertakan |
| Complete original material | Diarsipkan byte-identical di `original_project/`; lihat [Backup](BACKUP.md) |
| Fresh clone recovery dari GitHub | PASS; seluruh original hashes, preparation, V27/V28, metrics, dan final notebook |
| Release assets diunduh ulang | PASS; SHA-256 kedua ZIP, ZIP CRC, dan 98 snapshot member hashes |

Notebook execution pertama mengungkap missing optional dependency Jinja2 untuk pandas Styler. Dependency tersebut ditambahkan ke `requirements-notebooks.txt`; execution berikutnya berhasil. Tidak ada shared package yang diganti.

## Yang belum diuji ulang

- Full raw-image OCR dan training models.
- Complete original image-quality EDA run pada 732 images.
- GPU PP-OCRv6 vs hybrid OCR ablation.
- Historical experiments yang memerlukan full development assets.
- Deployment pada unseen identities atau real-world KYC.

Ablation metrics dalam dokumentasi berasal dari retained executed notebook. Image integrity checks tidak setara dengan review manual setiap dokumen. Reproduksi artifacts mengonfirmasi konsistensi results; ia tidak menghilangkan validation limitations pada [Evaluation](EVALUATION.md).
