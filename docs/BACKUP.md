# Backup lengkap dan pemulihan

## Isi yang diarsipkan

| Lokasi | Isi | Verifikasi |
|---|---|---|
| [original_project/](../original_project/) | Semua 896 file hasil ekstraksi, total 407.964.518 bytes | SHA-256 per file di [archive_inventory.csv](archive_inventory.csv) |
| [notebooks/](../notebooks/) | Final notebook yang dijalankan ulang dan retained EDA/ablation analytical outputs | Provenance di [Notebook guide](NOTEBOOKS.md) |
| [Release archive-v1](https://github.com/maulnite/identity-document-ocr-compfest18/releases/tag/archive-v1) — `FINAL.PROJECT.DSA.COMPFEST.18.zip` | Master ZIP asli, 386.952.264 bytes, tanpa repacking | SHA-256 di [release_checksums.sha256](release_checksums.sha256) |
| Release archive-v1 — `reproduction_artifacts.zip` | Generated V24–V28 results, prepared frozen inputs/reference, notebook evidence, executed final notebook, logs, dan environment record | SHA-256 di [release_checksums.sha256](release_checksums.sha256) |

`original_project/` mempertahankan relative paths dan bytes setiap file dari hasil ekstraksi, termasuk kedua nested ZIPs serta historical bytecode/checkpoints. Inventaris merinci semua file tanpa mengubah original data.

Master ZIP ditempatkan sebagai Release asset agar repository menyimpan expanded content yang dapat ditelusuri sekaligus menyediakan arsip asli untuk diunduh. Snapshot reproduksi menyimpan hasil kerja lokal yang tidak seluruhnya ada dalam master ZIP.

GitHub menormalisasi spasi pada nama master ZIP menjadi titik pada download asset. Isi ZIP tetap byte-identical dengan file asli `FINAL PROJECT DSA COMPFEST 18.zip`.

## Hasil uji pemulihan

Repository telah di-clone ulang dari GitHub ke folder baru. Seluruh 896 original file hashes cocok; preparation dari bundled originals, V27, V28, independent metric verification, dan seluruh 58 code cells notebook final berhasil dijalankan dari clone tersebut.

Kedua ZIP Release telah diunduh ulang dan diperiksa SHA-256 serta ZIP CRC. Seluruh 98 snapshot member hashes juga cocok. Detail mesin tercatat di [backup_verification.json](backup_verification.json). Uji ini tidak menggunakan folder dataset asli atau working artifacts dari server lama sebagai input.

## Pemulihan dari clone

```bash
git clone https://github.com/maulnite/identity-document-ocr-compfest18.git
cd identity-document-ocr-compfest18
python tools/verify_original_archive.py
python -m venv .venv
source .venv/bin/activate
python -m pip install -r pipeline/requirements.txt
python tools/prepare_private_artifacts.py
python pipeline/run_v27.py
python pipeline/run_v28.py
python tools/verify_artifacts.py
```

Original dataset tersedia di `original_project/dataset/`. PDF, executed notebooks, dan frozen bundle tetap dapat dibaca tanpa menyiapkan Python environment.

## Pemulihan dari Release

Unduh kedua ZIP serta checksum asset dari Release. Dari folder download:

```bash
sha256sum -c release_checksums.sha256
python -m zipfile -e 'FINAL.PROJECT.DSA.COMPFEST.18.zip' .
python -m zipfile -e 'FINAL PROJECT DSA COMPFEST 18/v27-28.zip' 'FINAL PROJECT DSA COMPFEST 18/v27-28'
```

Snapshot `reproduction_artifacts.zip` berisi `snapshot_manifest.json` yang mencatat path, ukuran, dan SHA-256 setiap member. Ekstrak ke folder baru untuk meninjau hasil run yang telah diverifikasi. Snapshot bukan virtual environment; gunakan requirements/environment record untuk membentuk ulang dependency.

## Batas kelengkapan

Backup mencakup seluruh materi dalam ZIP yang diberikan serta artifacts dokumentasi/reproduksi yang dibuat dalam pekerjaan ini. Original source menyebut model weights, full OCR core, cache besar, dan workspace A100 lain yang **tidak ada dalam ZIP sumber**; file tersebut tidak bisa dipulihkan dari backup ini. Full raw-image OCR tetap memiliki batas yang dijelaskan dalam [Reproducibility](REPRODUCIBILITY.md).

Tidak ada penghapusan file server yang dilakukan. Original materials dan master ZIP harus tetap dipertahankan sampai pemeriksaan upload dan checksum selesai.
