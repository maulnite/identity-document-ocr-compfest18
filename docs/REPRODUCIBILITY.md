# Reproducibility

## Jalur yang didukung

| Jalur | Input | GPU | Scope |
|---|---|---|---|
| V27 downstream reproduction | Frozen V22/V17/V3 tables + retained V27 reference | Tidak | Regenerasi V24→V27 dan exact reference comparison |
| V28 downstream reproduction | Retained V27 reference + V17 routes | Tidak | Unknown Generic-ID peer experiment |
| Final project notebook | Private embedded evidence ZIP | Tidak | EDA, narrative, diagnostics, metric consistency, retained test export |
| Original pipeline-driven EDA | Raw images/XLSX + frozen pipeline results | Tidak | Quality EDA dan pipeline-assisted diagnostics |
| OCR identity ablation | Raw images, geometry, frozen OCR, Paddle runtime | GPU direkomendasikan | OCR inference dan identity-pair diagnostic |
| Full raw-image pipeline rebuild | Full core, models, third-party assets, config, preprocessing cache | Ya pada development asal | Belum tersedia lengkap dalam arsip |

## 1. Environment terpisah

Dari root repository, gunakan virtual environment sendiri. Verifikasi lokal memakai Python 3.10.4, dependency downstream sesuai `pipeline/requirements.txt`, dan Matplotlib 3.8.4 untuk final notebook.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r pipeline/requirements.txt
```

Tidak perlu memasang PyTorch, PaddleOCR, atau OpenCV untuk V27/V28 downstream chain. Jangan mengganti packages shared environment untuk mengikuti notebook historical.

## 2. Ekstraksi private source

Gunakan arsip asli yang akses dan penggunaannya diizinkan. Contoh berikut dijalankan pada folder yang menyimpan ZIP utama, bukan pada root repository:

```bash
python -m zipfile -e 'FINAL PROJECT DSA COMPFEST 18.zip' .
python -m zipfile -e 'FINAL PROJECT DSA COMPFEST 18/v27-28.zip' 'FINAL PROJECT DSA COMPFEST 18/v27-28'
```

Untuk mempelajari historical sources secara terpisah, `v3-27.zip` dapat diekstrak ke subfolder `v3-27/`. Source tersebut sudah tersedia di `pipeline/` repository sehingga tidak wajib untuk final reproduction.

Kembali ke root repository:

```bash
python tools/prepare_private_artifacts.py --source '/absolute/path/FINAL PROJECT DSA COMPFEST 18'
```

Helper menyalin private input/reference tables ke lokasi yang diabaikan Git dan mengambil embedded ZIP dari original final notebook menggunakan AST literal parsing. Original notebook source tidak dieksekusi. Helper tidak menimpa file berbeda, tidak mengunduh models, dan tidak melakukan upload.

## 3. Reproduksi V27

```bash
python pipeline/run_v27.py
```

Runner mengeksekusi V24, V25, V26, dan V27. Ia membandingkan 11 prediction/evaluation columns dengan retained reference, memeriksa aggregate metric dictionary, dan memeriksa official test copy hash. Hasil yang diharapkan:

```json
{"verification": "passed", "labeled_records": 632}
```

Output sebenarnya juga menyertakan official test SHA-256. Jika generated V24–V27 folders sudah ada, runner menolak menjalankan ulang tanpa `--force`. Opsi `--force` menghapus dan mengganti **hanya generated V24–V27 folders** di bawah `pipeline/`; gunakan setelah memastikan tidak ada hasil lokal yang ingin dipertahankan di sana.

## 4. Reproduksi V28 dan independent metric check

```bash
python pipeline/run_v28.py
python tools/verify_artifacts.py
```

V28 membaca `pipeline/reference_outputs/results_v27_fuzzy_peer_medoid_names/`, bukan regenerated V27 folder. Ia dapat dijalankan secara independen setelah private preparation. Runner V28 menulis ulang generated V28 files pada pemanggilan berikutnya; simpan hasil penting di lokasi lain terlebih dahulu.

Verifier menghitung ulang field/record metrics menggunakan portable scoring core, memastikan 632 labeled rows dan 112 address-present rows, lalu memeriksa 100-row official test schema dan hash. Ia tidak mencetak identitas atau per-person predictions.

## 5. Notebook final

```bash
python -m pip install -r requirements-notebooks.txt
python tools/run_final_notebook.py
```

Runner mengeksekusi seluruh code cells menggunakan interpreter environment aktif dan menyimpan execution copy di `data/private/final_project_executed.ipynb`. Alternatifnya, buka `notebooks/01_final_project.ipynb` memakai kernel virtual environment tersebut dan jalankan seluruh cells. Root repository ditemukan dari working directory/ancestor yang mempunyai `pipeline/run_v27.py`.

Notebook membaca `data/private/embedded_evidence.zip`; lokasi alternatif dapat diberikan melalui environment variable `OCR_EVIDENCE_ZIP`. Generated artifacts dan outputs disimpan di `data/private/final_project_run/`. Semua itu diabaikan Git.

Final notebook menggunakan frozen V28 artifacts di embedded ZIP, sehingga tidak bergantung pada generated output V28 yang baru. Ini jalur artifact-analysis reproduction yang terpisah.

## 6. Original EDA

Siapkan private artifacts dan jalankan V28 agar released result tersedia, lalu pasang notebook dependencies. Set dataset directory secara eksplisit sebelum membuka notebook:

```bash
export OCR_DATASET_DIR='/absolute/path/FINAL PROJECT DSA COMPFEST 18/dataset'
```

Jalankan `notebooks/02_pipeline_driven_eda.ipynb`. Ia memakai `ground_truth.xlsx` dan raw images serta V28 table. Discovery dibatasi ke explicit dataset path, `pipeline/`, dan `data/private/`. Generated `pipeline_driven_eda/` diabaikan Git karena dapat berisi identity examples.

Notebook EDA lengkap belum dijalankan ulang dalam verifikasi dokumentasi ini; source syntax dan notebook structure diperiksa. Full EDA tetap memerlukan image-quality computation pada 732 images.

## 7. OCR ablation

Gunakan environment GPU yang terpisah dari downstream environment. Retained execution melaporkan **PaddlePaddle 3.3.0** dan **PaddleOCR 3.7.0**. Pilihan GPU wheel harus sesuai driver/CUDA host; notebook public memeriksa versi dan tidak melakukan uninstall/upgrade otomatis. Model downloads memerlukan akses network yang sesuai environment.

Notebook juga memerlukan NumPy, pandas, pyarrow, scikit-learn, OpenCV, Pillow, Matplotlib, dan tqdm. Sediakan authorized raw images melalui `OCR_DATASET_DIR`; gunakan `RUN_MODE="smoke"` untuk initial check sebelum full 632-image run. Geometry dan hybrid raw OCR dibaca dari retained artifacts.

Ablation tidak dijalankan ulang di sini. Angka pada [Experiments](EXPERIMENTS.md) berasal dari retained executed notebook, bukan pengukuran hardware sekarang.

## Troubleshooting

| Gejala | Penyebab dan tindakan |
|---|---|
| Private input tidak ditemukan | Periksa folder nested ZIP extraction dan `--source` |
| Helper menolak target berbeda | Bandingkan artifacts terlebih dahulu; helper sengaja tidak menimpa |
| `Generated folders already exist` | Simpan hasil bila perlu, kemudian gunakan V27 `--force` |
| Raw OCR unavailable pada historical script | Portable core tidak menyediakan full development API |
| Ground truth CSV terlihat banyak missing fields | Gunakan unwrapping pada [Dataset](DATASET.md) atau normalized frozen labels |
| EDA tidak menemukan images | Set `OCR_DATASET_DIR` ke dataset asli |
| Notebook public tidak menampilkan executed outputs | Outputs sengaja dihapus; hasil agregat tersedia di docs dan `results/aggregate/` |
