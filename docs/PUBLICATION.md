# Catatan publikasi

## Yang tersedia di GitHub

- Documentation metodologi, dataset audit, architecture, experiment history, evaluation, dan reproducibility.
- Historical Python source dan portable downstream core.
- Tiga notebook dengan executed/retained outputs, beserta original notebook yang tetap utuh.
- Aggregate metric/validation JSON dan seluruh original prediction/OCR tables.
- Diagram pipeline dan chart yang hanya memakai aggregate metrics.
- Metadata file inventory dan source provenance dengan SHA-256.
- `original_project/`: seluruh 896 file hasil ekstraksi arsip asli, termasuk 732 images, ground truth, tiga PDF, nested ZIPs, dan original embedded evidence.
- Release `archive-v1`: ZIP utama asli serta snapshot artifacts dan hasil reproduksi lokal.

## Working copies dan original archive

Materi lengkap dipublikasikan sesuai permintaan pemilik proyek untuk pengarsipan sebelum pembersihan server. Raw identity images, labels, per-person predictions, OCR transcripts, dan contoh dokumen dalam PDF/notebooks dapat diakses dari repository public.

`data/private/`, generated `pipeline/results*/`, dan caches tetap diabaikan Git sebagai working copies yang dapat dibangun ulang. Nama `private` dipertahankan untuk kompatibilitas script, tetapi original inputs tidak lagi bergantung pada file yang hanya tersedia di server. `original_project/**` dikecualikan dari ignore rules agar semua original files ikut tersimpan. Snapshot hasil run tersedia sebagai Release asset.

Virtual environments, `.git` internals, authentication configuration, dan credentials tidak dimasukkan ke backup proyek. Environment dapat dibentuk ulang dari pinned requirements; versi dependency yang digunakan untuk verifikasi juga dicatat dalam snapshot.

## Provenance dan hak penggunaan

Kode dalam `pipeline/` disalin byte-identical dari source yang tersedia; [source_provenance.csv](source_provenance.csv) merekam pemetaannya. Public notebooks merupakan salinan yang disesuaikan, sebagaimana [Notebook guide](NOTEBOOKS.md).

Tidak ditemukan explicit project license dalam arsip. Repository tidak menambahkan lisensi baru untuk kode tim, dataset, slides, academy visual, atau third-party models. Project credits mengikuti presentasi/notebook sumber. Publikasi arsip tidak mengubah hak penggunaan original assets.

## Batas hasil dokumentasi

Tidak dilakukan training baru, full OCR inference, benchmark hardware baru, atau competition submission. Aggregate V27/V28 direproduksi dari frozen artifacts dan notebook final diverifikasi sebagaimana [Verification](VERIFICATION.md). Ablation results berasal dari original executed notebook.

Pemulihan tidak memerlukan server lama: clone repository menyediakan expanded originals, sedangkan Release menyediakan master ZIP dan reproduction snapshot. Petunjuk dan checksum tersedia di [Backup](BACKUP.md).
