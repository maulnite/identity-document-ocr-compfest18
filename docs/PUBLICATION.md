# Catatan publikasi

## Yang tersedia di GitHub

- Documentation metodologi, dataset audit, architecture, experiment history, evaluation, dan reproducibility.
- Historical Python source dan portable downstream core.
- Tiga notebook tanpa outputs, attachments, embedded binary identity evidence, atau runtime metadata.
- Aggregate metric/validation JSON tanpa per-person rows.
- Diagram pipeline dan chart yang hanya memakai aggregate metrics.
- Metadata file inventory dan source provenance dengan SHA-256.

## Yang disimpan lokal

Raw identity images; ground truth CSV/XLSX; prediction/peer-candidate tables; OCR transcripts; private manifests/evidence bundles; executed notebook copies; dan original PDFs tidak dipublikasikan. PDF presentasi serta visual examples memuat contoh dokumen dan teks identitas.

`data/private/`, `pipeline/results*/`, `pipeline/reference_outputs/`, private report tables, caches, generated notebook folders, ZIPs, dan raw image formats diabaikan melalui `.gitignore`. Agregat public dipilih secara eksplisit; `.gitignore` saja bukan pengganti pemeriksaan staged files.

Tidak ada private download URL atau credentials yang ditambahkan. Public clone memerlukan original artifacts yang diperoleh secara berizin untuk menjalankan reproduction.

## Provenance dan hak penggunaan

Kode dalam `pipeline/` disalin byte-identical dari source yang tersedia; [source_provenance.csv](source_provenance.csv) merekam pemetaannya. Public notebooks merupakan salinan yang disesuaikan, sebagaimana [Notebook guide](NOTEBOOKS.md).

Tidak ditemukan explicit project license dalam arsip. Repository tidak menambahkan lisensi baru untuk kode tim, dataset, slides, academy visual, atau third-party models. Project credits mengikuti presentasi/notebook sumber. Keberadaan repository public tidak otomatis menetapkan izin redistribusi semua original assets.

## Batas hasil dokumentasi

Tidak dilakukan training baru, full OCR inference, benchmark hardware baru, atau competition submission. Aggregate V27/V28 direproduksi dari frozen artifacts dan notebook final diverifikasi sebagaimana [Verification](VERIFICATION.md). Ablation results berasal dari original executed notebook.

Future public additions sebaiknya menggunakan synthetic examples atau aggregate evidence agar pembaca dapat memahami metode tanpa raw identity data.
