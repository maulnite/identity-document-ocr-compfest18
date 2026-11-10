# Project overview

## Konteks

Proyek final DSA COMPFEST 18 meminta pipeline OCR untuk mengekstrak informasi terstruktur dari foto/scan dokumen identitas. Tim Ketolak Pacil berfokus pada **Name**, **Birth Date**, dan **Address**. Dataset mencakup banyak document layouts, script/language, serta kualitas capture yang berbeda.

Dalam workflow KYC, membaca teks belum cukup: OCR perlu membedakan nama pemilik dari judul dokumen, tanggal lahir dari tanggal penerbitan, serta alamat dari teks lain. Karena itu, solusi menghubungkan text recognition dengan routing dan specialist parsing.

## Masalah, tujuan, dan pendekatan

| Kebutuhan | Konsekuensi pada desain |
|---|---|
| Mengurangi input data identitas secara manual | Menghasilkan tiga field yang terstruktur |
| Menangani blur, glare, dan low light | Quality assessment dan preprocessing candidates |
| Menghadapi format dokumen yang berbeda | Coarse family routing dan country/layout specialists |
| Mengetahui alasan hasil ekstraksi | Menyimpan OCR text, confidence, geometry, dan route evidence |
| Menghadapi output yang ambigu | Mengusulkan confidence gate dan human review |

Nilai bisnis yang diharapkan adalah mengurangi pekerjaan repetitif, mempercepat pengisian data, serta mengurangi rework. Proyek belum mengukur penghematan waktu operasional, production error rate, ROI, atau review workload secara langsung.

Business analysis dan presentasi asal mengutip studi eksternal mengenai efisiensi OCR. Angka eksternal itu tidak diangkat sebagai hasil eksperimen tim dalam repository ini. Identitas dan ringkasan dokumen sumber tersedia di [Archive guide](ARCHIVE_GUIDE.md); referensi ilmiahnya belum diverifikasi ulang terhadap sumber eksternal dalam pekerjaan dokumentasi ini.

## EDA yang membentuk solusi

1. **Capture heterogeneity:** enhancement yang membantu low-light images belum tentu membantu clean images. Raw, Gray, dan Gray + CLAHE diperlakukan sebagai candidates.
2. **Structure heterogeneity:** posisi dan arti OCR lines bergantung pada document family. Secondary routing menentukan specialist extraction.
3. **Repeated captures:** 506 dari 632 labeled images berada dalam repeated label-derived groups. Ini membuka kesempatan cross-capture analysis sekaligus risiko leakage pada random image split.
4. **Field availability:** hanya 112 labeled images mempunyai operational Address label. Metrik subset perlu dilaporkan bersama all-row accuracy.
5. **Unknown routes:** fallback Name extraction menjadi bottleneck; peer consensus hanya memberikan perbaikan terbatas ketika OCR evidence tidak memuat nama yang benar.

## Rekomendasi pengembangan

Alur bisnis yang diusulkan:

`Upload → Quality Gate → OCR + Structured Extraction → Confidence / Consistency Check → Prefill atau Human Review`

Guided retake dapat dipakai untuk input yang terlalu blur, glare, gelap, atau terpotong. Ketika routing ambigu, metadata country/document type dapat membantu specialist selection. Reviewer memerlukan field evidence dan alasan eskalasi.

Semua ini merupakan rekomendasi desain; repository tidak menyediakan aplikasi KYC production, liveness detection, document authenticity verification, atau sistem keputusan otomatis pelanggan. Reliable repeated-capture reconciliation memerlukan identity linking yang tervalidasi secara terpisah.
