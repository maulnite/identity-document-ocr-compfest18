# Dataset dan integrity audit

## Unit analisis

| Komponen | Jumlah |
|---|---:|
| Seluruh raw JPG images | 732 |
| Labeled rows | 632 |
| Test rows tanpa ground truth dalam arsip | 100 |
| Label-derived identity groups pada labeled set | 152 |
| Repeated label-derived groups | 26 |
| Captures dalam repeated groups | 506 |
| Address-present labeled rows setelah normalisasi | 112 |
| Blank Address labeled rows setelah normalisasi | 520 |

Schema competition output adalah `filename,name,birth_date,address`. Prediction artifacts menambahkan ground truth bersufiks `_gt`, prediction fields bersufiks `_pred`, routing, geometry, quality, OCR evidence, dan evaluation columns.

## Masalah format ground truth

CSV memiliki header empat kolom yang valid. Namun, **131 dari 632 data rows** terbungkus sebagai satu CSV field yang masih memuat record empat kolom di dalamnya. `pd.read_csv()` secara langsung menghasilkan banyak field kosong dan filename yang salah untuk rows tersebut.

Pembacaan defensif menggunakan standard library:

```python
import csv

with open("ground_truth.csv", newline="", encoding="utf-8") as handle:
    reader = csv.reader(handle)
    header = next(reader)
    rows = []
    for row in reader:
        if len(row) == 1:
            row = next(csv.reader([row[0]]))
        if len(row) != len(header):
            raise ValueError("Unexpected CSV row width")
        rows.append(dict(zip(header, row)))
```

Setelah unwrapping, 632 rows mempunyai empat field; Name dan DOB terisi pada seluruh rows, sementara Address terisi pada 112 rows. Source asli tidak ditulis ulang.

XLSX menyimpan 633 worksheet rows termasuk header. Date cells dapat berupa Excel serial values dengan number formats, sehingga membandingkan raw worksheet XML dengan CSV secara string tidak setara. Ada 113 raw Address cells yang tidak kosong; salah satunya menjadi blank setelah operational normalization. Gunakan normalized frozen benchmark labels untuk mereproduksi released metrics.

## Document families dan routes

| Coarse family, labeled set | N |
|---|---:|
| `MYKAD` | 100 |
| `MRZ_DOCUMENT` | 182 |
| `GENERIC_ID` | 350 |

`GENERIC_ID` adalah routing bucket. Ia mencakup EU driving licences, country-specific IDs, permits, dan unknown layouts. Detailed route diturunkan dari V17 field proposals untuk diagnostics, bukan independent annotation accuracy.

| Detailed route | Labeled N |
|---|---:|
| MyKad | 100 |
| MRZ / Passport | 182 |
| EU Driving Licence | 92 |
| Unknown Route | 147 |
| Ukraine Passport | 20 |
| HK Mainland Permit | 16 |
| Turkey ID | 9 |
| Czech ID | 20 |
| Chile ID | 20 |
| Spain ID | 9 |
| Portugal Citizen Card | 17 |

## Identity groups dan leakage

V3 `identity_key()` menyusun training label dari normalized Name, DOB, dan Address. Group yang diturunkan dari label bukan verifikasi jumlah orang unik. Shared specimen labels atau label similarity juga dapat memengaruhi grouping.

Repeated captures tidak independen. Final notebook mensimulasikan overlap repeated groups pada random image split, dengan sekitar 91,2% groups muncul pada kedua sisi. Angka ini merupakan diagnostic simulation dengan seed dan konfigurasi notebook, bukan hasil validation OCR.

Group-disjoint validation dipakai pada supervised components tertentu. Peer-consensus validation V27/V28 merupakan document-level split di dalam known groups dan memiliki scope berbeda. Lihat [Evaluation](EVALUATION.md).

## Kualitas image

Audit file lokal memeriksa 732 JPG dengan Pillow: seluruh image lolos integrity check, ditemukan 62 kombinasi width/height, dan tidak ada byte-identical duplicate files. Repeated captures tetap dapat memperlihatkan dokumen atau label yang sama.

Image-quality EDA menggunakan brightness, contrast, Laplacian blur score, glare ratio, dan illumination variation. Nilai yang dihitung pada resized raw images di EDA tidak harus identik dengan retained preprocessing metadata. Pipeline-assisted quality/route diagnostics tidak boleh diperlakukan sebagai manual ground truth.
