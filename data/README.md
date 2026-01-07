# Dataset & Anotasi

Folder ini berisi data mentah, mask hasil anotasi, serta pembagian data (split) yang digunakan dalam proyek.

---

## Struktur Folder
data/
├── raw/
│ ├── Tomato___Early_blight/
│ └── Tomato___healthy/
├── masks/
│ ├── leaf/
│ └── disease/
├── splits.json
└── README.md

---

## Raw Data

Dataset berasal dari **PlantVillage** dan difokuskan pada dua kelas:
- Tomato Early Blight
- Tomato Healthy

Citra berupa foto daun tomat dengan latar belakang homogen.

---

## Anotasi

Anotasi dilakukan secara manual menggunakan **LabelMe** dengan dua label:
- `leaf` → area daun
- `disease` → area lesi penyakit

Hasil anotasi (.json) dikonversi menjadi mask biner:
- Mask daun (`leaf`)
- Mask penyakit (`disease`)

Mask disimpan terpisah untuk memudahkan pipeline dua tahap.

---

## Data Split

Pembagian data dilakukan menggunakan `splits.json` dengan tiga subset:
- Train
- Validation
- Test

Pembagian dilakukan pada level file untuk memastikan konsistensi antar citra dan mask.

---

## Catatan Penting

- Nama file antar kelas dapat sama (mis. `image (123).JPG`)
- Pipeline mengandalkan path folder untuk membedakan kelas
- Mask penyakit **tidak tersedia** untuk kelas healthy

---

## Tujuan Penyimpanan Terpisah

Struktur ini memungkinkan:
- Training ulang model tanpa anotasi ulang
- Evaluasi yang reproducible
- Ekstensi dataset di masa depan

