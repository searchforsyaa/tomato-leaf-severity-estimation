# Model Checkpoints

Folder ini menyimpan bobot model hasil training.

---

## Isi

- `leaf_best.pth`  
  Model segmentasi daun dengan performa validasi terbaik

- `disease_best.pth`  
  Model segmentasi penyakit dengan performa validasi terbaik

---

## Catatan

- Model disimpan berdasarkan **validation loss terbaik**
- File `.pth` tidak disertakan dalam version control (gitignore)
- Digunakan untuk evaluasi dan inferensi

---

## Reproducibility

Untuk mereproduksi model:
1. Jalankan training ulang
2. Model terbaik akan tersimpan otomatis di folder ini
