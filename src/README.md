# Source Code

Folder ini berisi seluruh implementasi pipeline computer vision dalam proyek.

---

## Struktur
src/
├── datasets.py # Dataset PyTorch
├── models.py # Arsitektur U-Net
├── losses.py # Loss functions
├── utils.py # Training & evaluation utilities
├── train_leaf.py # Training segmentasi daun
├── train_disease.py # Training segmentasi penyakit
├── evaluate.py # Evaluasi model
├── labelme_to_mask.py # Konversi anotasi ke mask
└── inference_demo.py # Demo inferensi

---

##  Prinsip Desain

- Kode modular dan reusable
- Training dan evaluasi dipisahkan
- Notebook hanya untuk analisis, bukan logic utama

---

## Cara Menjalankan

Training segmentasi daun:
```bash
python -m src.train_leaf
```
Training segmentasi penyakit:
```bash
python -m src.train_disease
```
