# Tomato Leaf Disease Severity Estimation
**Two-Stage Semantic Segmentation untuk Estimasi Tingkat Keparahan Penyakit Daun Tomat**

---

## Overview

Proyek ini mengembangkan pipeline *computer vision* end-to-end untuk **mengestimasi tingkat keparahan penyakit Early Blight pada daun tomat** berbasis citra RGB.  
Berbeda dari pendekatan klasifikasi sederhana, proyek ini memformulasikan masalah sebagai **dua tahap segmentasi semantik**, sehingga menghasilkan **estimasi severity berbasis rasio area penyakit terhadap area daun**.

---

## Metodologi

Pipeline terdiri dari dua tahap utama:

1. **Leaf Segmentation**
   - Input: citra RGB
   - Output: mask biner daun
   - Model: U-Net

2. **Disease Segmentation**
   - Input: citra RGB yang dimask oleh hasil segmentasi daun
   - Output: mask biner penyakit
   - Model: U-Net

3. **Severity Estimation**
   - Menghitung rasio area penyakit terhadap area daun
   - Digunakan sebagai indikator tingkat keparahan penyakit

Pendekatan dua tahap ini bertujuan mengurangi *false positive* penyakit di area non-daun.

---

## Struktur Repository
```
learn-comvis/
├── checkpoints/ # Model terlatih (.pth)
├── data/ # Dataset, mask, dan split
├── notebooks/ # Eksplorasi & evaluasi
├── src/ # Implementasi pipeline
├── .venv/ # Virtual environment
└── README.md
```
---


## Keterbatasan

- Dataset relatif kecil dan dianotasi manual
- Lesi penyakit berukuran sangat kecil sulit disegmentasi presisi
- Evaluasi difokuskan pada severity, bukan pixel-perfect segmentation

---

## Demo Inferensi

Proyek ini menyediakan demo inferensi sederhana untuk memprediksi tingkat keparahan penyakit dari satu citra daun tomat.
Dapat diakses melalui notebook `inference_demo.ipynb`.

---
## Environment

Proyek ini dijalankan menggunakan:
- Python 3.11.5
- Windows 11
- NVIDIA GeForce RTX 5050 GPU (CUDA supported)
