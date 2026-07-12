# ml_pipelines — Road Anomaly Detection Pipeline

Pipeline *Machine Learning* untuk deteksi anomali jalan (lubang/pothole dan polisi tidur/speed bump) menggunakan data sensor murni (akselerometer, giroskop) dan GPS dari smartphone yang dipasang pada sepeda motor.

## Gambaran Umum

Proyek ini memproses data sensor mentah untuk mendeteksi dan mengklasifikasikan anomali jalan secara otomatis. Pipeline ini mengevaluasi dan mengoptimasi dua arsitektur utama:
1. **Classical Machine Learning (XGBoost)** dengan fitur statistik dan morfologi *hand-crafted*.
2. **Deep Learning (Lightweight 1D-CNN)** yang dibangun secara *End-to-End* dengan augmentasi sinyal fisis.
3. **Ensemble (Soft Voting)** yang menggabungkan kekuatan XGBoost dan 1D-CNN.

---

## Arsitektur & Pipeline Produksi

`	ext
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Raw Data       │────▶│  Sensor Fusion   │────▶│  Windowing &       │
│  (100Hz CSV)    │     │  (Causal Filter) │     │  Feature/Signal    │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  ONNX Export    │◀────│  Model Training  │◀────│  Ensemble          │
│  (Android Ready)│     │  & Calibration   │     │  Evaluation        │
└─────────────────┘     └──────────────────┘     └────────────────────┘
`

**Tahapan Pemrosesan Utama:**
1. **Sensor Fusion (sensor_fusion.py)** — Memisahkan gravitasi dari akselerasi linear menggunakan *causal low-pass filter* (scipy.signal.lfilter) untuk eksekusi latensi-nol pada perangkat mobile, menghasilkan _vertical, _horizontal, dan speed.
2. **Feature Extraction (eature_extraction.py)** — Mengekstrak 47 fitur statistik dan morfologi (*shape-aware*). Rasio spesifik-domain (misal 
ise_time_ratio, down_up_asymmetry) distabilkan secara numerik melalui pembatasan nilai (*clipping*) untuk mencegah *outlier* tak terhingga.
3. **Prapemrosesan Sinyal & Augmentasi Fisis (1D-CNN)** — Melakukan resampling ketat ke 100 Hz (interval 10ms) dengan toleransi celah maksimum 50ms untuk menghindari halusinasi data. Menggunakan *zero-padding* dan **Global Z-Score Normalization**. Menerapkan **Time Warping** (menyimulasikan variasi kecepatan motor) dan **Channel Dropout** (menyimulasikan kesalahan orientasi sensor) secara dinamis selama pelatihan.
4. **Isotonic Calibration (XGBoost)** — Kalibrasi probabilitas pasca-pelatihan melalui *Out-Of-Fold (OOF) Isotonic Regression* untuk meredam inflasi probabilitas pada kelas minoritas.
5. **Auto-ONNX Export** — Mengekspor model akhir PyTorch dan XGBoost langsung ke format universal .onnx untuk inferensi *Edge AI* secara *real-time* di Android/Kotlin.
6. **Sinkronisasi Android Kotlin** — Logika aplikasi Android dijamin sinkron 1-banding-1 dengan *pipeline* ini, termasuk toleransi kekosongan data (*dropout gap*) 50ms, normalisasi tingkat instansi (*global-level*) dengan Z-Score eps=1e-6, dan penentuan batas putusan standar (*Default Argmax*) tanpa pergeseran probabilitas buatan.

---

## Metrik Evaluasi Akhir (Aligned Holdout Test Set)

Metrik berikut mewakili performa akhir di dunia nyata yang dievaluasi pada 1.386 *event Holdout Test Set* (30% dari perjalanan terisolasi) yang diselaraskan secara ketat, mengandung ketidakseimbangan kelas ekstrem (rasio minoritas ~1:12):

| Model / Metrik | Pothole Precision | Pothole Recall | Pothole F1-Score | Speed Bump Precision | Speed Bump Recall | Speed Bump F1-Score | Macro F1-Score | Global Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Calibrated)** | 0.774 | 0.456 | 0.573 | 0.500 | 0.442 | 0.469 | 0.671 | 91% |
| **1D-CNN (InceptionTime)** | 0.705 | **0.689** | **0.697** | 0.581 | **0.837** | **0.686** | **0.786** | **94%** |

### Analisis Hasil
* **Keunggulan 1D-CNN:** Arsitektur *Lightweight 1D-CNN* yang dipadukan dengan *Multi-Label Focal Loss* menunjukkan superioritas mutlak dalam menerjemahkan benturan fisik temporal menjadi klasifikasi. Model ini mencapai **Macro F1-Score sebesar 0.786**, dengan mudah mengalahkan pendekatan rekayasa fitur manual.
* **Recall vs Keselamatan:** 1D-CNN secara signifikan meningkatkan Pothole Recall menjadi **68.9%** (dibandingkan XGBoost 45.6%) sembari mempertahankan Precision yang tangguh di angka **70.5%**. Keseimbangan ini sangat krusial bagi sistem keselamatan *Edge AI* di dunia nyata untuk mencegah *False Negative* (lubang yang terlewat) tanpa membanjiri pengguna dengan *False Positive*.
* **Ketahanan Edge (Edge Resilience):** Integrasi augmentasi dinamis berhasil meniadakan masalah *Translation Variance* (di mana posisi anomali bergeser dalam jendela pemrosesan), membuktikan ketahanan arsitektur di berbagai lingkungan jalan dan konfigurasi suspensi.

---

## Cara Menjalankan Pipeline

Gunakan Virtual Environment proyek (.venv) untuk mengeksekusi *pipeline* dari terminal Anda.

### 1. Ekstraksi Dataset
`ash
python src/dataset/build_xgboost_data.py
python src/dataset/build_cnn_data.py
`

### 2. Pelatihan Model & Ekspor ONNX
`ash
# Melatih XGBoost + Isotonic Calibration + Ekspor ONNX
python src/xgboost_model/train.py

# Melatih 1D-CNN + Dynamic Augmentation + Auto-ONNX Export
python src/cnn_model/train.py
`

### 3. Evaluasi & Analisis
`ash
# Membandingkan performa XGBoost vs CNN
python evaluation/compare_models.py

# Menjalankan Audit Anomali Sensor Perjalanan
python evaluation/visualize_trip_anomaly.py
`
