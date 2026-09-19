# 🛣️ Road Anomaly Detection — ML Pipeline

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![ONNX](https://img.shields.io/badge/export-ONNX-informational.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

Pipeline Machine Learning dan Deep Learning *end-to-end* untuk deteksi anomali permukaan jalan (*Pothole* dan *Speed Bump*) secara *real-time* pada perangkat edge Android. Sistem memanfaatkan data sensor gerak IMU (*Accelerometer* dan *Gyroscope*) berfrekuensi 100 Hz yang direkam menggunakan smartphone terpasang pada sepeda motor, dan mengekspor model teroptimasi ke format ONNX untuk inferensi latensi rendah langsung di perangkat.

> **Konteks Akademis:** *Pipeline* ini dikembangkan sebagai sistem inti untuk skripsi sarjana di **Universitas Dian Nuswantoro (UDINUS)**. Pipeline ini mengimplementasikan evaluasi ketat berbasis rute terisolasi (*Trip-Isolated Stratified Group Split 80:20*) dan studi ablasi *head-to-head* antara model *Feature Engineering* (XGBoost) dan *Deep Learning* (1D-CNN InceptionTime).

---

## 📐 Arsitektur Sistem

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────────────────┐
│  Raw Sensor Data│────▶│  Sensor Fusion   │────▶│  Windowing (2.0s @ 100Hz)        │
│  (100Hz CSV)    │     │  (Causal Filter) │     │  Causal Real-time Preprocessing  │
└─────────────────┘     └──────────────────┘     └─────────────────┬────────────────┘
                                                                   │
                                  ┌────────────────────────────────┴───────────────────────────────┐
                                  ▼                                                                ▼
              ┌───────────────────────────────────────┐                        ┌───────────────────────────────────────┐
              │ Jalur 1D-CNN (Tensor 3D)              │                        │ Jalur XGBoost (Tabular 2D)            │
              │ • 7 Sinyal Raw Berurutan (C=7, L=200) │                        │ • Ekstraksi 88 Fitur Waktu & Frekuensi│
              │ • Global Z-Score Normalization        │                        │ • Seleksi Non-Redundan (|r| <= 0.75)  │
              └───────────────────┬───────────────────┘                        └───────────────────┬───────────────────┘
                                  │                                                                │
                                  ▼                                                                ▼
              ┌───────────────────────────────────────┐                        ┌───────────────────────────────────────┐
              │ InceptionTime1D + SE-Block            │                        │ XGBoost Classifier                    │
              │ • MultiClass Focal Loss (γ=1.67)      │                        │ • Sample Weighting: Balanced          │
              │ • Optuna Bayesian HPO                 │                        │ • Kalibrasi Isotonik Probabilitas     │
              └───────────────────┬───────────────────┘                        └───────────────────┬───────────────────┘
                                  │                                                                │
                                  └────────────────────────────────┬───────────────────────────────┘
                                                                   ▼
                                              ┌────────────────────────────────────────┐
                                              │  Evaluasi Holdout & Studi Ablasi       │
                                              │  • 20% Rute Baru (Trip-Isolated)       │
                                              │  • Ekspor Model ONNX untuk Android     │
                                              └────────────────────────────────────────┘
```

---

## 📊 Hasil Evaluasi Model (Holdout Test Set — 20% Rute Baru)

Pengujian dilakukan pada **rute perjalanan baru** yang diisolasi secara ketat (*Trip-Isolated*), sehingga data uji tidak pernah dilihat oleh model selama proses pelatihan ataupun penyetelan hiperparameter.

### Studi Ablasi Head-to-Head: XGBoost vs 1D-CNN

| Kelas / Evaluasi | Model | Precision | Recall | F1-Score | Support |
|:---|:---|:---:|:---:|:---:|:---:|
| **Non-Event** | XGBoost (Optuna Tuned) | 0.9732 | 0.9847 | 0.9789 | 847 |
| | 1D-CNN (InceptionTime) | 0.9664 | 0.9881 | 0.9771 | 843 |
| **Pothole (Target Utama)** | XGBoost (Optuna Tuned) | 0.7308 | 0.6786 | 0.7037 | 56 |
| | 1D-CNN (InceptionTime) | **0.8810** | 0.6607 | **0.7551** | 56 |
| **Speed Bump** | XGBoost (Optuna Tuned) | 0.6111 | 0.5238 | 0.5641 | 42 |
| | 1D-CNN (InceptionTime) | **0.8378** | **0.7381** | **0.7848** | 42 |
| **Macro Average** | XGBoost (Optuna Tuned) | 0.7717 | 0.7290 | 0.7489 | 945 |
| | 1D-CNN (InceptionTime) | **0.8950** | **0.7956** | **0.8390** | 941 |
| **Overall Accuracy** | XGBoost (Optuna Tuned) | 94.60% | — | — | 945 |
| | 1D-CNN (InceptionTime) | **95.75%** | — | — | 941 |

> **Analisis Singkat:** Model deep learning 1D-CNN (InceptionTime) unggul dalam menangkap dinamika temporal gelombang getaran jalan (*F1 Pothole = 0.7551* dan *F1 Speed Bump = 0.7848*), sementara model XGBoost berbasis 25 fitur non-redundan memberikan F1 Pothole = 0.7037 dengan efisiensi komputasi sangat tinggi pada perangkat edge.

---

## ⚙️ Parameter Terbaik Hasil Optimasi Optuna

### 1. XGBoost (Objektif: Maksimalisasi PR-AUC Pothole)
File konfigurasi: `evaluation/models/xgboost/best_params.json`

| Parameter | Nilai Terpilih | Ruang Pencarian (*Search Space*) |
|:---|:---:|:---|
| `n_estimators` | **290** | [50, 300] (Integer) |
| `max_depth` | **7** | [3, 9] (Integer) |
| `min_child_weight` | **1** | [1, 10] (Integer) |
| `learning_rate` | **0.1879** | [0.01, 0.30] (Log Scale) |
| `subsample` | **0.6126** | [0.50, 1.00] (Uniform Float) |
| `colsample_bytree` | **0.5640** | [0.50, 1.00] (Uniform Float) |
| `reg_alpha` | **4.6666** | [0.00, 10.00] (Uniform Float) |
| `reg_lambda` | **7.8339** | [0.00, 10.00] (Uniform Float) |

### 2. 1D-CNN InceptionTime (Objektif: Robust Score `mean_F1 - std_F1`)
File konfigurasi: `src/stage4_modeling/cnn/best_optuna_params.json`

| Parameter | Nilai Terpilih | Deskripsi |
|:---|:---:|:---|
| `channels` | **48** | Lebar channel konvolusi Inception Block |
| `dropout` | **0.2094** | Regularisasi dropout rate |
| `learning_rate` | **0.001253** | Kecepatan konvergensi AdamW |
| `batch_size` | **16** | Ukuran mini-batch per iterasi |
| `weight_decay` | **2.7e-5** | Penalti bobot L2 |
| `focal_gamma` (γ) | **1.67** | Parameter fokus MultiClass Focal Loss |

---

## 🔍 Alur Pengerjaan & Tahapan Pipeline Terperinci

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 1: DETEKSI PEAK & GROUND TRUTH MAPPING (GROUND TRUTH ENGINE)          │
│  • Pengecekan Fluktuasi Sampling Rate (80Hz - 110Hz -> Regularisasi 100Hz)  │
│  • Deteksi Puncak Abnormalitas (Adaptive Thresholding Peak Detector)         │
│  • Spatio-Temporal Clustering (Pengelompokan Puncak Berulang < 0.8 Detik)   │
│  • Ground Truth Mapping diikat pada dimensi fisik absolut (time_s / GPS)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 2: SENSOR FUSION & PRAPEMROSESAN SINYAL (SIGNAL PREPROCESSING)        │
│  • Causal Low-pass Filter (scipy.signal.lfilter, Latensi 0ms, Zero Lookahead)│
│  • Sensor Fusion: Pemisahan Gravitasi (g) & Akselerasi Linier Dinamis        │
│  • Windowing Terpusat 2.0 Detik (200 Sampel @ 100Hz)                        │
│  • Pembentukan 3D Tensor untuk 1D-CNN (cnn_1d_X.npy, cnn_1d_y.npy)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 3: FEATURE ENGINEERING & DATASET SPLITTING (TABULAR & SPLITTING)      │
│  • Ekstraksi 88 Fitur Domain Waktu, Frekuensi (FFT) & Karakteristik Gelombang│
│  • Seleksi Fitur Non-Redundan: Pemangkasan multikolinieritas Pearson |r|<=0.75│
│  • Terpilih 25 Fitur Independen terkuat untuk representasi tabular XGBoost  │
│  • Trip-Isolated Stratified Group Split (80% Dev Set vs 20% Holdout Test)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 4: PEMODELAN & OPTIMASI HIPERPARAMETER (MODELING & OPTUNA HPO)        │
│  • XGBoost: Optimasi Optuna (100 Trials, PR-AUC Pothole) + Kalibrasi Isotonik│
│  • 1D-CNN: InceptionTime1D + SE-Block, MultiClass Focal Loss (γ=1.67)       │
│  • Physics-Aware Augmentation: Time Warping, Channel Dropout, Jitter        │
│  • 4-Fold Stratified Group K-Fold Cross-Validation pada Dev Set              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 5: EVALUASI MODEL & STUDI ABLASI (EVALUATION & BENCHMARKING)          │
│  • Pengujian Independen pada Holdout Test Set (5 Rute Baru yang Terisolasi) │
│  • Metrik Lengkap: Confusion Matrix, PR-AUC, Precision, Recall, F1-Score     │
│  • Studi Ablasi Komparatif Head-to-Head (XGBoost vs 1D-CNN)                  │
│  • Error Audit (Analisis Sampel False Positive & False Negative pada Peta)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 6: EKSPOR ONNX & DEPLOYMENT ANDROID (EDGE AI DEPLOYMENT)              │
│  • Pembungkusan MobileInferenceWrapper (Global Scaler Buffer terintegrasi)   │
│  • Ekspor Model PyTorch & XGBoost ke format ONNX (Opset 15)                  │
│  • Benchmark Latensi Inferensi CPU Single-Thread Mobile (< 5ms per window)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Panduan Eksekusi (Quickstart)

> **Catatan:** Selalu gunakan lingkungan virtual Python (`.venv\Scripts\python.exe`) yang telah terinstal dependensinya.

```bash
# 1. Masuk ke direktori proyek dan aktifkan venv
cd ml_pipelines
.\.venv\Scripts\activate

# 2. Tahap 1 - Deteksi Peak & Pembuatan Label Ground Truth
.venv\Scripts\python.exe src/stage1_dataset/01_detect_peaks_label.py
.venv\Scripts\python.exe src/stage1_dataset/02_generate_background.py

# 3. Tahap 2 - Prapemrosesan Sinyal & Pembentukan Tensor 1D-CNN
.venv\Scripts\python.exe src/stage2_preprocessing/03_build_cnn_tensors.py

# 4. Tahap 3 - Ekstraksi Fitur Tabular XGBoost & Analisis Distribusi
.venv\Scripts\python.exe src/stage3_feature_engineering/04_extract_xgb_features.py
.venv\Scripts\python.exe src/stage3_eda_and_splitting/eda_distribution.py

# 5. Tahap 4 - Optimasi Hiperparameter (Optuna) & Pelatihan Model
# A. Model XGBoost
.venv\Scripts\python.exe src/stage4_modeling/xgboost/05_tune_xgb.py
.venv\Scripts\python.exe src/stage4_modeling/xgboost/06_train_xgb.py

# B. Model 1D-CNN
.venv\Scripts\python.exe src/stage4_modeling/cnn/05_tune_cnn.py
.venv\Scripts\python.exe src/stage4_modeling/cnn/06_train_cnn.py

# 6. Tahap 5 - Evaluasi Komparatif & Studi Ablasi Head-to-Head
.venv\Scripts\python.exe src/stage5_evaluation/07_compare_models.py

# 7. Tahap 6 - Ekspor ONNX & Pengujian Latensi Edge
.venv\Scripts\python.exe src/stage6_reports_deployment/08_export_onnx.py
.venv\Scripts\python.exe src/stage6_reports_deployment/09_benchmark_latency.py
.venv\Scripts\python.exe src/stage6_reports_deployment/plot_comparisons.py
```

---

## 🗂️ Struktur Direktori Proyek

```
ml_pipelines/
├── src/
│   ├── stage1_dataset/                 # TAHAP 1: Penentuan Target, Ground Truth & Deteksi Peak
│   │   ├── 01_detect_peaks_label.py    # Deteksi peak & spatio-temporal clustering
│   │   ├── 02_generate_background.py   # Ekstraksi window non-event normal
│   │   ├── clustering.py               # Spatio-temporal DBSCAN-like clustering
│   │   ├── peak_detection.py           # Adaptive thresholding peak detector
│   │   ├── label_suggester.py          # Ground truth candidate heuristic scorer
│   │   └── manual_labeling_per_trip.py # Antarmuka verifikasi label per rute
│   ├── stage2_preprocessing/           # TAHAP 2: Prapemrosesan Sinyal & Sensor Fusion
│   │   ├── sensor_fusion.py            # Causal low-pass filter (scipy.signal.lfilter)
│   │   ├── signal_windowing.py         # Windowing sinyal 2.0s @ 100Hz & penanganan gap
│   │   └── 03_build_cnn_tensors.py     # Ekstraksi tensor 3D sinyal mentah untuk 1D-CNN
│   ├── stage3_feature_engineering/     # TAHAP 3A: Ekstraksi Fitur Domain Waktu-Frekuensi
│   │   ├── 04_extract_xgb_features.py  # Ekstraksi 88 fitur tabular per window
│   │   └── feature_extraction.py       # Modul kalkulasi statistik, energi, spectral & wavelet
│   ├── stage3_eda_and_splitting/       # TAHAP 3B: EDA, Trip Split & Normalisasi
│   │   ├── eda_distribution.py         # Visualisasi distribusi label & uji ketidakseimbangan
│   │   ├── data_splitting.py           # Trip-isolated stratified group split (80:20)
│   │   └── normalizer.py               # Global Z-Score normalization (fit on train only)
│   ├── stage4_modeling/                # TAHAP 4: Pemodelan, Tuning & Pelatihan
│   │   ├── cnn/                        # Jalur Deep Learning 1D-CNN
│   │   │   ├── architecture.py         # InceptionTime1D + SE-Block + Focal Loss
│   │   │   ├── 05_tune_cnn.py          # Optuna HPO untuk arsitektur & parameter CNN
│   │   │   ├── 06_train_cnn.py         # Pelatihan K-Fold + Holdout test model CNN
│   │   │   └── hard_negative_mining.py # Penambangan sampel false positive berulang
│   │   └── xgboost/                    # Jalur Klasik ML XGBoost
│   │       ├── 05_tune_xgb.py          # Optuna HPO seleksi fitur & parameter XGBoost
│   │       ├── 06_train_xgb.py         # Pelatihan K-Fold, kalibrasi isotonik & holdout
│   │       └── export.py               # Konversi model XGBoost ke format ONNX
│   ├── stage5_evaluation/              # TAHAP 5: Evaluasi Komparatif & Audit Kesalahan
│   │   ├── 07_compare_models.py        # Komparasi evaluasi holdout head-to-head
│   │   ├── calculate_prauc.py          # Analisis kurva Precision-Recall AUC
│   │   ├── error_audit.py              # Investigasi mendalam sampel FP & FN
│   │   ├── feature_importance.py       # Kontribusi fitur terpenting model XGBoost
│   │   └── visualize_trip_anomaly.py   # Visualisasi peta anomali jalan per rute
│   ├── stage6_reports_deployment/      # TAHAP 6: Visualisasi Akhir, Ekspor & Benchmark
│   │   ├── 08_export_onnx.py           # Ekspor PyTorch & XGBoost ke ONNX (Opset 15)
│   │   ├── 09_benchmark_latency.py     # Pengujian latensi eksekusi CPU edge
│   │   └── plot_comparisons.py         # Plot komparatif F1-score vs Latensi Komputasi
│   ├── utils/                          # Konfigurasi Global & Utility Matematika
│   │   ├── config.py                   # Konstanta global, path file, dan threshold sensor
│   │   └── helpers.py                  # Fungsi kalkulasi jarak Haversine & rotasi
│   └── tests/                          # Rangkaian Pengujian Unit (Unit Test Suite)
├── data/                               # Direktori data mentah dan olahan
├── evaluation/                         # Checkpoints Model (.pkl, .pth, .onnx) & Laporan Evaluasi
│   ├── models/                         # Model weights, label encoder & scaler params
│   └── reports/                        # Confusion matrices, kurva PR, dan laporan CSV
├── log/                                # Log sesi pelatihan & tuning
├── ruff.toml                           # Konfigurasi linter & code formatter Ruff
└── requirements.txt                    # Dependensi pustaka Python
```

---

## 🔬 Prinsip Rekayasa & Keputusan Teknis Utama

| Keputusan Teknis | Alasan & Rasionalitas Metodologis |
|:---|:---|
| **Causal Filter (Zero Lookahead)** | Menjamin inferensi *real-time* tanpa latensi pada perangkat mobile. Filter non-kausal membutuhkan sampel masa depan yang mustahil tersedia saat deteksi langsung di jalan. |
| **Trip-Isolated Data Split (80:20)** | Mencegah kebocoran data (*data leakage*). Sampel dari rute yang sama memiliki korelasi spasio-temporal tinggi; *shuffling* acak akan menghasilkan metrik evaluasi yang *overoptimistic*. |
| **Ground Truth Mapping Absolut** | Label ground truth diikat pada koordinat spasial dan dimensi waktu absolut (`time_s`), bukan indeks baris. Hal ini mencegah korupsi label saat parameter *filtering* diubah. |
| **Seleksi Fitur Pearson (\|r\| <= 0.75)** | Mengeliminasi multikolinieritas pada model XGBoost. Dari 56 fitur kandidat, 19 fitur redundan dipangkas dan 25 fitur independen terkuat dipertahankan. |
| **Kalibrasi Isotonik Probabilitas** | Memperbaiki kalibrasi skor probabilitas keluaran XGBoost pada kelas minoritas (*Pothole* dan *Speed Bump*) sehingga ambang deteksi optimal dapat diterapkan secara stabil. |
| **MultiClass Focal Loss (γ=1.67)** | Mengatasi ketidakseimbangan kelas ekstrem (~15:1 rasio Non-Event terhadap Anomali) pada 1D-CNN dengan menurunkan bobot sampel Non-Event yang mudah dipelajari. |
| **MobileInferenceWrapper (ONNX)** | Menyematkan parameter normalisasi Z-Score (*mean* dan *std*) langsung sebagai *buffer* ke dalam graf komputasi ONNX, sehingga aplikasi Android hanya perlu mengirimkan *window* sinyal mentah. |

---

## 🛠️ Perintah Pengembang (Developer Tooling)

```bash
make help         # Menampilkan daftar perintah yang tersedia
make install      # Menginstal semua dependensi Python ke dalam venv
make lint         # Menjalankan linter Ruff & auto-formatting
make test         # Menjalankan unit tests dengan pytest
make train        # Melatih model (XGBoost & 1D-CNN)
make tune         # Menjalankan Optuna Bayesian HPO
make export-onnx  # Mengekspor checkpoint model ke format ONNX
make clean        # Membersihkan cache Python (__pycache__, .pytest_cache)
```

---
