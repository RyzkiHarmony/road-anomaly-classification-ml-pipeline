# 🛣️ Road Anomaly Detection — ML Pipeline

![CI Pipeline](https://github.com/YOUR_USERNAME/ml_pipelines/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c.svg)
![ONNX](https://img.shields.io/badge/export-ONNX-informational.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

An **end-to-end ML pipeline** for real-time road anomaly detection (potholes & speed bumps) on Android edge devices. The system uses raw IMU sensor data (accelerometer + gyroscope) collected via smartphone mounted on a motorcycle, and exports a production-ready model to ONNX for zero-latency on-device inference.

> **Context:** This pipeline was developed as the core ML system for an undergraduate thesis (*Skripsi*) at UDINUS. The final model achieves **Macro F1-Score = 0.86** on the holdout test set after Optuna hyperparameter optimization.

---

## 📐 System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Raw Sensor Data│────▶│  Sensor Fusion   │────▶│  Windowing &       │
│  (100Hz CSV)    │     │  (Causal Filter) │     │  Signal Extraction │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  ONNX Export    │◀────│  CNN Training    │◀────│  Optuna HPO        │
│  (Android Ready)│     │  (Focal Loss)    │     │  (10 Trials)       │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

**Key Pipeline Stages:**
1. **Sensor Fusion** — Separates gravity from linear acceleration using a causal low-pass filter (`scipy.signal.lfilter`) with zero lookahead, ensuring real-time viability on mobile.
2. **Signal Preprocessing** — Strict resampling to 100 Hz (10ms interval) with a maximum gap tolerance of 50ms. Applies **Global Z-Score Normalization** anchored on training set statistics (not per-instance), preventing inference-time distribution shift.
3. **Physics-Aware Augmentation** — During training: **Time Warping** (simulates variable motor speed), **Channel Dropout** (simulates sensor orientation faults), and **Temporal Jitter** (simulates mounting vibration).
4. **Hyperparameter Optimization** — Optuna with Median Pruner searches across LR, Dropout, Batch Size, Channels, Focal Loss Gamma, and Weight Decay. The objective is a **Robust Score** (`mean_F1 − std_F1`) to penalize unstable cross-fold behavior.
5. **ONNX Export** — PyTorch model wrapped with `MobileInferenceWrapper` (global scaler baked in as buffers) and exported via TorchScript to ONNX for on-device Android inference.
6. **Android Synchronization** — Processing logic is kept 1-to-1 between this pipeline and the Kotlin Android app, including the 50ms dropout gap tolerance and `eps=1e-6` Z-Score normalization.

---

## 📊 Model Performance (Holdout Test Set — 20%)

Results on 941 samples from **5 trip-isolated** test routes (never seen during training):

| Metric | Baseline 1D-CNN | Optuna Tuned 1D-CNN | Δ Improvement |
|:---|:---:|:---:|:---:|
| **Non-Event F1** | 0.98 | 0.98 | — |
| **Pothole F1** | 0.72 | **0.78** | +0.06 ✅ |
| **Pothole Recall** | 0.59 | **0.71** | +0.12 ✅ |
| **Speed Bump F1** | 0.79 | **0.81** | +0.02 ✅ |
| **Macro F1-Score** | 0.83 | **0.86** | +0.03 ✅ |
| **Accuracy** | 96% | **96%** | — |

> **Optuna Best Parameters:** `LR=0.001253`, `Batch=16`, `Channels=48`, `Dropout=0.2094`, `WeightDecay=2.7e-5`, `Gamma=1.67`

---

## 🔍 Alur Pengerjaan & Tahapan Pipeline Terperinci

Pipeline dikembangkan secara terstruktur melalui **6 Tahap Utama**, dimulai dari inspeksi data kotor mentah hingga ekspor model ke format ONNX untuk Android Edge AI:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 1: INSPEKSI DATA KOTOR & AUDIT KUALITAS (RAW DATA QUALITY AUDIT)      │
│  • Pengecekan Fluktuasi Sampling Rate (80Hz - 110Hz -> Irregular Interval)  │
│  • Deteksi Dropout Gaps (> 50ms akibat Throttling OS Android)               │
│  • Verifikasi Satuan Sensor (m/s² & rad/s) & Noise Level Getaran Motor      │
│  • Analisis Ketidakseimbangan Kelas Ground Truth (~15:1 Non-Event Ratio)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 2: PEMBERSIHAN DATA & SENSOR FUSION (DATA CLEANING & PREPROCESSING)    │
│  • Interpolasi Linier Resampling Teratur 100 Hz (10ms Interval)             │
│  • Causal Low-pass Filter (scipy.signal.lfilter, Latensi 0ms, Zero-Lookahead)│
│  • Sensor Fusion: Pemisahan Gravitasi (g) & Akselerasi Linier (a_lin)       │
│  • Spatial Alignment: Pencocokan GPS Haversine (< 10m) & Timestamp IMU      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 3: PEMBUATAN WINDOW & DATASET (WINDOWING & DATASET BUILDING)          │
│  • Windowing Terpusat 2.0 Detik (200 Sampel = 100 sebelum & 100 sesudah)    │
│  • Kalkulasi Z-Score Global Normalization (mean, std pada Training Set)     │
│  • Trip-Isolated Group Split (Stratified Group K-Fold berdasar trip_id)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 4: PELATIHAN MODEL & OPTIMASI HIPERPARAMETER (TRAINING & HPO)         │
│  • Arsitektur 1D-CNN InceptionTime1D + Squeeze-and-Excitation (SEBlock1D)   │
│  • MultiClassFocalLoss (Gamma=1.67) untuk Penanganan Extreme Class Imbalance │
│  • Physics-Aware Augmentation: Time Warping, Channel Dropout, Jitter        │
│  • Optuna HPO: Search Space (LR, Batch, Weight Decay, Channels, Gamma)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 5: EVALUASI MODEL, AUDIT KESALAHAN & STUDA ABLASI (EVALUATION & AUDIT) │
│  • Evaluasi Holdout Test Set (20% Trip Terisolasi yang Belum Pernah Dilihat) │
│  • Matriks Evaluasi: Confusion Matrix, PR-AUC, Precision, Recall, F1-Score  │
│  • Studi Ablasi Head-to-Head: Baseline 1D-CNN vs Optuna Tuned 1D-CNN        │
│  • Error Audit (Analisis False Positive & False Negative) & Uji Kalibrasi   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 6: EKSPOR ONNX & INTEGRASI ANDROID (ONNX EXPORT & EDGE AI)             │
│  • Wrap Model dengan MobileInferenceWrapper (Bake Z-Score Scaler Buffers)   │
│  • TorchScript Export -> 1dcnn_optuna_tuned.onnx                             │
│  • Benchmark Inference Latency (< 5ms per 2-second window) pada Device      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

```bash
# 1. Clone dan aktifkan venv proyek
git clone https://github.com/YOUR_USERNAME/ml_pipelines.git
cd ml_pipelines
.venv/Scripts/activate  # Windows PowerShell

# 2. Tahap 1 - Deteksi peak & labeling ground truth event
python src/stage1_dataset/01_detect_peaks_label.py
python src/stage1_dataset/02_generate_background.py

# 3. Tahap 2 - Prapemrosesan, Causal Filter, Pemisahan X & y (Tensor 3D & Tabular 2D)
python src/stage2_preprocessing/03_build_cnn_tensors.py
python src/stage2_preprocessing/04_build_xgb_tabular.py

# 4. Tahap 3 - EDA Distribusi Kelas & Imbalance
python src/stage3_eda_and_splitting/eda_distribution.py

# 5. Tahap 4 - Pelatihan & Optimasi Hiperparameter
python src/stage4_modeling/xgboost/06_train_xgb.py
python src/stage4_modeling/cnn/06_train_cnn.py

# 6. Tahap 5 - Evaluasi Klasifikasi Holdout Test Set Head-to-Head
python src/stage5_evaluation/07_compare_models.py

# 7. Tahap 6 - Ekspor ONNX, Benchmark Latensi Edge & Visualisasi Barchart Komparasi
python src/stage6_reports_deployment/08_export_onnx.py
python src/stage6_reports_deployment/09_benchmark_latency.py
python src/stage6_reports_deployment/plot_comparisons.py
```

---

## 🗂️ Project Structure

```
ml_pipelines/
├── src/
│   ├── stage1_dataset/                 # TAHAP 1: Penentuan Target, Ground Truth & Deteksi Peak
│   │   ├── 01_detect_peaks_label.py    # Ekstraksi peak & spatio-temporal clustering
│   │   ├── 02_generate_background.py   # Sampling non-event background
│   │   ├── clustering.py               # Algoritma spatio-temporal clustering
│   │   ├── peak_detection.py           # Adaptive thresholding peak detector
│   │   └── label_suggester.py          # Ground truth candidate heuristic scorer
│   ├── stage2_preprocessing/           # TAHAP 2: Prapemrosesan Sinyal & Pemisahan Fitur (X, y)
│   │   ├── sensor_fusion.py            # Causal low-pass filter (scipy.signal.lfilter)
│   │   ├── feature_extraction.py       # Ekstraksi 88 fitur domain waktu-frekuensi
│   │   ├── signal_windowing.py         # Windowing sinyal 2.0s @ 100Hz & gap handling
│   │   ├── 03_build_cnn_tensors.py     # Pemisahan fitur X (3D) & y untuk 1D-CNN
│   │   └── 04_build_xgb_tabular.py     # Pemisahan fitur X (2D) & y untuk XGBoost
│   ├── stage3_eda_and_splitting/       # TAHAP 3: EDA Imbalance, Data Splitting & Normalisasi
│   │   ├── eda_distribution.py         # Visualisasi diagram batang distribusi kelas target
│   │   ├── data_splitting.py           # Trip-isolated stratified group split (80:20)
│   │   └── normalizer.py               # Global Z-Score normalization (fit on train only)
│   ├── stage4_modeling/                # TAHAP 4: Pemodelan & Pelatihan (XGBoost & 1D-CNN)
│   │   ├── cnn/
│   │   │   ├── architecture.py         # InceptionTime1D + SEBlock1D (PyTorch)
│   │   │   ├── 05_tune_cnn.py          # Optuna Bayesian HPO
│   │   │   └── 06_train_cnn.py         # Training K-Fold + Holdout final model
│   │   └── xgboost/
│   │       ├── 05_tune_xgb.py          # Optuna Bayesian HPO
│   │       └── 06_train_xgb.py         # Training K-Fold + Holdout final model
│   ├── stage5_evaluation/              # TAHAP 5: Evaluasi Klasifikasi & Pengujian Testing
│   │   ├── 07_compare_models.py        # Komparasi Holdout head-to-head (Precision, Recall, F1)
│   │   ├── calculate_prauc.py          # PR-AUC curve analysis kelas minoritas
│   │   ├── error_audit.py              # Audit False Positives & False Negatives
│   │   └── feature_importance.py       # Analisis kontribusi fitur XGBoost
│   ├── stage6_reports_deployment/      # TAHAP 6: Visualisasi Akhir, Komparasi, Ekspor & Latensi
│   │   ├── 08_export_onnx.py           # Ekspor PyTorch (MobileInferenceWrapper) & XGB ke ONNX
│   │   ├── 09_benchmark_latency.py     # Benchmark latensi inferensi CPU single-thread edge
│   │   └── plot_comparisons.py         # Diagram batang komparasi Akurasi vs Latensi Komputasi
│   ├── utils/                          # Shared Global Config & Helpers
│   │   ├── config.py                   # Konstanta global, path, & threshold
│   │   └── helpers.py                  # Haversine, math utilities
│   └── tests/                          # Pytest Suite (31 passing tests)
├── data/                               # Data Mentah (raw) & Data Terproses (processed)
├── evaluation/                         # Checkpoints Model (.pth, .onnx) & Laporan Evaluasi (.csv, .png)
│   ├── models/                         # Model weights, label encoder & scaler params
│   └── reports/                        # Confusion matrices, PR curves, benchmark reports
├── log/                                # Training run logs
├── ruff.toml                           # Code quality configuration
└── requirements.txt                    # All dependencies (pinned versions)
```

---

## 🔬 Key Technical Decisions

| Decision | Rationale |
|:---|:---|
| **Causal Filter (no lookahead)** | Ensures zero-latency real-time inference on device. Non-causal filters would require future samples, making live detection impossible. |
| **Trip-Based Train/Test Split** | Prevents data leakage. Samples from the same trip share temporal correlations; shuffling would produce artificially inflated metrics. |
| **Global Z-Score (not per-instance)** | Anchoring normalization on training set statistics ensures the inference distribution matches training, preventing silent accuracy degradation at deployment. |
| **Focal Loss (γ=1.67)** | Addresses the extreme class imbalance (~1:15 ratio of anomaly to normal road). Down-weights easy Non-Event samples so the model focuses on learning the minority anomaly pattern. |
| **Robust Score Objective** | `mean(F1) − std(F1)` penalizes high-variance solutions in Optuna. A model that scores 0.70 in all folds is preferred over one that scores 0.90 in one fold and 0.50 in another. |
| **MobileInferenceWrapper (ONNX)** | Bakes the global scaler (mean, std) as PyTorch buffers into the ONNX graph, so the Android app only sends raw sensor windows — no separate preprocessing code required. |

---

## 🛠️ Developer Commands

```bash
make help         # List all available commands
make install      # Install all Python dependencies
make lint         # Run Ruff linter + auto-fix
make test         # Run unit tests with pytest
make train        # Train 1D-CNN (local)
make tune         # Run Optuna HPO (local)
make export-onnx  # Export model to ONNX (local)
make clean        # Remove __pycache__ and .pytest_cache
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
