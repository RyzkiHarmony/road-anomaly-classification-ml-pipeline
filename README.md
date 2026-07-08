# ml_pipelines — Road Anomaly Detection Pipeline

Pipeline *Machine Learning* untuk deteksi anomali jalan (lubang/pothole dan polisi tidur/speed bump) menggunakan data sensor smartphone pada sepeda motor.

## Gambaran Umum

Project ini memproses data murni dari **akselerometer**, **giroskop**, dan **GPS** dari smartphone yang dipasang pada motor untuk mendeteksi anomali jalan secara otomatis. Pipeline ini membandingkan dan mengoptimasi dua model utama:
1. **Classical Machine Learning (XGBoost)** dengan *hand-crafted statistical & shape features*.
2. **Deep Learning (1D-CNN)** secara *End-to-End* dengan augmentasi sinyal fisik.
3. **Ensemble (Soft Voting)** menggabungkan kekuatan XGBoost (Recall tinggi) dan 1D-CNN (Precision tinggi).

---

## Arsitektur & Pipeline Produksi

```
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
```

**Tahapan Utama:**
1. **Sensor Fusion (`sensor_fusion.py`)** — Memisahkan gravitasi dari akselerasi linear menggunakan *causal filter* (`scipy.signal.lfilter`) untuk zero-latency di perangkat *mobile*, menghasilkan `a_vertical`, `a_horizontal`, dan `speed`.
2. **Feature Extraction (`feature_extraction.py`)** — Mengekstrak 56 fitur statistik dan morfologi (*shape-aware*). Ratios seperti `rise_time_ratio` dan `down_up_asymmetry` distabilkan secara numerik menggunakan batas atas (*clipping*) untuk mencegah pencilan tak berhingga (divisi oleh nol).
3. **Prapemrosesan Sinyal & Augmentasi Fisik (1D-CNN)** — Melakukan _resampling_ secara ketat ke 100 Hz (interval 10ms) dengan batas toleransi _gap_ 50ms untuk menghindari _hallucinated data_. Menggunakan _zero-padding_ dan _instance-level Z-Score scaling_. Menerapkan **Time Warping** (simulasi kecepatan motor bervariasi) dan **Channel Dropout** (simulasi kesalahan/pergeseran orientasi sensor) secara dinamis saat training.
4. **Isotonic Calibration (XGBoost)** — Mengkalibrasi probabilitas XGBoost pasca-latih secara *out-of-fold* menggunakan Isotonic Regression untuk meredam inflasi probabilitas pada kelas minoritas.
5. **Auto-ONNX Export** — Mengekspor model final PyTorch dan XGBoost secara langsung ke format universal (`.onnx`) untuk dijalankan secara real-time di Kotlin/Android Studio.
6. **Android Kotlin Synchronization** — Aplikasi Android terjamin sinkron 1-to-1 dengan _pipeline_ ini. Termasuk _dropout gap_ 50ms, `eps=1e-6` Z-Score _instance-level normalization_, dan penggunaan _Default Argmax_ (`0.50`) tanpa Threshold modifikasi buatan.

---

## Struktur Folder Relevan

```
ml_pipelines/
├── README.md                   ← dokumen ini
├── requirements.txt            ← dependencies Python
│
├── src/                        ← Kode sumber utama pipeline aktif
│   ├── dataset/
│   │   ├── build_xgboost_data.py  ← Membentuk CSV fitur untuk XGBoost
│   │   ├── build_cnn_data.py      ← Membentuk NumPy tensors untuk CNN
│   │   ├── sensor_fusion.py       ← Modul fusi filter kausal low-latency
│   │   └── feature_extraction.py  ← Ekstraksi 56 fitur statistik & bentuk
│   │
│   ├── xgboost_model/
│   │   └── train.py               ← Latih XGBoost + Isotonic Calibration + ONNX
│   │
│   ├── cnn_model/
│   │   ├── model.py               ← Arsitektur Lightweight 1D-CNN PyTorch
│   │   └── train.py               ← Latih 1D-CNN + Augmentasi Fisik + Auto-ONNX
│   │
│   └── utils/
│       └── config.py              ← Konfigurasi parameter sensor & window
│
├── evaluation/                 ← Evaluasi, Audit, dan Komparasi Model
│   ├── compare_models.py          ← Membandingkan performa XGB vs CNN
│   ├── evaluate_ensemble.py       ← Penggabungan probabilitas (Soft Voting)
│   ├── error_audit.py             ← Analisis False Positive Pothole secara mendalam
│   ├── visualize_trip_anomaly.py  ← Visualisasi anomali fisik sensor per trip
│   │
│   ├── models/                    ← Model tersimpan (.pth, .pkl, & .onnx)
│   └── reports/                   ← Grafik Confusion Matrix & CSV perbandingan
│
└── archive/                    ← Arsip eksperimen awal (05_pipeline dan 06_1dcnn)
```

---

## Hasil Performa Model (Holdout Test Set)

Berikut perbandingan performa XGBoost, 1D-CNN, dan Ensemble pada Holdout Test Set (30% trip terpisah) setelah perbaikan instabilitas numerik:

| Model / Metrik | Pothole Precision | Pothole Recall | Pothole F1-score | Speed Bump Precision | Speed Bump Recall | Speed Bump F1-score | Akurasi Global |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Calibrated)** | 0.60 | **0.76** | **0.67** | 0.43 | 0.48 | 0.45 | 93% |
| **1D-CNN (100 Hz strict)** | 0.51 | 0.69 | 0.58 | 0.52 | **0.58** | 0.55 | 93% |
| **Ensemble (Soft Voting)** | **0.65** | 0.68 | **0.67** | **0.77** | 0.48 | **0.59** | **94%** |

### Analisis Hasil
*   **Ensemble Soft Voting** dengan bobot **0.30 XGBoost + 0.70 CNN** menghasilkan performa terbaik dan terefisien untuk operasional *real-world*. F1-score Pothole mencapai **0.67** dan presisi melonjak ke angka fantastis **65%**, sangat menekan masalah "alarm palsu" yang sering dialami oleh *Edge AI* pada kendaraan roda dua. Presisi pendeteksian polisi tidur (*Speed Bump*) juga mencapai puncaknya di **77%**.
*   **Stabilisasi Sinyal:** Masalah instabilitas numerik pada fitur waveform `rise_time_ratio` diselesaikan dengan clipping `[0.0, 50.0]`, menurunkan rasio pencilan False Positive dari **1,3 Juta** menjadi **1,13** (TP: 2.31, FP: 2.62).
*   **Validasi Pipeline (100 Hz):** Pipeline CNN kini secara ketat menolak jendela sinyal jika terdapat *gap* > 50ms dan mengaplikasikan *zero-padding* serta *instance-level normalization* yang seratus persen kongruen dengan aplikasi Android, menghindari halusinasi saat OS mengalami *lag*.

---

## Cara Menjalankan Pipeline

Gunakan Virtual Environment proyek (`.venv`) untuk mengeksekusi perintah di bawah ini pada Windows PowerShell:

### 1. Ekstraksi Dataset
```powershell
.venv\Scripts\python.exe src/dataset/build_xgboost_data.py
.venv\Scripts\python.exe src/dataset/build_cnn_data.py
```

### 2. Pelatihan & Ekspor Model
```powershell
# Melatih XGBoost + Kalibrasi Isotonic + Ekspor ONNX
.venv\Scripts\python.exe src/xgboost_model/train.py

# Melatih 1D-CNN + Augmentasi Fisik + Ekspor ONNX otomatis
.venv\Scripts\python.exe src/cnn_model/train.py
```

### 3. Evaluasi & Analisis
```powershell
# Membandingkan model XGBoost vs CNN
.venv\Scripts\python.exe evaluation/compare_models.py

# Mengoptimasi dan mengevaluasi Ensemble (Soft Voting)
.venv\Scripts\python.exe evaluation/evaluate_ensemble.py

# Menjalankan Audit Anomali Sensor Trip
.venv\Scripts\python.exe evaluation/visualize_trip_anomaly.py
```

### 4. Unit Testing
```powershell
.venv\Scripts\python.exe -m pytest src/tests/
```
