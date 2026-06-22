# ml_pipelines — Road Anomaly Detection Pipeline

Pipeline *Machine Learning* untuk deteksi anomali jalan (lubang/pothole dan polisi tidur/speed bump) menggunakan data sensor smartphone pada sepeda motor.

## Gambaran Umum

Project ini memproses data murni dari **akselerometer**, **giroskop**, dan **GPS** dari smartphone yang dipasang pada motor untuk mendeteksi anomali jalan secara otomatis. Pipeline telah berevolusi dari *Classical Machine Learning* (XGBoost + *Handcrafted Features*) menuju arsitektur *Deep Learning* secara *End-to-End* (1D-CNN) untuk menuntaskan masalah *Translation Invariance* dan batas komputasi perangkat *Edge* (Android).

## Arsitektur Final (Fase 6: 1D-CNN)

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Raw Data       │────▶│  Sensor Fusion   │────▶│  Windowing &       │
│  (100Hz CSV)    │     │  (Causal Filter) │     │  Random Jittering  │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  ONNX Export    │◀────│  1D-CNN Training │◀────│  Hard Negative     │
│  (Android Ready)│     │  (PyTorch)       │     │  Mining            │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

**Tahapan Utama Final:**
1. **Sensor Fusion (`sensor_fusion.py`)** — Memisahkan gravitasi dari akselerasi linear menggunakan *causal filter* (`scipy.signal.lfilter`) untuk zero-latency di perangkat *mobile*, menghasilkan `a_vertical`, `a_horizontal`, dan `speed`.
2. **Windowing & Curation (`build_dataset_cnn.py`)** — Mengekstrak jendela waktu 2 detik (200 *timestamps*). Menerapkan **Linear Interpolation** (dilanjutkan ffill/bfill) untuk membersihkan data kosong tanpa lonjakan buatan, serta **Random Jittering** dinamis untuk menghindari *Alignment Bias* (memastikan model mengenali lubang di posisi acak, bukan hanya di tengah *window*).
3. **Hard Negative Mining (`hard_negative_mining.py`)** — Menggali data sensor murni dengan *sliding window* untuk menemukan *False Positive* ekstrem dan memaksanya masuk sebagai kelas `Non-Event` tambahan.
4. **CNN Training (`train_cnn.py`)** — Melatih arsitektur Lightweight 1D-CNN murni dari data 10-channel `(10, 200)`. Mengimplementasikan **Focal Loss + Dampened Class Weights** dan mengevaluasi performa riil menggunakan **Optimized Threshold** (bukan sekadar default argmax 0.5) yang meminimalkan *False Negative*.
5. **Hyperparameter Tuning (`tune_cnn.py`)** — Optimisasi otomatis dengan **Optuna** secara selaras (menggunakan augmentasi dinamis, Focal Loss, dan LR scheduler yang identik dengan pipeline training) untuk meminimalkan *gap* evaluasi.
6. **ONNX Export (`export_onnx.py`)** — Membekukan model PyTorch ke format universal (`.onnx`) yang super ringan (<15K parameter) untuk *inference real-time* di Kotlin/Android.

## Struktur Folder Relevan

```
ml_pipelines/
├── README.md                   ← dokumen ini
├── requirements.txt            ← dependencies Python
│
├── 05_pipeline_experiment/     ← Lingkungan Ekstraksi Data Dasar & Eksperimen Awal
│   ├── sensor_fusion.py        ← Modul fusi sensor (Akselerometer + Giroskop)
│   ├── config.py               ← Konfigurasi pusat parameter sensor
│   └── manual_labeling_per_trip.py ← Script visualisasi & pelabelan Peta
│
├── 06_1dcnn/                   ← Lingkungan Utama (Arsitektur Deep Learning)
│   ├── model.py                ← Definisi arsitektur PyTorch 1D-CNN
│   ├── build_dataset_cnn.py    ← Pengekstrakan time-series array (Interpolasi Linier)
│   ├── hard_negative_mining.py ← Skrip pencarian False Positive ekstrem
│   ├── tune_cnn.py             ← Optimisasi Hyperparameter (Optuna selaras 10-Ch)
│   ├── train_cnn.py            ← Skrip training (K-Fold Validation) & Thresholding Optimal
│   ├── export_onnx.py          ← Skrip konversi PyTorch ke ONNX
│   ├── data/                   ← [GitIgnored] NPY tensors untuk training
│   └── models/                 ← [GitIgnored] Model tersimpan (.pth & .onnx)
│
└── scratch/                    ← Skrip diagnostic & uji coba sementara
```

## Taxonomy Label (3-class)

| Label | Deskripsi |
|-------|-----------|
| `Non-Event` | Semua yang bukan anomali diskrit (jalan mulus, engine vibration, manuver, polisi tidur rusak/rel kereta api keras yang direject) |
| `Pothole` | Lubang jalan (anomali negatif) |
| `Speed Bump` | Polisi tidur (anomali positif bertahap) |

## Cara Menjalankan Pipeline Akhir

### 1. Bangun Tensor Data (Time-Series)
```bash
python 06_1dcnn/build_dataset_cnn.py
```
*Ini akan menyapu semua label CSV, menginterpolasi data kosong, dan menghasilkan file `X.npy`, `y.npy`, `groups.npy` di folder `data/`.*

### 2. Jalankan Tuning Optuna (Opsional)
```bash
python 06_1dcnn/tune_cnn.py
```
*Mengeksplorasi kombinasi dropout & filter yang optimal dengan simulasi pipeline training yang presisi.*

### 3. Tambang Data Hard Negative (Opsional/Iteratif)
```bash
python 06_1dcnn/hard_negative_mining.py
```
*Memindai ratusan jam rekaman sensor mentah untuk mencari "jalan mulus/kasar" yang disalahpahami model sebagai lubang.*

### 4. Latih Model Utama (Cross-Validation)
```bash
python 06_1dcnn/train_cnn.py
```
*Melatih model dengan algoritma Stratified Group K-Fold untuk mencegah Data Leakage antar trip, kemudian mencetak Classification Report (Default vs. Optimized Threshold) serta menyimpan visualisasi Confusion Matrix yang telah disesuaikan dengan threshold optimal.*

### 5. Ekspor ke Android (ONNX)
```bash
python 06_1dcnn/export_onnx.py
```
*Menghasilkan file `.onnx` yang siap dimasukkan ke dalam folder `assets/` di Android Studio.*

## Catatan Engineering (Limitasi Fisika)
- Model ini tidak di-deploy menggunakan fitur XGBoost karena **Feature Engineering** secara manual terbukti rapuh terhadap **Translation Variance** (posisi lubang yang bergeser dalam detak 2 detik).
- Walaupun CNN mengungguli XGBoost, model ini menderita **High Bias (Fisika)** karena sensor IMU murni tidak memiliki komponen visual (kamera), menyebabkan batas absolut dalam membedakan benturan suspensi ekstrem (seperti rel kereta api) dengan lubang asli.
- Proses komputasi pada Edge Device ditekan habis-habisan (O(N) *complexity*) dengan mengandalkan filter konvolusi 1D murni tanpa operasi *sorting* atau statistik rekursif.
- Preprocessing data menggunakan interpolasi linier untuk menjamin kontinuitas sinyal sensor sebelum dihitung nilai turunannya (jerk & crest factor).
- Threshold probabilitas kelas dikalibrasi pasca-latih menggunakan kurva Precision-Recall untuk meminimalkan misklasifikasi krusial pada lubang jalan.
