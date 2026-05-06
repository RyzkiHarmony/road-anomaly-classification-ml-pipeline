# ml_pipelines — Road Anomaly Detection Pipeline

Pipeline machine learning untuk deteksi anomali jalan (lubang/pothole dan polisi tidur/speed bump) menggunakan data sensor smartphone pada sepeda motor.

## Gambaran Umum

Project ini memproses data **akselerometer** dan **giroskop** dari smartphone yang dipasang pada motor untuk mendeteksi anomali jalan secara otomatis. Pipeline mencakup tahapan dari pengumpulan data mentah hingga dataset siap training.

## Alur Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Data Mentah    │────▶│  Labeling        │────▶│  Ground Truth      │
│  (sensor CSV)   │     │  Pipeline        │     │  (labeled events)  │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                              │                           │
                              ▼                           ▼
                        ┌──────────────────┐     ┌────────────────────┐
                        │  Feature         │     │  Training Set      │
                        │  Extraction      │────▶│  Builder           │
                        └──────────────────┘     └────────────────────┘
```

**Tahapan utama:**

1. **Sensor Fusion** (`sensor_fusion.py`) — Memisahkan gravitasi dari akselerasi linear, menghitung `a_vertical`.
2. **Peak Detection** (`peak_detection.py`) — Deteksi puncak anomali pada sinyal `a_vertical` menggunakan adaptive threshold.
3. **Clustering** (`clustering.py`) — Mengelompokkan peak yang berdekatan secara temporal & spasial menjadi satu event.
4. **Feature Extraction** (`feature_extraction.py`) — Mengekstrak fitur shape, frekuensi, dan distribusi dari setiap event.
5. **Scoring** (`scoring.py`) — Menghitung composite score untuk memprioritaskan event yang paling meyakinkan.
6. **Label Suggestion** (`label_suggester.py`) — Hybrid ML + rule-based label suggestion (RandomForest + XGBoost ensemble).
7. **Manual Labeling** (`manual_labeling_per_trip.py`) — Peta interaktif (Folium) untuk verifikasi label secara manual.
8. **Training Set** (`build_train_set.py`) — Membangun dataset training dengan sampling background Non-Event yang proporsional.

## Struktur Folder

```
ml_pipelines/
├── README.md                   ← dokumen ini
├── requirements.txt            ← dependencies Python
│
├── 01_labeling_pipeline/       ← FASE 1: Deteksi & Labeling Kandidat
│   ├── labeling.py             ← orchestrator deteksi
│   ├── ...                     ← modul pipeline (config, helpers, dll)
│   ├── manual_labeling_per_trip.py  ← UI peta interaktif
│   ├── data/                   ← raw data sensor
│   ├── labels/                 ← label JSON per trip
│   └── out/                    ← kandidat event CSV
│
├── 02_dataset_pipeline/        ← FASE 2: Validasi & Persiapan Dataset
│   ├── run_labeling_copilot.py ← orchestrator validasi & split
│   └── config/                 ← taxonomy & pipeline config
│
├── 03_dataset_output/          ← FASE 3: Dataset Akhir (Siap Training)
│
├── 04_quality_control/         ← FASE 4: Audit & Verifikasi
│   ├── labeling_guardrails.py  ← cek konsistensi label
│   └── out/                    ← laporan audit
│
├── artifacts/                  ← dokumen referensi
└── verify_labeling_output.py   ← script verifikasi determinism
```

## Taxonomy Label (3-class)

| Label | Deskripsi |
|-------|-----------|
| `Non-Event` | Semua yang bukan anomali diskrit (normal, engine vibration, maneuver, noise) |
| `Pothole` | Lubang jalan |
| `Speed Bump` | Polisi tidur |

## Setup & Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Utama (Fase 1)

```bash
cd 01_labeling_pipeline
python labeling.py
```

Output akan tersimpan di `01_labeling_pipeline/out/`.

### 3. Labeling Manual

Buka `manual_labeling_per_trip.py` di folder `01_labeling_pipeline` sebagai notebook.
1. Pilih trip via `PILIHAN_INDEX_TRIP`
2. Review event di peta interaktif
3. Edit file label JSON di `labels/<trip_id>_labels.json` (dibuat otomatis)
4. Jalankan sel "Load Labels" untuk memuat label dari JSON
5. Jalankan sel "Simpan" untuk commit ke `ground_truth_labels.csv`

### 4. Build Dataset & Training Set (Fase 2)

```bash
cd 02_dataset_pipeline
python run_labeling_copilot.py
```

Output: `labeling/out/manual_labeled_windows.csv`

## Catatan Teknis

- **Data dikumpulkan pada sepeda motor** — engine vibration tinggi sehingga speed tidak digunakan sebagai hard filter, hanya konteks scoring.
- **Sensor fusion** — gravitasi diestimasi via low-pass Butterworth filter, `a_vertical` adalah proyeksi akselerasi linear terhadap vektor gravitasi.
- **Peak detection** menggunakan fixed threshold (`a_vertical ≥ 3G`) alih-alih adaptive MAD untuk menghindari data leakage antar trip.
