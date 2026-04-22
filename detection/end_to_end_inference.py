# %% [markdown]
# # Tahap Akhir (Fase 4): End-to-End Inference & Mapping
# 
# Puncak dari proyek Anda. Skrip interaktif ini mensimulasikan Production API Backend:
# 1. Menerima data MENTAH dari ponsel (Sensor Accel & Gyro).
# 2. Melakukan ekstraksi *Sliding Window* seketika.
# 3. Menggunakan "Otak" AI (`road_damage_model_v1.pkl`) untuk mendeteksi kerusakan jalan.
# 4. Menggambar rute perjalanan Anda di sistem peta (Folium) berbasis Segmen Kerusakan.

# %%
import pandas as pd
import numpy as np
import os
import glob
import sys

# Konfigurasi Path File
_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
RAW_CSV_DIR = os.path.join(_DIR, "..", "labeling", "data", "csv")

# Supaya kita bisa impor script labeling.py dari folder sebelah
sys.path.append(os.path.join(_DIR, "..", "labeling"))
from labeling import extract_windows_features

from detection import DamageDetector
from aggregation import MapAggregator

# %% [markdown]
# ## 1. Mengambil Satu Trip Mentah dari Aplikasi
# Anggap saja ada motor yang baru selesai berkendara dan mengunggah CSV-nya ke server.

# %%
print("Mencari data mentah Android terbaru...")
raw_csvs = glob.glob(os.path.join(RAW_CSV_DIR, "*.csv"))

if not raw_csvs:
    print("Tidak ada file CSV mentah yang ditemukan!")
    sys.exit(0)

# Ambil perjalanan terbaru (simulasi data fresh dari HP)
latest_trip_path = max(raw_csvs, key=os.path.getctime)
trip_name = os.path.basename(latest_trip_path)

print(f"\n---> Menerima Kiriman Trip: {trip_name}")
df_raw = pd.read_csv(latest_trip_path)

# Jika kolom mutlak magnitude belum dikalkulasi Android, kita bantu hitungkan.
if "magnitude" not in df_raw.columns:
    df_raw["magnitude"] = np.sqrt(df_raw["ax"] ** 2 + df_raw["ay"] ** 2 + df_raw["az"] ** 2)

print(f"Total baris mentah (Samples) yang diunduh: {len(df_raw)} sampel per sekian ms.")

# %% [markdown]
# ## 2. Ekstraksi Fitur Berjalan (Sliding Window Multisensor)
# Kita putar engkol ekstraksi yang sama dengan yang kita gunakan saat skripsi training.

# %%
print("\nMengekstrak 12 Parameter Fitur (Accel + Giroskop) menggunakan Sliding Window...")
df_trip_windows = extract_windows_features(df_raw)

print(f"Berhasil diubah menjadi {len(df_trip_windows)} detik (Windows) siap prediksi.")
df_trip_windows.head(3)

# %% [markdown]
# ## 3. AI Inference (Multisensor Model)
# Kita nyalakan model `.pkl` kita, masukkan semua 12 Gyro/Accel param, dan tunggu putusannya.

# %%
print("\nMenghidupkan Model Deteksi Multisensor...")
detector = DamageDetector(model_path="road_damage_model_v1.pkl")
detector.load_model()

# Cek kesesuaian kolom
missing_cols = [c for c in detector.feature_columns if c not in df_trip_windows.columns]
if missing_cols:
    print(f"[FATAL ERROR] Data hasil ekstraksi kehilangan fitur: {missing_cols}")
    sys.exit(1)

print("\nModel AI sedang menscoring kualitas aspal per detik...")
predictions = detector.predict(df_trip_windows)

# Gabungkan hasil tebakan AI dengan lokasi GPS dan atribut window-nya
df_inference = pd.concat([df_trip_windows, predictions], axis=1)

print("\nHasil Putusan Kondisi Jalan:")
print(df_inference['prediction'].value_counts().to_string())

# %% [markdown]
# ## 4. Pemetaan GPS (Folium Segments)
# Terjemahkan putusan AI menjadi segmen-segmen garis rute peta merah-kuning-hijau.

# %%
print("\nMemanggil Algoritma Pembuat Peta GPS...")

aggregator = MapAggregator()
# Potong nama file agar ringkas
output_map_path = os.path.join(_DIR, f"inference_map_{trip_name[:8]}.html")

# Draw the map!
my_map = aggregator.create_damage_map(df_inference, output_html=output_map_path)

print(f"\n[SELESAI] API Simulation Berakhir. Silakan buka file: {output_map_path}")
# my_map
