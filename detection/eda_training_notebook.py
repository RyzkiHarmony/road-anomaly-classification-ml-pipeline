# %% [markdown]
# # Tahapan 3: EDA & Model Training (Random Forest)
# 
# Notebook ini merupakan alur kerja untuk:
# 1. Mengeksplorasi dataset berlabel yang telah selesai dibuat (`auto_labeled_windows.csv`)
# 2. Mengukur tingkat kepentingan setiap fitur (Feature Importance).
# 3. Melatih dan memvalidasi Model Deteksi Jalan (berbasis file `detection.py`).
# 4. Mengekspor file Model yang siap-*deploy*.
# 
# *Pastikan file `detection.py` dan `auto_labeled_windows.csv` berada pada jalurnya.*

# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import Kelas Model utama kita
from detection import DamageDetector

import warnings
warnings.filterwarnings('ignore')

# Konfigurasi Path File
_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_PATH = os.path.join(_DIR, "..", "labeling", "out", "manual_labeled_windows.csv")

# %% [markdown]
# ## 1. Inisialisasi Model & Memuat Data
# Kita akan menggunakan modul kelas `DamageDetector` yang dikembangkan, yang otomatis menangani filter fitur dan penyesuaian Missing Value.

# %%
print("Inisialisasi Detektor...")
detector = DamageDetector(model_path="road_damage_model_v1.pkl")

print(f"\nMencoba memuat data dari: {DATA_PATH}")
df = detector.load_training_data(DATA_PATH)

if df is not None:
    print("\nSekilas Data Latih:")
    display(df[detector.feature_columns + ['label']].head())
    print("\nDistribusi Label Akhir:")
    print(df['label'].value_counts())

# %% [markdown]
# ## 2. Exploratory Data Analysis (Visualisasi Sederhana)
# Mari kita lihat apakah data kita mudah dipisahkan secara linier atau harus menggunakan pohon kompleks (Random Forest).
# Kita plot `Speed` dengan `Magnitude Maximal` yang diderita HP.

# %%
if df is not None:
    plt.figure(figsize=(10, 6))
    
    # Kita sample data agar visualisasi tidak terlalu berat jika data >50k
    df_sample = df.sample(min(5000, len(df)), random_state=42)
    
    sns.scatterplot(
        data=df_sample, 
        x='speed_mean', y='mag_max', 
        hue='label', 
        palette={'Normal':'#2ecc71', 'Severe Anomaly':'#e74c3c'},
        alpha=0.6, s=30
    )
    plt.title('Hubungan Kecepatan Kendaraan vs Guncangan Max (Max Magnitude)')
    plt.xlabel('Kecepatan (m/s)')
    plt.ylabel('Max Magnitude (m/s^2)')
    plt.legend(title='Jenis Kondisi')
    
    # Tambahkan garis ambang batas 
    plt.axhline(39.2, color='green', linestyle='--', alpha=0.5, label='Batas Normal (4G)')
    plt.axhline(73.5, color='red', linestyle='--', alpha=0.5, label='Batas Rusak (7.5G)')
    plt.show()

# %% [markdown]
# ## 3. Memulai Proses Training Produksi
# Mulai memanggil metode `train()` dan `cross_validate()` dari *backend class* utama `detection.py`.
# Algoritma tersebut secara otomatis sudah disisipkan metode SMOTE dan Balanced-weight!

# %%
if df is not None:
    # Memisahkan matriks Fitur (X), Target (y), dan Grup (trip_id)
    X = df[detector.feature_columns]
    y = df['label']
    groups = df['trip_id']  # <-- Kunci Trip-based Split!
    
    print(f"Total Trip Unik: {groups.nunique()}")
    print(f"Total Windows  : {len(X)}")
    
    # Training dengan Trip-based Split
    trained_model = detector.train(X, y, groups=groups, test_size=0.2)
    
    # Skoring K-Fold berbasis Trip (GroupKFold)
    detector.cross_validate(X, y, groups=groups, n_splits=5)

# %% [markdown]
# ## 4. Plotting Confusion Matrix
# Kita buat gambar *Confusion Matrix* agar lebih mudah dianalisis. Semakin tebal warna diagonal yang membentang dari Kiri Atas ke Kanan Bawah, berarti semakin hebat model Anda.

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if df is not None and detector.model is not None:
    # Menggunakan data test yang sudah tersimpan dari trip-based split
    X_test = detector._last_X_test
    y_test = detector._last_y_test
    
    preds = detector.model.predict(X_test)
    
    cm = confusion_matrix(y_test, preds, labels=detector.model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=detector.model.classes_)
    
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Trip-based Test Set)')
    plt.show()

# %% [markdown]
# ## 5. Analisis Fitur Terpenting (Feature Importances)
# Mana dari 6 fitur sensor tersebut yang paling membantu ML dalam menebak ada jalan berlubang?

# %%
if df is not None and detector.model is not None:
    importances = detector.model.feature_importances_
    cols = detector.feature_columns
    
    # Mengurutkan dari yang terbesar
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 4))
    plt.title("Feature Importances untuk Pothole Detection")
    plt.bar(range(len(cols)), importances[indices], color='indigo', align="center")
    plt.xticks(range(len(cols)), [cols[i] for i in indices], rotation=25)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Simpan Model Akhir (Export/Persistence)
# Terakhir, kita menyimpannya menjadi file `.pkl`.
# Ini adalah model yang akan Anda pakai di Aplikasi Android/Django backend ke depannya.

# %%
if df is not None and detector.model is not None:
    detector.save_model()
    print("MANTAP! Model siap digunakan =)")

# %%
