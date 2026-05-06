import pandas as pd
import numpy as np
import glob
import os
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt

# Setup paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(BASE_DIR, "01_labeling_pipeline", "data", "csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "out")

os.makedirs(OUT_DIR, exist_ok=True)

def analyze_sampling():
    """
    Menganalisis interval antar baris data (dt) untuk mengetahui 
    sampling rate (Hz) aktual dan tingkat jitter (ketidakteraturan).
    """
    csv_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
    if not csv_files:
        print(f"[ERROR] Tidak ditemukan file CSV di {RAW_DATA_DIR}")
        return

    print("\n" + "="*100)
    print(f"{'File Name':<45} | {'Median Hz':<10} | {'Mean Hz':<10} | {'Std ms':<8} | {'Max ms':<6}")
    print("-" * 100)

    all_stats = []

    for file_path in csv_files:
        # Load hanya kolom timestamp untuk kecepatan
        try:
            df = pd.read_csv(file_path, usecols=['timestamp'])
        except Exception:
            continue
            
        ts = df['timestamp'].astype(float).values
        
        # Hitung interval (dt) dalam milidetik
        dt = np.diff(ts)
        
        # Filter dt yang tidak valid (0 atau negatif karena bug logging)
        dt_clean = dt[dt > 0]
        
        if len(dt_clean) == 0:
            continue
            
        med_dt = np.median(dt_clean)
        mean_dt = np.mean(dt_clean)
        std_dt = np.std(dt_clean)
        max_dt = np.max(dt_clean)
        
        # Konversi ms ke Hz (1000 / ms)
        med_hz = 1000.0 / med_dt
        mean_hz = 1000.0 / mean_dt
        
        file_name = os.path.basename(file_path)
        print(f"{file_name[:45]:<45} | {med_hz:<10.2f} | {mean_hz:<10.2f} | {std_dt:<8.2f} | {max_dt:<6.0f}")
        
        all_stats.append({
            'file': file_name,
            'dt': dt_clean,
            'med_hz': med_hz
        })

    # Buat visualisasi histogram untuk file pertama sebagai sampel jitter
    if all_stats:
        sample = all_stats[0]
        plt.figure(figsize=(10, 6))
        
        # Batasi range histogram agar jitter terlihat jelas (misal 0 - 100ms)
        plt.hist(sample['dt'], bins=100, range=(0, 100), color='#3b82f6', edgecolor='white', alpha=0.8)
        
        plt.axvline(np.median(sample['dt']), color='red', linestyle='dashed', linewidth=1.5, label=f"Median: {np.median(sample['dt']):.1f}ms")
        
        plt.title(f"Distribusi Interval Sampling (Jitter)\nFile: {sample['file']}", fontsize=12)
        plt.xlabel("Interval dt (ms)", fontsize=10)
        plt.ylabel("Frekuensi", fontsize=10)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plot_path = os.path.join(OUT_DIR, "sampling_jitter_analysis.png")
        plt.savefig(plot_path, dpi=150)
        print("="*100)
        print(f"\n[SUKSES] Analisis selesai.")
        print(f"📊 Plot histogram jitter disimpan ke: 04_quality_control/out/sampling_jitter_analysis.png")
        print(f"💡 Interpretasi: Jika Std ms tinggi (> 5ms), data Anda sangat tidak teratur (jittery).")

if __name__ == "__main__":
    analyze_sampling()
