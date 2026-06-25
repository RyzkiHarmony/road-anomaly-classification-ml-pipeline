# alignment_analysis.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ks_2samp
from sensor_fusion import apply_sensor_fusion
from config import OUT_FOLDER, get_logger

logger = get_logger(__name__)

def run_alignment_analysis():
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "new-data", "csv")
    new_files = glob.glob(os.path.join(csv_dir, "*2026-05-26*.csv"))
    
    if not new_files:
        logger.error("No 26 Mei files found in data/csv/ to perform alignment analysis.")
        return

    # Gunakan file pertama untuk analisis
    file_path = new_files[0]
    filename = os.path.basename(file_path)
    logger.info(f"Loading data from {filename} for alignment verification...")
    
    df_raw = pd.read_csv(file_path)
    
    # ── JALUR 1: SOFTWARE SEPARATION (Filtfilt Offline) ──
    # Drop kolom native untuk memaksa apply_sensor_fusion masuk ke fallback Butterworth LPF
    df_legacy_sim = df_raw.drop(columns=["lin_ax", "lin_ay", "lin_az", "grav_x", "grav_y", "grav_z"])
    logger.info("Processing Path A: Traditional Software Separation (Filtfilt Offline)...")
    df_software = apply_sensor_fusion(df_legacy_sim, offline=True)
    
    # ── JALUR 2: HARDWARE SEPARATION (Android Native Hardware Fusion) ──
    logger.info("Processing Path B: Android Native Hardware Sensor Fusion...")
    df_native = apply_sensor_fusion(df_raw)
    
    # Sinkronisasi index / timestamp kedua output
    # Karena resample_100hz menghasilkan datetime index yang sama, kita bisa merge
    df_compare = pd.DataFrame({
        "timestamp": df_software["timestamp"],
        "a_vert_software": df_software["a_vertical"] / 9.80665,  # Convert to G-force
        "a_vert_native": df_native["a_vertical"] / 9.80665       # Convert to G-force
    }).dropna()
    
    # ── HITUNG STATISTIK ALINYEMEN ──
    soft_arr = df_compare["a_vert_software"].values
    nat_arr = df_compare["a_vert_native"].values
    
    # 1. Pearson Correlation (Keselarasan Fasa/Bentuk Sinyal)
    corr_coef, _ = pearsonr(soft_arr, nat_arr)
    
    # 2. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((soft_arr - nat_arr) ** 2))
    
    # 3. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(soft_arr - nat_arr))
    
    # 4. Kolmogorov-Smirnov Test (Kesamaan Distribusi)
    ks_stat, ks_p_val = ks_2samp(soft_arr, nat_arr)
    
    print("\n" + "="*60)
    print(f"{'DSP SENSOR ALIGNMENT ANALYSIS REPORT':^60}")
    print("="*60)
    print(f"File Tested        : {filename}")
    print(f"Total Samples      : {len(df_compare)}")
    print(f"Pearson Correlation: {corr_coef:.4f}  (Target: >0.90)")
    print(f"RMSE (G-force)     : {rmse:.4f}")
    print(f"MAE (G-force)      : {mae:.4f}")
    print(f"KS-Test Statistic  : {ks_stat:.4f}")
    print(f"KS-Test p-value    : {ks_p_val:.4e}")
    print("-" * 60)
    
    if corr_coef >= 0.90:
        print("STATUS: SUCCESS (Sinyal Software Filtfilt terbukti selaras dengan Hardware Native!)")
    else:
        print("STATUS: WARNING (Ada deviasi fasa/bentuk sinyal antara Software vs Hardware!)")
    print("="*60 + "\n")
    
    # ── BUAT GRAFIK PLOT VISUALISASI ──
    # Temukan jendela 15 detik di sekitar peak tertinggi untuk visualisasi dinamis
    # (Ini menghindari plotting bagian jalan rata yang membosankan)
    max_idx = np.argmax(np.abs(soft_arr))
    time_s = (df_compare["timestamp"] - df_compare["timestamp"].iloc[0]) / 1000.0
    
    t_center = time_s.iloc[max_idx]
    t_start = max(0, t_center - 7.5)
    t_end = min(time_s.iloc[-1], t_center + 7.5)
    
    mask_zoom = (time_s >= t_start) & (time_s <= t_end)
    zoom_time = time_s[mask_zoom]
    zoom_soft = df_compare["a_vert_software"][mask_zoom]
    zoom_nat = df_compare["a_vert_native"][mask_zoom]
    
    plt.figure(figsize=(14, 6))
    
    # Plot Sinyal
    plt.plot(zoom_time, zoom_soft, label="Legacy Path (Software Filtfilt LPF)", color="#1f77b4", alpha=0.85, linewidth=2.0)
    plt.plot(zoom_time, zoom_nat, label="New Path (Android Hardware Fusion)", color="#ff7f0e", alpha=0.85, linewidth=1.8, linestyle="--")
    
    plt.title(f"DSP SENSOR ALIGNMENT VALIDATION\nFile: {filename} (Zoom-in 15 Detik di Sekitar Peak Utama)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Waktu Perjalanan (Detik)", fontsize=12)
    plt.ylabel("Akselerasi Vertikal (G-force)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Tambahkan anotasi teks performa di dalam plot
    stats_text = (
        f"Alinyemen Hasil Pemrosesan Sinyal:\n"
        f"• Pearson Correlation: {corr_coef:.4f}\n"
        f"• RMSE: {rmse:.4f} G\n"
        f"• MAE: {mae:.4f} G\n"
        f"• KS-Test p-val: {ks_p_val:.2e}"
    )
    plt.gca().text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=11,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dcdcdc", alpha=0.9))
    
    plt.legend(loc="upper right", fontsize=11, frameon=True, facecolor="white", edgecolor="#dcdcdc")
    plt.tight_layout()
    
    # Simpan plot ke out folder
    plot_path = os.path.join(OUT_FOLDER, "sensor_alignment_validation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Visualisation plot successfully saved at: {plot_path}")

if __name__ == "__main__":
    run_alignment_analysis()
