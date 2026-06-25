import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "data", "raw", "active", "csv")
REPORT_DIR = os.path.join(BASE_DIR, "evaluation", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def analyze_trip(file_path):
    df = pd.read_csv(file_path)
    # Ensure lin_az exists
    if 'lin_az' not in df.columns:
        return None
        
    lin_az = df['lin_az'].dropna().values
    speed = df['speed'].dropna().values
    
    # Basic Stats
    std_az = np.std(lin_az)
    max_az = np.max(np.abs(lin_az))
    p95_az = np.percentile(np.abs(lin_az), 95)
    mean_speed = np.mean(speed) if len(speed) > 0 else 0.0
    
    # Power Spectral Density (PSD)
    # Target HZ is 100
    freqs, psd = welch(lin_az, fs=100.0, nperseg=1024)
    # Average power in 5-20Hz (typical road anomaly range) vs 20-50Hz (vibration noise)
    power_low = np.mean(psd[(freqs >= 2) & (freqs <= 15)])
    power_high = np.mean(psd[(freqs > 15) & (freqs <= 45)])
    
    return {
        "filename": os.path.basename(file_path),
        "std_az": std_az,
        "max_az": max_az,
        "p95_az": p95_az,
        "mean_speed": mean_speed,
        "power_low": power_low,
        "power_high": power_high,
        "freqs": freqs,
        "psd": psd,
        "lin_az_sample": lin_az[10000:15000] if len(lin_az) > 15000 else lin_az
    }

def main():
    print("============================================================")
    print("             TRIP ANOMALY DATA AUDIT (OPTI C)               ")
    print("============================================================")
    
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print("Error: Tidak ada file CSV ditemukan di data/raw/active/csv/")
        return
        
    anomaly_trip_file = "RoadDamage_2026-05-27_14-07-07.csv"
    anomaly_path = os.path.join(CSV_DIR, anomaly_trip_file)
    
    if not os.path.exists(anomaly_path):
        print(f"Error: File anomali {anomaly_trip_file} tidak ditemukan.")
        return
        
    print(f"Menganalisis trip anomali: {anomaly_trip_file}...")
    anomaly_stats = analyze_trip(anomaly_path)
    
    # Analyze other trips for comparison
    comparison_stats = []
    for f in csv_files:
        if os.path.basename(f) == anomaly_trip_file:
            continue
        stats = analyze_trip(f)
        if stats:
            comparison_stats.append(stats)
            
    # Find a normal comparison trip (closest to median std_az)
    stds = [s["std_az"] for s in comparison_stats]
    median_idx = np.argsort(stds)[len(stds) // 2]
    normal_stats = comparison_stats[median_idx]
    
    print(f"Trip kontrol terpilih untuk perbandingan: {normal_stats['filename']} (std_az: {normal_stats['std_az']:.4f})")
    
    # ─── 1. WRITE REPORT ───
    report_text = []
    report_text.append("======================================================================")
    report_text.append("             DATA QUALITY AUDIT REPORT: TRIP ANOMALY                  ")
    report_text.append("======================================================================")
    report_text.append(f"Trip Anomali (dfb4f7d0): {anomaly_stats['filename']}")
    report_text.append(f"Trip Kontrol (Normal):   {normal_stats['filename']}")
    report_text.append("----------------------------------------------------------------------")
    report_text.append(f"{'Metric':<30} | {'Anomaly Trip':<15} | {'Normal Trip':<15} | {'Ratio (Anom/Norm)':<15}")
    report_text.append("-" * 80)
    
    metrics = [
        ("Std Dev (lin_az)", "std_az", ".4f"),
        ("Max Absolute (lin_az)", "max_az", ".4f"),
        ("95th Percentile (lin_az)", "p95_az", ".4f"),
        ("Mean Speed (m/s)", "mean_speed", ".2f"),
        ("PSD Low Freq (2-15Hz)", "power_low", ".6f"),
        ("PSD High Freq (15-45Hz)", "power_high", ".6f"),
    ]
    
    for label, key, fmt in metrics:
        val_anom = anomaly_stats[key]
        val_norm = normal_stats[key]
        ratio = val_anom / (val_norm + 1e-9)
        val_anom_str = f"{val_anom:{fmt}}"
        val_norm_str = f"{val_norm:{fmt}}"
        report_text.append(f"{label:<30} | {val_anom_str:<15} | {val_norm_str:<15} | {ratio:<15.2f}")
        
    report_text.append("======================================================================")
    report_text.append("\nTEMUAN & DIAGNOSIS:")
    
    ratio_high = anomaly_stats["power_high"] / (normal_stats["power_high"] + 1e-9)
    ratio_low = anomaly_stats["power_low"] / (normal_stats["power_low"] + 1e-9)
    ratio_std = anomaly_stats["std_az"] / normal_stats["std_az"]
    
    if ratio_high > 3.0:
        report_text.append(f"- Anomali Vibrasi Frekuensi Tinggi: Energi frekuensi tinggi (15-45Hz) trip anomali {ratio_high:.1f}x lipat lebih tinggi dari trip normal.")
        report_text.append("  Hal ini menunjukkan getaran motor yang sangat kuat atau pemasangan ponsel pada dudukan (holder) yang tidak stabil/kendor.")
    if ratio_std > 1.5:
        report_text.append(f"- Deviasi Standar Sinyal Berlebihan: Variansi akselerasi vertikal {ratio_std:.1f}x lebih besar.")
        report_text.append("  Model XGBoost dan CNN salah mengklasifikasikan ini sebagai lubang karena fluktuasi akselerasi konstan menyerupai shock jalan.")
    
    report_content = "\n".join(report_text)
    print(report_content)
    
    with open(os.path.join(REPORT_DIR, "trip_anomaly_report.txt"), "w") as f:
        f.write(report_content)
    print(f"\nLaporan tertulis disimpan di: {os.path.join(REPORT_DIR, 'trip_anomaly_report.txt')}")
    
    # ─── 2. PLOT COMPARISONS ───
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Waveform slice comparison
    ax = axes[0]
    time_arr = np.arange(len(anomaly_stats["lin_az_sample"])) / 100.0
    ax.plot(time_arr, anomaly_stats["lin_az_sample"], label=f"Anomaly ({anomaly_trip_file})", color='red', alpha=0.7)
    ax.plot(time_arr, normal_stats["lin_az_sample"][:len(time_arr)], label=f"Normal ({normal_stats['filename']})", color='blue', alpha=0.5)
    ax.set_title("Perbandingan Amplitudo Vertikal Sinyal Mentah (50 Detik)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Akselerasi Vertikal (m/s²)")
    ax.legend()
    ax.grid(True)
    
    # Plot 2: PSD comparison
    ax = axes[1]
    ax.semilogy(anomaly_stats["freqs"], anomaly_stats["psd"], label="Anomaly Trip", color='red')
    ax.semilogy(normal_stats["freqs"], normal_stats["psd"], label="Normal Trip", color='blue')
    ax.set_title("Analisis Spektrum Frekuensi (PSD lin_az)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frekuensi (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Boxplot of all trips standard deviation
    ax = axes[2]
    all_stds = [s["std_az"] for s in comparison_stats] + [anomaly_stats["std_az"]]
    ax.boxplot(all_stds, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
    ax.scatter([anomaly_stats["std_az"]], [1], color='red', s=100, zorder=5, label="Anomaly Trip (dfb4f7d0)")
    ax.set_title("Distribusi Deviasi Standar Sinyal (std_az) di Seluruh Trip", fontsize=12, fontweight='bold')
    ax.set_xlabel("Standard Deviation (m/s²)")
    ax.set_yticks([])
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "trip_anomaly_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grafik visualisasi disimpan di: {plot_path}")

if __name__ == "__main__":
    main()
