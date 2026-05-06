import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setup paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CANDIDATES_PATH = os.path.join(BASE_DIR, "01_labeling_pipeline", "out", "candidates_events.csv")
GT_PATH = os.path.join(BASE_DIR, "01_labeling_pipeline", "out", "ground_truth_labels.csv")
REPORT_DIR = os.path.join(SCRIPT_DIR, "out")

os.makedirs(REPORT_DIR, exist_ok=True)

def run_audit():
    """
    Menganalisis separabilitas fitur (kemampuan fitur membedakan kelas).
    Membuktikan secara statistik bahwa Pothole vs Speed Bump memiliki 
    karakteristik fisik yang berbeda nyata.
    """
    if not os.path.exists(CANDIDATES_PATH) or not os.path.exists(GT_PATH):
        print(f"[ERROR] File kandidat atau ground truth tidak ditemukan.")
        print(f"Pastikan sudah melakukan labeling di 01_labeling_pipeline.")
        return

    df_cand = pd.read_csv(CANDIDATES_PATH)
    df_gt = pd.read_csv(GT_PATH)

    # Merge untuk mendapatkan fitur dari event yang sudah dilabeli
    df = pd.merge(df_gt, df_cand, on=["trip_id", "event_id"], how="inner")
    
    if df.empty:
        print("[ERROR] Tidak ada kecocokan data antara Ground Truth dan Candidates.")
        return

    # Filter hanya untuk event diskrit (abaikan Non-Event agar perbandingan fokus pada anomali)
    classes = ["Pothole", "Speed Bump"]
    df_filtered = df[df["label"].isin(classes)].copy()
    
    # Hitung distribusi label
    counts = df_filtered["label"].value_counts()
    print("\n" + "="*100)
    print("FEATURE SEPARABILITY & STATISTICAL INTEGRITY AUDIT")
    print("="*100)
    print(f"Jumlah sampel terlabeli: {len(df_filtered)}")
    for cls in classes:
        print(f" - {cls:<12}: {counts.get(cls, 0)} sampel")
    
    if len(counts) < 2 or any(counts < 3):
        print("\n[WARN] Sampel terlalu sedikit untuk analisis statistik yang valid.")
        return

    # Fitur-fitur kunci yang ingin kita uji bedanya
    features_to_check = [
        ("vert_jrk", "Vertical Jerk (m/s³)"),
        ("peak_vertical_g", "Peak Vertical G-force"),
        ("asymmetry_score", "Asymmetry Score"),
        ("gyro_pitch_roll_ratio", "Gyro Pitch/Roll Ratio"),
        ("fft_high_low_ratio", "FFT High/Low Ratio"),
        ("duration_above_threshold", "Impact Duration (s)")
    ]

    print("\n" + "-"*100)
    print(f"{'Feature Metric':<30} | {'Pothole (Mean)':<15} | {'Speed Bump (Mean)':<15} | {'P-Value':<10}")
    print("-" * 100)

    # Plot configuration
    fig, axes = plt.subplots(len(features_to_check), 1, figsize=(10, 4 * len(features_to_check)))
    if len(features_to_check) == 1: axes = [axes]

    for i, (feat, label_name) in enumerate(features_to_check):
        if feat not in df_filtered.columns:
            continue
            
        # Pisahkan grup
        p_vals = df_filtered[df_filtered["label"] == "Pothole"][feat].dropna()
        sb_vals = df_filtered[df_filtered["label"] == "Speed Bump"][feat].dropna()
        
        # Hitung T-Test (Welch's t-test karena varians bisa berbeda)
        t_stat, p_val = stats.ttest_ind(p_vals, sb_vals, equal_var=False)
        
        p_mean = p_vals.mean()
        sb_mean = sb_vals.mean()
        
        # Formatting p-value
        p_val_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        significance = " (VALID)" if p_val < 0.05 else " (Weak)"
        
        print(f"{label_name:<30} | {p_mean:<15.2f} | {sb_mean:<15.2f} | {p_val_str}{significance}")
        
        # Boxplot visualization
        sns.boxplot(x="label", y=feat, hue="label", data=df_filtered, ax=axes[i], palette=["#ef4444", "#3b82f6"], legend=False)
        axes[i].set_title(f"Separability: {label_name} (p={p_val_str})", fontsize=12)
        axes[i].set_ylabel("")
        axes[i].set_xlabel("")

    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "feature_separability_audit.png")
    plt.savefig(plot_path, dpi=150)
    
    print("-" * 100)
    print("\n[SUKSES] Audit Kelayakan selesai.")
    print(f"Grafik Boxplot Separabilitas disimpan ke: 04_quality_control/out/feature_separability_audit.png")
    print("Interpretasi: Fitur dengan P-Value < 0.05 membuktikan bahwa sensor Anda secara akurat ")
    print("mampu membedakan Pothole vs Speed Bump secara fisik. Ini adalah bukti kuat untuk skripsi.")

if __name__ == "__main__":
    # Ensure REPORT_DIR exists inside the function to use correct SCRIPT_DIR
    REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(REPORT_DIR, exist_ok=True)
    run_audit()
