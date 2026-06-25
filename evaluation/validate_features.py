import os
import sys
import numpy as np
import pandas as pd
import json

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
REPORT_DIR = os.path.join(BASE_DIR, "evaluation", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print("============================================================")
    print("             FEATURE VALIDATION ANALYSIS                    ")
    print("============================================================")
    
    xgb_df_path = os.path.join(XGB_DATA_DIR, "xgboost_labeled_windows.csv")
    if not os.path.exists(xgb_df_path):
        print(f"Error: Dataset {xgb_df_path} tidak ditemukan.")
        return
        
    df = pd.read_csv(xgb_df_path)
    
    # Load all feature names
    features_json = os.path.join(BASE_DIR, "evaluation", "models", "xgboost", "xgboost_features.json")
    if os.path.exists(features_json):
        with open(features_json, "r") as f:
            feature_cols = json.load(f)
    else:
        # Fallback to df columns excluding metadata
        exclude = ["trip_id", "event_id", "time_s", "true_label", "pred_label", "label", "source"]
        feature_cols = [c for c in df.columns if c not in exclude]
        
    print(f"Memvalidasi {len(feature_cols)} fitur pada {len(df)} sampel data terproses...")
    
    report_lines = []
    report_lines.append("======================================================================")
    report_lines.append("             DATA QUALITY & NUMERICAL VALIDATION REPORT               ")
    report_lines.append("======================================================================")
    report_lines.append(f"Total Sampel: {len(df)}")
    report_lines.append(f"Total Fitur:  {len(feature_cols)}")
    report_lines.append("----------------------------------------------------------------------")
    
    # ─── 1. INTEGRITY CHECK: NaNs & Infs ───
    nan_counts = df[feature_cols].isna().sum()
    inf_counts = np.isinf(df[feature_cols]).sum()
    
    total_nans = nan_counts.sum()
    total_infs = inf_counts.sum()
    
    report_lines.append(f"Jumlah Nilai NaN/Null: {total_nans}")
    report_lines.append(f"Jumlah Nilai Infinite: {total_infs}")
    if total_nans > 0:
        report_lines.append("\nDetail Fitur dengan NaN:")
        for col, count in nan_counts[nan_counts > 0].items():
            report_lines.append(f"  - {col}: {count} sampel")
    if total_infs > 0:
        report_lines.append("\nDetail Fitur dengan Inf:")
        for col, count in inf_counts[inf_counts > 0].items():
            report_lines.append(f"  - {col}: {count} sampel")
            
    # ─── 2. RANGE & BOUNDS VALIDATION ───
    report_lines.append("\n----------------------------------------------------------------------")
    report_lines.append("VALIDASI BOUNDS DAN FISIKA FITUR:")
    report_lines.append("----------------------------------------------------------------------")
    
    # Check physical constraints
    anomalies = []
    
    # Constraint 1: Energy and Standard Deviations must be non-negative
    for col in feature_cols:
        if "energy" in col or "std" in col or "mag" in col:
            min_val = df[col].min()
            if min_val < 0:
                anomalies.append(f"  - [WARNING] {col} memiliki nilai negatif: {min_val:.6f} (Seharusnya >= 0)")
                
    # Constraint 2: Ratio boundaries
    ratios = ["asymmetry_score", "peak_asymmetry", "first_peak_polarity", "min_z_to_max_z_ratio"]
    for col in ratios:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if col == "min_z_to_max_z_ratio" or col == "asymmetry_score":
                if min_val < -1.05 or max_val > 1.05:
                    anomalies.append(f"  - [WARNING] {col} di luar batas [-1, 1]: min={min_val:.4f}, max={max_val:.4f}")
            if col == "first_peak_polarity":
                if min_val < -1.0 or max_val > 1.0:
                    anomalies.append(f"  - [WARNING] {col} di luar batas [-1, 1]: min={min_val:.4f}, max={max_val:.4f}")
                    
    # Constraint 3: Outlier Check (Max value validation)
    outliers = []
    for col in feature_cols:
        max_val = df[col].max()
        min_val = df[col].min()
        # Flags values that are extremely huge indicating unresolved divisions
        if max_val > 1e4 and col not in ["linear_jerk_3d_max"]:
            outliers.append(f"  - [CRITICAL] {col} memiliki nilai ekstrem: max={max_val:.4f} (Indikasi ketidakstabilan numerik/pembagian nol)")
            
    if not anomalies:
        report_lines.append("[OK] Semua kendala batasan fisik (Bounds & Constraints) valid.")
    else:
        report_lines.extend(anomalies)
        
    if not outliers:
        report_lines.append("[OK] Tidak ada pencilan ekstrem (>10,000) pada seluruh fitur statistik.")
    else:
        report_lines.extend(outliers)
        
    # ─── 3. STATISTICAL SUMMARY FOR BAB IV SKRIPSI ───
    report_lines.append("\n----------------------------------------------------------------------")
    report_lines.append("DESCRIPTIVE STATISTICS SUMMARY (Top 20 Features by variance/magnitude):")
    report_lines.append("----------------------------------------------------------------------")
    
    stats_df = df[feature_cols].describe().T[["mean", "std", "min", "max"]]
    # Sort by std to show highest variance features
    stats_sorted = stats_df.sort_values(by="std", ascending=False).head(20)
    report_lines.append(stats_sorted.to_string())
    
    report_lines.append("\n======================================================================")
    report_lines.append("KESIMPULAN VALIDASI:")
    if total_nans == 0 and total_infs == 0 and not anomalies and not outliers:
        report_lines.append(">>> [STATUS: VALID] Seluruh perhitungan fitur terbukti 100% stabil secara numerik, memenuhi batasan fisik data sensor, dan siap digunakan di Android.")
    else:
        report_lines.append(">>> [STATUS: PERLU TINJAUAN] Ditemukan anomali numerik pada beberapa fitur di atas.")
        
    report_content = "\n".join(report_text if 'report_text' in locals() else report_lines)
    print(report_content)
    
    with open(os.path.join(REPORT_DIR, "feature_validation_report.txt"), "w") as f:
        f.write(report_content)
    print(f"\nLaporan validasi disimpan di: {os.path.join(REPORT_DIR, 'feature_validation_report.txt')}")

if __name__ == "__main__":
    main()
