import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Tambahkan src ke path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import OUT_FOLDER, CSV_FOLDER, CNN_OUT_DIR

# Direktori Output Visualisasi
SAVE_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "reports", "cnn_1d", "skipped_events")
os.makedirs(SAVE_DIR, exist_ok=True)

def get_csv_path_for_trip(trip_id):
    meta_dir = os.path.join(os.path.dirname(CSV_FOLDER), "meta")
    meta_files = glob.glob(os.path.join(meta_dir, "*.json"))
    for jf in meta_files:
        try:
            with open(jf, 'r') as f:
                meta = json.load(f)
            if meta.get("tripId") == str(trip_id):
                csv_name = os.path.basename(jf).replace(".json", ".csv")
                return os.path.join(CSV_FOLDER, csv_name)
        except Exception:
            continue
    return None

def main():
    # 1. Load Ground Truth (Total Candidates)
    gt_path = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
    if not os.path.exists(gt_path):
        print(f"Error: {gt_path} tidak ditemukan.")
        return
        
    df_gt = pd.read_csv(gt_path)
    all_event_ids = set(df_gt["event_id"].values)
    
    # 2. Load Saved CNN Event IDs (Extracted successfully)
    cnn_ids_path = os.path.join(CNN_OUT_DIR, "cnn_1d_event_ids.npy")
    if not os.path.exists(cnn_ids_path):
        print(f"Error: {cnn_ids_path} tidak ditemukan. Silakan jalankan build_cnn_data.py terlebih dahulu.")
        return
        
    cnn_event_ids = set(np.load(cnn_ids_path, allow_pickle=True))
    
    # 3. Find Skipped Events
    skipped_ids = all_event_ids - cnn_event_ids
    print(f"Total kandidat di ground truth : {len(all_event_ids)}")
    print(f"Total kandidat berhasil diekstrak: {len(cnn_event_ids)}")
    print(f"Total kandidat yang DIBUANG    : {len(skipped_ids)}")
    
    if len(skipped_ids) == 0:
        print("Tidak ada event yang dibuang.")
        return
        
    df_skipped = df_gt[df_gt["event_id"].isin(skipped_ids)].copy()
    
    # 4. Plot each skipped event
    sns.set_theme(style="whitegrid")
    
    for _, row in df_skipped.iterrows():
        trip_id = row["trip_id"]
        ev_id = row["event_id"]
        t_event = row["event_peak"]
        label = row["label"]
        
        csv_path = get_csv_path_for_trip(trip_id)
        if not csv_path or not os.path.exists(csv_path):
            print(f"CSV untuk trip {trip_id} tidak ditemukan. Melewati event {ev_id}.")
            continue
            
        try:
            df_raw = pd.read_csv(csv_path)
            # Hitung time dalam detik
            df_raw["time_s"] = df_raw["timestamp"] / 1000.0
            
            # Ambil window [-2.5s, +2.5s] di sekitar t_event agar terlihat konteksnya
            window_start = t_event - 2.5
            window_end = t_event + 2.5
            
            df_window = df_raw[(df_raw["time_s"] >= window_start) & (df_raw["time_s"] <= window_end)].copy()
            
            # Buat plot
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            # Plot Z-Axis Acceleration
            if "az" in df_window.columns:
                ax1.plot(df_window["time_s"], df_window["az"], color="#d62728", linewidth=1.5, label="Z-Axis Accel (az)")
            elif "lin_az" in df_window.columns:
                ax1.plot(df_window["time_s"], df_window["lin_az"], color="#d62728", linewidth=1.5, label="Linear Z-Axis (lin_az)")
                
            ax1.set_xlabel("Time (s)", fontsize=12)
            ax1.set_ylabel("Acceleration (m/s²)", color="#d62728", fontsize=12)
            ax1.tick_params(axis="y", labelcolor="#d62728")
            
            # Garis penanda pusat event
            ax1.axvline(x=t_event, color="black", linestyle="--", linewidth=2, label="Event Center")
            
            # Plot Speed di Y-axis kedua (jika ada)
            if "speed" in df_window.columns:
                ax2 = ax1.twinx()
                ax2.plot(df_window["time_s"], df_window["speed"] * 3.6, color="#1f77b4", linewidth=1.5, label="Speed (km/h)")
                ax2.set_ylabel("Speed (km/h)", color="#1f77b4", fontsize=12)
                ax2.tick_params(axis="y", labelcolor="#1f77b4")
                
                # Gabungkan legenda
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            else:
                ax1.legend(loc="upper right")
            
            # Deteksi alasan mengapa dibuang secara heuristik visual
            # 1. Cek gap
            time_diffs = df_window["time_s"].diff()
            max_gap = time_diffs.max() if not time_diffs.empty else 0
            
            # 2. Cek boundary (apakah durasi window terlalu pendek)
            dur_left = t_event - (df_raw["time_s"].min())
            dur_right = (df_raw["time_s"].max()) - t_event
            
            reason = "Unknown"
            if max_gap > 0.5: # 500ms gap
                reason = "Large Gap Detected (gap_buffer)"
            elif dur_left < 1.0 or dur_right < 1.0:
                reason = "Too close to recording bounds (extract_failed)"
            else:
                reason = "Short contiguous segment (silently dropped)"
                
            plt.title(f"Skipped Event: {label} (ID: {ev_id})\nReason: {reason}", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(SAVE_DIR, f"skipped_{ev_id}.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            print(f"Tersimpan: {save_path} ({reason})")
            
        except Exception as e:
            print(f"Gagal memproses event {ev_id}: {e}")

if __name__ == "__main__":
    main()
