import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import json
from sensor_fusion import resample_100hz

CSV_FOLDER = os.path.join("data", "raw", "active", "csv")

def get_csv_path_for_trip(trip_id):
    meta_dir = os.path.join(os.path.dirname(CSV_FOLDER), "meta")
    meta_files = glob.glob(os.path.join(meta_dir, "*.json"))
    for jf in meta_files:
        try:
            with open(jf, 'r') as f:
                meta = json.load(f)
            if meta.get("tripId") == str(trip_id):
                csv_name = os.path.basename(jf).replace(".json", ".csv")
                csv_path = os.path.join(CSV_FOLDER, csv_name)
                if os.path.exists(csv_path):
                    return csv_path
        except Exception:
            continue
    return None

def main():
    ev_p = {"trip_id": "7fe6874a-33f5-4a42-bc51-972d099b647a", "time_s": 1779865505.37}
    ev_sb = {"trip_id": "fe628252-e600-47d4-875d-7c6f005a735e", "time_s": 1779785408.64}
    ev_ne = {"trip_id": "f5d44722-1c89-4654-a61d-938f66c416e4", "time_s": 1779963042.32}
    
    events = [ev_p, ev_sb, ev_ne]
    titles = ["Pothole", "Speed Bump", "Non-Event"]
    filenames = ["resample_comparison_pothole.png", "resample_comparison_speedbump.png", "resample_comparison_nonevent.png"]
    
    os.makedirs(os.path.join("evaluation", "reports"), exist_ok=True)
    
    for ev, title, filename in zip(events, titles, filenames):
        csv_path = get_csv_path_for_trip(ev["trip_id"])
        if not csv_path: 
            print(f"CSV missing for {ev['trip_id']}")
            continue
            
        df_raw = pd.read_csv(csv_path)
        df_resampled = resample_100hz(df_raw.copy())
        
        t_center = ev["time_s"]
        t_start = t_center - 1.0
        t_end = t_center + 1.0
        
        # Ekstrak window untuk data mentah asli
        df_raw['time_s_col'] = df_raw['timestamp'] / 1000.0
        mask_raw = (df_raw["time_s_col"] >= t_start) & (df_raw["time_s_col"] < t_end)
        window_raw = df_raw[mask_raw].copy()
        
        # Ekstrak window untuk data resampled 100Hz
        df_resampled['time_s_col'] = df_resampled['timestamp'] / 1000.0
        mask_res = (df_resampled["time_s_col"] >= t_start) & (df_resampled["time_s_col"] < t_end)
        window_res = df_resampled[mask_res].copy()
        
        if len(window_raw) == 0 or len(window_res) == 0:
            continue
            
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey='row')
        fig.suptitle(f"True Raw vs Resampled (100Hz) - {title}", fontsize=16, fontweight='bold')
        
        t_plot_raw = window_raw["time_s_col"] - t_start
        t_plot_res = window_res["time_s_col"] - t_start
        
        # Accel X, Y, Z ke bawah
        for row, (col, ax_name) in enumerate([("ax", 'X'), ("ay", 'Y'), ("az", 'Z')]):
            # Left: Original Raw
            ax_left = axes[row, 0]
            ax_left.plot(t_plot_raw, window_raw[col], marker='o', markersize=3, linestyle='--', color='gray', alpha=0.8, label='Original Raw')
            if ax_name == 'Z':
                ax_left.axhline(y=9.81, color='black', linestyle=':', alpha=0.8, label='Gravity 1G')
            ax_left.set_ylabel(f"Accel {ax_name} (m/s²)", fontsize=12)
            ax_left.grid(True, linestyle='--', alpha=0.7)
            ax_left.legend(loc='upper right')
            
            # Right: Resampled
            ax_right = axes[row, 1]
            ax_right.plot(t_plot_res, window_res[col], linewidth=2, color='blue', label='Resampled 100Hz')
            if ax_name == 'Z':
                ax_right.axhline(y=9.81, color='black', linestyle=':', alpha=0.8, label='Gravity 1G')
            ax_right.grid(True, linestyle='--', alpha=0.7)
            ax_right.legend(loc='upper right')
            
        axes[0, 0].set_title("Original Raw (Irregular Interval)", fontsize=14, fontweight='bold')
        axes[0, 1].set_title("Resampled 100Hz (Interpolated 10ms)", fontsize=14, fontweight='bold')
            
        axes[-1, 0].set_xlabel("Waktu (Detik) dalam Window 2.0s", fontsize=12)
        axes[-1, 1].set_xlabel("Waktu (Detik) dalam Window 2.0s", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        out_dir = os.path.join("evaluation", "reports", "waveforms", "comparison_resample")
        os.makedirs(out_dir, exist_ok=True)
        
        out_f = os.path.join(out_dir, filename)
        plt.savefig(out_f, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualisasi perbandingan resample untuk {title} berhasil disimpan ke: {out_f}")

if __name__ == "__main__":
    main()
