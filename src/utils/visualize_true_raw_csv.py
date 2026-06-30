import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import json

CSV_FOLDER = os.path.join("data", "raw", "active", "csv")
GT_PATH = os.path.join("data", "processed", "shared", "ground_truth_labels.csv")
EVENTS_PATH = os.path.join("data", "processed", "shared", "candidates_events.csv")

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
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        print("Missing dataset files.")
        return

    # Menggunakan event yang persis sama dengan yang divisualisasikan di script sebelumnya (hasil dari ranking max vertical acceleration)
    ev_p = {"trip_id": "7fe6874a-33f5-4a42-bc51-972d099b647a", "time_s": 1779865505.37}
    ev_sb = {"trip_id": "fe628252-e600-47d4-875d-7c6f005a735e", "time_s": 1779785408.64}
    ev_ne = {"trip_id": "f5d44722-1c89-4654-a61d-938f66c416e4", "time_s": 1779963042.32}
    
    events = [ev_p, ev_sb, ev_ne]
    titles = ["Pothole", "Speed Bump", "Non-Event"]
    filenames = ["true_raw_sensor_pothole.png", "true_raw_sensor_speedbump.png", "true_raw_sensor_nonevent.png"]
    
    os.makedirs(os.path.join("evaluation", "reports"), exist_ok=True)
    
    for ev, title, filename in zip(events, titles, filenames):
        csv_path = get_csv_path_for_trip(ev["trip_id"])
        if not csv_path: 
            print(f"CSV missing for {ev['trip_id']}")
            continue
            
        df = pd.read_csv(csv_path)
        
        t_center = ev["time_s"]
        t_start = t_center - 1.0
        t_end = t_center + 1.0
        
        df['time_s_col'] = df['timestamp'] / 1000.0
        mask = (df["time_s_col"] >= t_start) & (df["time_s_col"] < t_end)
        window = df[mask].copy()
        
        if len(window) == 0:
            continue
            
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"True Raw Sensor Data (Includes Gravity) - {title}", fontsize=16, fontweight='bold')
        
        t_plot = window["time_s_col"] - t_start
        
        # Accel (Left Column)
        for row, (col, ax_name, color) in enumerate([("ax", 'X', 'red'), ("ay", 'Y', 'green'), ("az", 'Z', 'blue')]):
            ax = axes[row, 0]
            ax.plot(t_plot, window[col], color=color, linewidth=1.5)
            if ax_name == 'Z':
                ax.axhline(y=9.81, color='black', linestyle='--', alpha=0.5, label='Gravity 1G')
                ax.legend(loc='upper right', fontsize=8)
            ax.set_ylabel(f"Accel {ax_name} (m/s²)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(-60, 60)
            
        # Gyro (Right Column)
        for row, (col, ax_name, color) in enumerate([("gx", 'X', 'red'), ("gy", 'Y', 'green'), ("gz", 'Z', 'blue')]):
            ax = axes[row, 1]
            ax.plot(t_plot, window[col], color=color, linewidth=1.5)
            ax.set_ylabel(f"Gyro {ax_name} (rad/s)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(-15, 15)
            
        axes[0, 0].set_title("True Raw Accelerometer", fontsize=12, fontweight='bold')
        axes[0, 1].set_title("True Raw Gyroscope", fontsize=12, fontweight='bold')
        
        axes[2, 0].set_xlabel("Waktu (Detik) dalam Window 2.0s", fontsize=12)
        axes[2, 1].set_xlabel("Waktu (Detik) dalam Window 2.0s", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        out_dir = os.path.join("evaluation", "reports", "waveforms", "true_raw")
        os.makedirs(out_dir, exist_ok=True)
        
        out_f = os.path.join(out_dir, filename)
        plt.savefig(out_f, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualisasi true raw sensor untuk {title} berhasil disimpan ke: {out_f}")

if __name__ == "__main__":
    main()
