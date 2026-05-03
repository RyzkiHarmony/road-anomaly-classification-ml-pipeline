import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import os
import glob
import json

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FOLDER = os.path.join(_DIR, "out")
CSV_FOLDER = os.path.join(_DIR, "data", "csv")
META_FOLDER = os.path.join(_DIR, "data", "meta")
CANDIDATE_PATH = os.path.join(OUT_FOLDER, "candidates_events.csv")

# Manual labels from user's script
USER_LABELS = {
    2: "Non-Event", 3: "Non-Event", 4: "Non-Event", 5: "Non-Event", 6: "Non-Event",
    7: "Non-Event", 8: "Non-Event", 9: "Non-Event", 10: "Non-Event", 11: "Non-Event",
    12: "Non-Event", 15: "Non-Event", 23: "Non-Event", 28: "Non-Event", 30: "Non-Event",
    34: "Non-Event", 38: "Non-Event", 40: "Non-Event", 43: "Non-Event", 44: "Non-Event",
    45: "Non-Event", 46: "Non-Event", 47: "Pothole", 48: "Non-Event", 49: "Pothole",
    50: "Pothole", 51: "Non-Event", 52: "Pothole", 53: "Non-Event", 54: "Pothole",
    55: "Pothole", 56: "Pothole", 57: "Non-Event", 58: "Non-Event", 59: "Non-Event",
    60: "Non-Event", 61: "Pothole", 62: "Pothole", 63: "Pothole", 64: "Pothole",
    65: "Non-Event", 66: "Non-Event", 67: "Speed Bump", 68: "Non-Event",
    69: "Non-Event", 70: "Non-Event", 71: "Pothole", 72: "Non-Event",
    73: "Non-Event", 75: "Pothole", 76: "Non-Event", 77: "Non-Event",
    78: "Non-Event", 79: "Non-Event", 80: "Speed Bump", 82: "Pothole", 83: "Non-Event"
}

def extract_new_features(raw_df, event_time_s, window_s=1.0):
    """Extract temporal and shape features around event_time_s"""
    times = raw_df["timestamp"].astype(float) / 1000.0
    t_start = event_time_s - window_s
    t_end = event_time_s + window_s
    
    mask = (times >= t_start) & (times <= t_end)
    seg = raw_df[mask].copy()
    
    if len(seg) < 3:
        return {
            "num_peaks_accel": 0, "num_peaks_gyro": 0,
            "peak_interval_mean": 0.0, "peak_interval_std": 0.0,
            "symmetry_score": 0.0, "accel_energy": 0.0,
            "gyro_energy": 0.0, "accel_to_gyro_ratio": 0.0
        }
    
    mags = seg["magnitude"].astype(float).values
    t_rel = times[mask].values - event_time_s
    
    has_gyro = all(c in seg.columns for c in ("gx", "gy", "gz"))
    gyro_mag = np.zeros_like(mags)
    if has_gyro:
        gx = seg["gx"].fillna(0.0).astype(float).values
        gy = seg["gy"].fillna(0.0).astype(float).values
        gz = seg["gz"].fillna(0.0).astype(float).values
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    # 1. num_peaks_accel & 2. num_peaks_gyro
    # Use higher threshold to find *dominant* peaks. 
    # Normal driving is ~9.8 m/s2. So let's look for peaks > 12.0 m/s2 and prominence > 1.5
    # distance=10 means at least 200ms apart (at 50Hz) to avoid counting the same bump twice
    fs = 50.0 # assumed, approx
    min_dist = max(1, int(0.15 * fs)) # 150ms
    
    accel_peaks, props_a = find_peaks(mags, height=12.0, prominence=1.5, distance=min_dist)
    num_peaks_accel = len(accel_peaks)
    
    if has_gyro:
        gyro_peaks, props_g = find_peaks(gyro_mag, height=4.0, prominence=1.0, distance=min_dist)
        num_peaks_gyro = len(gyro_peaks)
    else:
        num_peaks_gyro = 0
        
    # 3 & 4. peak_interval_mean and std (accel)
    if num_peaks_accel > 1:
        peak_times = t_rel[accel_peaks]
        intervals = np.diff(peak_times)
        peak_interval_mean = np.mean(intervals)
        peak_interval_std = np.std(intervals) if len(intervals) > 1 else 0.0
    else:
        peak_interval_mean = 0.0
        peak_interval_std = 0.0
        
    # 5. symmetry_score
    left_mask = t_rel < 0
    right_mask = t_rel > 0
    # Normalize by subtracting baseline (9.8 for accel) so we compare impact energy
    mags_norm = np.maximum(0, mags - 9.8)
    left_energy = np.sum(mags_norm[left_mask])
    right_energy = np.sum(mags_norm[right_mask])
    symmetry_score = abs(left_energy - right_energy)
    
    # 7 & 8 & 9. Energy features
    accel_energy = np.sum(mags_norm**2)
    gyro_energy = np.sum(gyro_mag**2)
    accel_to_gyro_ratio = accel_energy / (gyro_energy + 1e-6)
    
    # Redefine event duration locally based on dominant peaks
    if num_peaks_accel > 1:
        local_duration = t_rel[accel_peaks[-1]] - t_rel[accel_peaks[0]]
    elif num_peaks_accel == 1:
        local_duration = 0.1 # short
    else:
        local_duration = 0.0
    
    return {
        "num_peaks_accel": num_peaks_accel,
        "num_peaks_gyro": num_peaks_gyro,
        "peak_interval_mean": float(peak_interval_mean),
        "peak_interval_std": float(peak_interval_std),
        "symmetry_score": float(symmetry_score),
        "accel_energy": float(accel_energy),
        "gyro_energy": float(gyro_energy),
        "accel_to_gyro_ratio": float(accel_to_gyro_ratio),
        "local_duration": float(local_duration)
    }

def main():
    print("Loading candidate events...")
    df = pd.read_csv(CANDIDATE_PATH)
    trips = df["trip_id"].unique()
    
    # Trip index 0 is what's used in manual_labeling_per_trip.py
    selected_trip = trips[0]
    df_trip = df[df["trip_id"] == selected_trip].copy()
    df_trip = df_trip.sort_values(by="time_s")
    df_trip["nomor_event"] = range(1, len(df_trip) + 1)
    
    # Load raw data for the correct trip
    mapping = {}
    for json_path in glob.glob(os.path.join(META_FOLDER, "*.json")):
        with open(json_path) as f:
            meta = json.load(f)
        trip_id = meta.get("tripId")
        if trip_id:
            csv_name = os.path.basename(json_path).rsplit(".", 1)[0] + ".csv"
            mapping[trip_id] = os.path.join(CSV_FOLDER, csv_name)
            
    raw_csv_path = mapping.get(selected_trip)
    if not raw_csv_path:
        print(f"Cannot find raw CSV for trip {selected_trip}")
        return
        
    raw_df = pd.read_csv(raw_csv_path)
    if "magnitude" not in raw_df.columns:
        raw_df["magnitude"] = np.sqrt(raw_df["ax"]**2 + raw_df["ay"]**2 + raw_df["az"]**2)
        
    print(f"Extracting features for {len(USER_LABELS)} labeled events...")
    
    feature_rows = []
    for nomor, label in USER_LABELS.items():
        matched = df_trip[df_trip["nomor_event"] == nomor]
        if matched.empty:
            continue
            
        row = matched.iloc[0]
        feats = extract_new_features(raw_df, row["time_s"])
        feats["label"] = label
        feats["nomor_event"] = nomor
        feats["peak_mag_g"] = row["peak_mag_g"]
        feats["peak_gyro_mag"] = row["peak_gyro_mag"]
        feats["mag_jrk"] = row["mag_jrk"]
        feats["event_duration"] = row["event_duration"]
        
        feature_rows.append(feats)
        
    fdf = pd.DataFrame(feature_rows)
    fdf.to_csv(os.path.join(OUT_FOLDER, "feature_validation_results.csv"), index=False)
    print("Feature extraction done.")
    
    # Filter only classes we care about
    valid_classes = ["Non-Event", "Pothole", "Speed Bump"]
    fdf_filtered = fdf[fdf["label"].isin(valid_classes)]
    
    print("\nClass distribution:")
    print(fdf_filtered["label"].value_counts())
    
    # Analyze separability
    features_to_plot = [
        "num_peaks_accel", "num_peaks_gyro", "peak_interval_mean", 
        "symmetry_score", "local_duration", "accel_energy", 
        "gyro_energy", "accel_to_gyro_ratio", "peak_mag_g"
    ]
    
    print("\nMean feature values per class:")
    print(fdf_filtered.groupby("label")[features_to_plot].mean().T)
    
    # Generate scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=fdf_filtered, x="symmetry_score", y="num_peaks_accel", hue="label", style="label", s=100)
    plt.title("Symmetry Score vs Num Peaks Accel")
    plt.savefig(os.path.join(OUT_FOLDER, "scatter_symmetry_peaks.png"))
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=fdf_filtered, x="accel_to_gyro_ratio", y="event_duration", hue="label", style="label", s=100)
    plt.title("Accel/Gyro Ratio vs Event Duration")
    plt.savefig(os.path.join(OUT_FOLDER, "scatter_ratio_duration.png"))

if __name__ == "__main__":
    main()
