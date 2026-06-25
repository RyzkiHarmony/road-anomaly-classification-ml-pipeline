import os
import glob
import pandas as pd
import numpy as np
import sys

# Tambahkan path ke folder utils untuk import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(__file__))

from config import OUT_FOLDER, CSV_FOLDER, CNN_OUT_DIR, WINDOW_SIZE_S, TARGET_HZ, get_logger
from sensor_fusion import apply_sensor_fusion

logger = get_logger(__name__)

# Direktori Output untuk 1D-CNN
CNN_DATA_DIR = CNN_OUT_DIR
os.makedirs(CNN_DATA_DIR, exist_ok=True)

GT_PATH      = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
EVENTS_PATH  = os.path.join(OUT_FOLDER, "candidates_events.csv")

BACKGROUND_RATIO = 2 
SEQ_LEN = int(WINDOW_SIZE_S * TARGET_HZ)  # 2.0 * 100 = 200
CHANNELS = [
    "a_vertical", "a_horizontal", "speed", 
    "a_vertical_crest_factor", "a_vertical_jerk",
    "gx", "gy", "gz", 
    "g_roll_accel", "g_pitch_accel",
    "a_vertical_rms", "a_vertical_zcr",
    "a_horizontal_rms", "energy_ratio_vh"
]

def compute_engineered_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    dt = df["timestamp"].diff().fillna(10.0) / 1000.0  # interval default 10ms
    dt = np.where(dt <= 0, 0.01, dt)
    
    # 1. Jerk
    jerk = df["a_vertical"].diff().fillna(0.0) / dt
    df["a_vertical_jerk"] = jerk
    
    # 2. Crest Factor
    window_sz = 10
    peak = df["a_vertical"].abs().rolling(window=window_sz, min_periods=1, center=True).max()
    rms = np.sqrt((df["a_vertical"]**2).rolling(window=window_sz, min_periods=1, center=True).mean())
    crest_factor = peak / (rms + 1e-6)
    df["a_vertical_crest_factor"] = crest_factor.fillna(1.0)
    
    # 3. Gyro derivatives (ang. acceleration)
    # Pastikan kolom gx, gy, gz ada di dataframe
    for col in ["gx", "gy", "gz"]:
        if col not in df.columns:
            df[col] = 0.0
            
    df["g_roll_accel"] = df["gx"].diff().fillna(0.0) / dt
    df["g_pitch_accel"] = df["gy"].diff().fillna(0.0) / dt
    
    # 4. Speed-Normalized Acceleration & Jerk
    # Normalisasi getaran terhadap kecepatan kendaraan untuk menghilangkan
    # ketergantungan amplitudo pada kecepatan berkendara.
    # epsilon=0.5 m/s mencegah division by zero saat kendaraan diam/sangat lambat.
    speed_safe = df["speed"].clip(lower=0).fillna(0.0) + 0.5
    df["a_vertical_speed_norm"] = df["a_vertical"] / speed_safe
    df["jerk_speed_norm"] = df["a_vertical_jerk"] / speed_safe
    
    # 5. Rolling RMS (Root Mean Square) dari akselerasi vertikal
    # Mengukur energi getaran rata-rata dalam jendela 200ms (20 sampel @100Hz).
    # Pothole: lonjakan RMS tajam & singkat. Jalan kasar: RMS menengah kontinu.
    rms_window = 20
    df["a_vertical_rms"] = np.sqrt(
        (df["a_vertical"]**2).rolling(window=rms_window, min_periods=1, center=True).mean()
    ).fillna(0.0)
    
    # 6. Zero Crossing Rate (ZCR) dari akselerasi vertikal
    # Menghitung fraksi perubahan tanda sinyal dalam jendela 200ms.
    # Speed Bump: ZCR rendah (osilasi lambat). Jalan berkerikil: ZCR tinggi.
    zcr_window = 20
    sign_changes = (np.sign(df["a_vertical"]).diff().abs() > 0).astype(float)
    df["a_vertical_zcr"] = sign_changes.rolling(
        window=zcr_window, min_periods=1, center=True
    ).mean().fillna(0.0)
    
    # 7. Rolling RMS dari akselerasi horizontal
    # Mengukur energi getaran horizontal. Pengereman mendadak memiliki
    # a_horizontal_rms tinggi tanpa a_vertical_rms tinggi (beda dari Pothole).
    df["a_horizontal_rms"] = np.sqrt(
        (df["a_horizontal"]**2).rolling(window=rms_window, min_periods=1, center=True).mean()
    ).fillna(0.0)
    
    # 8. Rasio Energi Vertikal / Horizontal
    # Speed Bump: rasio tinggi (getaran dominan vertikal).
    # Pothole: rasio menengah (campuran vertikal + horizontal).
    # Pengereman: rasio rendah (dominan horizontal).
    df["energy_ratio_vh"] = df["a_vertical_rms"] / (df["a_horizontal_rms"] + 1e-6)
    
    return df


def get_csv_path_for_trip(trip_id):
    import json
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

def extract_sequence(raw_df, t_center):
    """
    Memotong array (SEQ_LEN, 3) yang berpusat pada t_center.
    Menggunakan interpolasi nearest jika sample tidak tepat 200.
    """
    times = raw_df["timestamp"].astype(float).values / 1000.0
    
    # Toleransi untuk mencari nearest indices
    idx_start = np.searchsorted(times, t_center - (WINDOW_SIZE_S / 2.0))
    idx_end = np.searchsorted(times, t_center + (WINDOW_SIZE_S / 2.0))
    
    seg = raw_df.iloc[idx_start:idx_end]
    
    # Jika kurang dari 200, pad atau ambil yang terdekat agar pas 200.
    # Secara praktek, karena sudah di-resample 100Hz, biasanya ukurannya sekitar 200.
    
    seq = np.zeros((SEQ_LEN, len(CHANNELS)), dtype=np.float32)
    
    if len(seg) > 0:
        data_arr = seg[CHANNELS].interpolate(method='linear').ffill().bfill().fillna(0.0).values
        # Jika panjang lebih atau kurang dari SEQ_LEN, lakukan simple truncating / zero padding
        # (Lebih baik: linear interpolation untuk array 1D)
        if len(data_arr) == SEQ_LEN:
            seq = data_arr
        elif len(data_arr) > SEQ_LEN:
            # Truncate
            start_truncate = (len(data_arr) - SEQ_LEN) // 2
            seq = data_arr[start_truncate:start_truncate + SEQ_LEN]
        else:
            # Pad dengan copy elemen terakhir/pertama
            pad_left = (SEQ_LEN - len(data_arr)) // 2
            pad_right = SEQ_LEN - len(data_arr) - pad_left
            seq[pad_left:pad_left+len(data_arr)] = data_arr
            if pad_left > 0:
                seq[:pad_left] = data_arr[0]
            if pad_right > 0:
                seq[-pad_right:] = data_arr[-1]
    
    return seq

def main():
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        logger.error("File ground_truth_labels.csv atau candidates_events.csv tidak ditemukan.")
        return

    df_gt = pd.read_csv(GT_PATH)
    df_events = pd.read_csv(EVENTS_PATH)

    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
    
    if df_labeled.empty:
        logger.warning("Belum ada data yang dilabeli.")
        return

    X_list = []
    y_list = []
    groups_list = []
    
    grouped_by_trip = df_labeled.groupby("trip_id")
    for trip_id, group in grouped_by_trip:
        csv_path = get_csv_path_for_trip(trip_id)
        if csv_path:
            logger.info(f"Processing trip {trip_id}...")
            try:
                raw_df = pd.read_csv(csv_path)
                raw_df = apply_sensor_fusion(raw_df)
                if 'speed' not in raw_df.columns:
                    raw_df['speed'] = 0.0
                raw_df = compute_engineered_features(raw_df)
                    
                for _, row in group.iterrows():
                    t_event = row["time_s"]
                    # Jittering is now done dynamically during training
                    t_window_center = t_event
                    
                    seq = extract_sequence(raw_df, t_window_center)
                    
                    X_list.append(seq)
                    y_list.append(row["label"])
                    groups_list.append(trip_id)
            except Exception as e:
                logger.error(f"Failed to process trip {trip_id}: {e}")
                
    # Generate Background (Non-Event)
    n_pos = len(df_labeled[df_labeled["label"].isin(["Pothole", "Speed Bump"])])
    n_neg_manual = len(df_labeled[df_labeled["label"] == "Non-Event"])
    
    if n_neg_manual < n_pos * BACKGROUND_RATIO:
        n_needed = (n_pos * BACKGROUND_RATIO) - n_neg_manual
        logger.info(f"Mengambil {n_needed} sampel background tambahan...")
        for trip_id in df_labeled["trip_id"].unique():
            csv_candidates = glob.glob(os.path.join(CSV_FOLDER, f"*{trip_id}*.csv"))
            if not csv_candidates: continue
            try:
                raw_df = pd.read_csv(csv_candidates[0])
                raw_df = apply_sensor_fusion(raw_df)
                if 'speed' not in raw_df.columns:
                    raw_df['speed'] = 0.0
                raw_df = compute_engineered_features(raw_df)
                    
                event_times = df_events[df_events["trip_id"] == trip_id]["time_s"].values
                duration = (raw_df["timestamp"].iloc[-1] - raw_df["timestamp"].iloc[0]) / 1000.0
                attempts = 0
                added = 0
                while added < n_needed and attempts < 100:
                    attempts += 1
                    t_rand = raw_df["timestamp"].iloc[0]/1000.0 + np.random.uniform(5, duration - 5)
                    
                    if len(event_times) > 0:
                        dist_to_event = np.min(np.abs(event_times - t_rand))
                        if dist_to_event < 3.0: continue
                        
                    t_start = t_rand - 1.0
                    t_end = t_rand + 1.0
                    mask_speed = (raw_df["timestamp"] / 1000.0 >= t_start) & (raw_df["timestamp"] / 1000.0 <= t_end)
                    speed_seg = raw_df[mask_speed]["speed"] if "speed" in raw_df.columns else pd.Series()
                    speed_mean = speed_seg.mean() if len(speed_seg) > 0 else 0.0
                    
                    if speed_mean <= 2.0:
                        continue 
                        
                    seq = extract_sequence(raw_df, t_rand)
                    
                    X_list.append(seq)
                    y_list.append("Non-Event")
                    groups_list.append(trip_id)
                    added += 1
                    
                n_needed -= added
                if n_needed <= 0: break
            except Exception as e:
                pass

    X = np.stack(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)
    
    # Transpose X to (N_samples, Channels, Length) for PyTorch 1D-CNN
    # Shape saat ini: (N_samples, 200, 3)
    # PyTorch butuh (N_samples, 3, 200)
    X = np.transpose(X, (0, 2, 1))
    
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"), X)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy"), y)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy"), groups)
    
    logger.info(f"Dataset 1D-CNN disimpan. Shape X: {X.shape}, Shape y: {y.shape}")
    
if __name__ == "__main__":
    main()
