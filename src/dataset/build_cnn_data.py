import os
import glob
import json
import pandas as pd
import numpy as np
import sys

# Tambahkan path ke folder utils untuk import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(__file__))

from config import OUT_FOLDER, CSV_FOLDER, CNN_OUT_DIR, WINDOW_SIZE_S, TARGET_HZ, get_logger
from sensor_fusion import resample_100hz

logger = get_logger(__name__)

# Direktori Output untuk 1D-CNN
CNN_DATA_DIR = CNN_OUT_DIR
os.makedirs(CNN_DATA_DIR, exist_ok=True)

GT_PATH      = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
EVENTS_PATH  = os.path.join(OUT_FOLDER, "candidates_events.csv")

BACKGROUND_RATIO = 2 
MAX_JITTER_SAMPLES = 15
SEQ_LEN = int(WINDOW_SIZE_S * TARGET_HZ)  # 2.0 * 100 = 200
EXTENDED_SEQ_LEN = SEQ_LEN + 2 * MAX_JITTER_SAMPLES # 230
CHANNELS = [
    "speed",
    "ax", "ay", "az",
    "gx", "gy", "gz"
]

def compute_engineered_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 1. Pastikan kolom ax, ay, az tersedia (jika belum, jumlahkan lin dan grav)
    if "ax" not in df.columns:
        if "lin_ax" in df.columns and "grav_x" in df.columns:
            df["ax"] = df["lin_ax"] + df["grav_x"]
            df["ay"] = df["lin_ay"] + df["grav_y"]
            df["az"] = df["lin_az"] + df["grav_z"]
        elif "accel_x" in df.columns:
            df["ax"] = df["accel_x"]
            df["ay"] = df["accel_y"]
            df["az"] = df["accel_z"]
        else:
            df["ax"] = 0.0
            df["ay"] = 0.0
            df["az"] = 0.0
            
    # Pastikan gx, gy, gz ada
    for col in ["gx", "gy", "gz"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


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

def extract_sequence(raw_df, t_center):
    """
    Memotong array (EXTENDED_SEQ_LEN, C) yang berpusat pada t_center.
    Menggunakan interpolasi nearest jika sample tidak tepat EXTENDED_SEQ_LEN.
    """
    times = raw_df["timestamp"].astype(float).values / 1000.0
    
    # Toleransi untuk mencari nearest indices
    window_s = EXTENDED_SEQ_LEN / TARGET_HZ
    idx_start = np.searchsorted(times, t_center - (window_s / 2.0))
    idx_end = np.searchsorted(times, t_center + (window_s / 2.0))
    
    seg = raw_df.iloc[idx_start:idx_end]
    
    seq = np.zeros((EXTENDED_SEQ_LEN, len(CHANNELS)), dtype=np.float32)
    
    if len(seg) > 0:
        # [CRITICAL FIX]: limit=5 to avoid hallucinating large gaps
        data_arr = seg[CHANNELS].interpolate(method='linear', limit=5).ffill(limit=5).bfill(limit=5).values
        
        # Check coverage
        coverage_ratio = len(data_arr) / EXTENDED_SEQ_LEN
        if coverage_ratio < 0.7 or np.isnan(data_arr).any():
            return None
            
        if len(data_arr) == EXTENDED_SEQ_LEN:
            seq = data_arr
        elif len(data_arr) > EXTENDED_SEQ_LEN:
            # Truncate
            start_truncate = (len(data_arr) - EXTENDED_SEQ_LEN) // 2
            seq = data_arr[start_truncate:start_truncate + EXTENDED_SEQ_LEN]
        else:
            # Pad dengan 0.0 (sudah inisialisasi dari np.zeros), BUKAN edge values
            pad_left = (EXTENDED_SEQ_LEN - len(data_arr)) // 2
            seq[pad_left:pad_left+len(data_arr)] = data_arr
            
    else:
        return None
    
    return seq

def main():
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        logger.error("File ground_truth_labels.csv atau candidates_events.csv tidak ditemukan.")
        return

    df_gt = pd.read_csv(GT_PATH)
    df_events = pd.read_csv(EVENTS_PATH)

    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
    
    shared_bg_path = os.path.join(OUT_FOLDER, "shared_background.csv")
    if os.path.exists(shared_bg_path):
        try:
            df_bg = pd.read_csv(shared_bg_path)
            if not df_bg.empty:
                df_labeled = pd.concat([df_labeled, df_bg], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
        
    df_labeled = df_labeled.sort_values(["trip_id", "time_s"]).reset_index(drop=True)
    
    if df_labeled.empty:
        logger.warning("Belum ada data yang dilabeli.")
        return

    X_list = []
    y_list = []
    groups_list = []
    event_ids_list = []
    
    grouped_by_trip = df_labeled.groupby("trip_id")
    for trip_id, group in grouped_by_trip:
        csv_path = get_csv_path_for_trip(trip_id)
        if csv_path:
            logger.info(f"Processing trip {trip_id}...")
            try:
                raw_df = pd.read_csv(csv_path)
                if 'speed' not in raw_df.columns:
                    raw_df['speed'] = 0.0
                raw_df = compute_engineered_features(raw_df)
                
                # Resample to 100 Hz to ensure uniform sampling for CNN
                raw_df = resample_100hz(raw_df)
                    
                for _, row in group.iterrows():
                    t_event = row["time_s"]
                    # Jittering is now done dynamically during training
                    t_window_center = t_event
                    
                    seq = extract_sequence(raw_df, t_window_center)
                    
                    if seq is not None:
                        X_list.append(seq)
                        y_list.append(row["label"])
                        groups_list.append(trip_id)
                        event_ids_list.append(row["event_id"])
            except Exception as e:
                logger.error(f"Failed to process trip {trip_id}: {e}")
                


    X = np.stack(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)
    event_ids = np.array(event_ids_list)
    
    # Transpose X to (N_samples, Channels, Length) for PyTorch 1D-CNN
    X = np.transpose(X, (0, 2, 1))
    
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"), X)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy"), y)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy"), groups)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_event_ids.npy"), event_ids)
    
    logger.info(f"Dataset 1D-CNN disimpan. Shape X: {X.shape}, Shape y: {y.shape}")
    
if __name__ == "__main__":
    main()
