import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(__file__))

from config import CNN_OUT_DIR, CSV_FOLDER, OUT_FOLDER, TARGET_HZ, WINDOW_SIZE_S, get_logger
from sensor_fusion import find_large_timestamp_gaps, resample_100hz, split_contiguous_segments

logger = get_logger(__name__)

# Output directories and shared paths
CNN_DATA_DIR = CNN_OUT_DIR
GT_PATH = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
EVENTS_PATH = os.path.join(OUT_FOLDER, "candidates_events.csv")

BACKGROUND_RATIO = 2
MAX_JITTER_SAMPLES = 15
SEQ_LEN = int(WINDOW_SIZE_S * TARGET_HZ)  # 2.0 * 100 = 200
EXTENDED_SEQ_LEN = SEQ_LEN + 2 * MAX_JITTER_SAMPLES  # 230
EVENT_WINDOW_HALF_S = EXTENDED_SEQ_LEN / (2.0 * TARGET_HZ)
GAP_GUARD_BAND_S = 0.5
EVENT_GAP_BUFFER_S = EVENT_WINDOW_HALF_S + GAP_GUARD_BAND_S

CHANNELS = [
    "speed",
    "ax", "ay", "az",
    "gx", "gy", "gz"
]


def _require_columns(df, required_cols, trip_id):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Trip {trip_id}: missing required columns: {', '.join(missing_cols)}")


def compute_engineered_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Pastikan kolom ax, ay, az tersedia (jika belum, jumlahkan lin dan grav)
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
            raise ValueError("Missing accelerometer columns: cannot derive ax/ay/az")

    for col in ["gx", "gy", "gz"]:
        if col not in df.columns:
            raise ValueError(f"Missing gyroscope column: {col}")

    return df


def _event_overlaps_buffered_gap(t_event, gap_intervals):
    event_start = t_event - EVENT_WINDOW_HALF_S
    event_end = t_event + EVENT_WINDOW_HALF_S

    for gap in gap_intervals:
        gap_start_s = gap["start_ms"] / 1000.0
        gap_end_s = gap["end_ms"] / 1000.0
        if event_start <= (gap_end_s + GAP_GUARD_BAND_S) and event_end >= (gap_start_s - GAP_GUARD_BAND_S):
            distance_to_gap = min(abs(t_event - gap_start_s), abs(t_event - gap_end_s))
            return True, distance_to_gap

    return False, None


def get_csv_path_for_trip(trip_id):
    meta_dir = os.path.join(os.path.dirname(CSV_FOLDER), "meta")
    meta_files = sorted(glob.glob(os.path.join(meta_dir, "*.json")))
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

    window_s = EXTENDED_SEQ_LEN / TARGET_HZ
    idx_start = np.searchsorted(times, t_center - (window_s / 2.0))
    idx_end = np.searchsorted(times, t_center + (window_s / 2.0))

    seg = raw_df.iloc[idx_start:idx_end]
    seq = np.zeros((EXTENDED_SEQ_LEN, len(CHANNELS)), dtype=np.float32)

    if len(seg) > 0:
        data_arr = seg[CHANNELS].interpolate(method='linear', limit=5).ffill(limit=5).bfill(limit=5).values

        coverage_ratio = len(data_arr) / EXTENDED_SEQ_LEN
        if coverage_ratio < 0.7 or np.isnan(data_arr).any():
            return None

        if len(data_arr) == EXTENDED_SEQ_LEN:
            seq = data_arr
        elif len(data_arr) > EXTENDED_SEQ_LEN:
            start_truncate = (len(data_arr) - EXTENDED_SEQ_LEN) // 2
            seq = data_arr[start_truncate:start_truncate + EXTENDED_SEQ_LEN]
        else:
            pad_left = (EXTENDED_SEQ_LEN - len(data_arr)) // 2
            seq[pad_left:pad_left + len(data_arr)] = data_arr
    else:
        return None

    return seq
