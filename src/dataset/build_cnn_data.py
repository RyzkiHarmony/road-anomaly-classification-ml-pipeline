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
from sensor_fusion import resample_100hz, find_large_timestamp_gaps, split_contiguous_segments

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
            raise ValueError("Missing accelerometer columns: cannot derive ax/ay/az")
            
    # Pastikan gx, gy, gz ada
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
    # [DETERMINISM]: sorted() WAJIB agar urutan file konsisten lintas OS (Linux vs Windows).
    # glob.glob() tidak menjamin urutan — tanpa sorted(), pipeline non-deterministik.
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
    # [DOKUMENTASI]: shared_background.csv adalah sampel kelas Non-Event yang di-generate
    # secara acak dari trip yang sama dengan data ground truth untuk menyeimbangkan kelas.
    # Karena trip_id dipertahankan aslinya, pembagian StratifiedGroupKFold berdasarkan 
    # trip_id di train.py memastikan sampel background ini TIDAK menyebabkan data leakage 
    # lintas set (train vs test).
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
    skipped_events = 0
    skipped_by_trip = {}
    skipped_by_reason = {
        "gap_buffer": 0,
        "extract_failed": 0,
        "preprocess_failed": 0,
        "missing_csv": 0,
    }
    
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
                gap_intervals = find_large_timestamp_gaps(raw_df)
                segments = split_contiguous_segments(raw_df)

                if gap_intervals:
                    logger.info(
                        f"Trip {trip_id} has {len(gap_intervals)} large gap(s); applying {EVENT_GAP_BUFFER_S:.2f}s event buffer"
                    )

                for segment_idx, segment_raw in enumerate(segments):
                    segment_start_s = float(segment_raw["timestamp"].iloc[0]) / 1000.0
                    segment_end_s = float(segment_raw["timestamp"].iloc[-1]) / 1000.0
                    segment_events = group[(group["time_s"] >= segment_start_s) & (group["time_s"] <= segment_end_s)]

                    try:
                        segment_df = resample_100hz(segment_raw)
                    except Exception as e:
                        logger.warning(f"Trip {trip_id} segment {segment_idx} failed to resample: {e}")
                        skipped_events += len(segment_events)
                        skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + len(segment_events)
                        skipped_by_reason["preprocess_failed"] += len(segment_events)
                        continue

                    if len(segment_df) < EXTENDED_SEQ_LEN:
                        continue

                    for _, row in segment_events.iterrows():
                        t_event = float(row["time_s"])

                        overlaps_gap, _distance_to_gap = _event_overlaps_buffered_gap(t_event, gap_intervals)
                        if overlaps_gap:
                            skipped_events += 1
                            skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + 1
                            skipped_by_reason["gap_buffer"] += 1
                            continue

                        seq = extract_sequence(segment_df, t_event)

                        if seq is not None:
                            X_list.append(seq)
                            y_list.append(row["label"])
                            groups_list.append(trip_id)
                            event_ids_list.append(row["event_id"])
                        else:
                            skipped_events += 1
                            skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + 1
                            skipped_by_reason["extract_failed"] += 1
            except Exception as e:
                logger.error(f"Failed to process trip {trip_id}: {e}")
                skipped_events += len(group)
                skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + len(group)
        else:
            skipped_events += len(group)
            skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + len(group)
            skipped_by_reason["missing_csv"] += len(group)
                

    if not X_list:
        raise ValueError("No CNN samples were extracted. Check raw CSV schema and preprocessing rules.")


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
    if skipped_events:
        top_skipped = sorted(skipped_by_trip.items(), key=lambda item: item[1], reverse=True)[:5]
        logger.warning(f"Skipped {skipped_events} CNN samples during extraction.")
        logger.warning("Top skipped trips: " + ", ".join([f"{trip_id}:{count}" for trip_id, count in top_skipped]))
        logger.warning(
            "Skip reasons: "
            + ", ".join([f"{reason}={count}" for reason, count in skipped_by_reason.items()])
        )
    
if __name__ == "__main__":
    main()
