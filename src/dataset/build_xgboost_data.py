import pandas as pd
import numpy as np
import os
import glob
import json

# Removing tqdm dependency
def tqdm(iterable, **kwargs):
    return iterable

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(__file__))

from config import OUT_FOLDER, CSV_FOLDER, XGB_OUT_DIR, get_logger, WINDOW_SIZE_S
from sensor_fusion import apply_sensor_fusion
from feature_extraction import extract_event_shape_features

logger = get_logger(__name__)

GT_PATH      = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
EVENTS_PATH  = os.path.join(OUT_FOLDER, "candidates_events.csv")
OUTPUT_PATH  = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Rasio sampling background Non-Event tambahan jika data manual Non-Event sedikit.
BACKGROUND_RATIO = 2 
RANDOM_SEED = 42

import argparse

# --- HELPER: Find CSV for Trip ---
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

# Augmentasi linear telah dihapus secara permanen (menghindari bias fisika)
def main():
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        logger.error("File ground_truth_labels.csv atau candidates_events.csv tidak ditemukan.")
        return

    df_gt = pd.read_csv(GT_PATH)
    df_events = pd.read_csv(EVENTS_PATH)

    # 1. ATTACH LABELS TO CANDIDATES
    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
    
    if df_labeled.empty:
        logger.warning("Belum ada data yang dilabeli di ground_truth_labels.csv.")
        return

    # [CRITICAL] Re-extract features to include new Senior ML Engineer recommendations
    logger.info("Re-extracting features for labeled events to include new features (PSD, Interaction)...")
    re_extracted_records = []
    
    # GROUP BY TRIP TO AVOID RE-READING AND RE-PROCESSING FILES
    grouped_by_trip = df_labeled.groupby("trip_id")
    for trip_id, group in grouped_by_trip:
        csv_path = get_csv_path_for_trip(trip_id)
        if csv_path:
            try:
                raw_df = pd.read_csv(csv_path)
                raw_df = apply_sensor_fusion(raw_df)
                for _, row in group.iterrows():
                    t_event = row["time_s"]
                    try:
                        times_raw = raw_df["timestamp"].astype(float).values / 1000.0
                        
                        # [CRITICAL] Random Jittering to prevent Alignment Bias
                        # Geser window secara acak antara -0.5 hingga 0.5 detik dari pusat event
                        jitter = np.random.uniform(-0.5, 0.5)
                        t_window_center = t_event + jitter
                        
                        mask = (times_raw >= t_window_center - (WINDOW_SIZE_S / 2.0)) & (times_raw <= t_window_center + (WINDOW_SIZE_S / 2.0))
                        window_df = raw_df[mask]
                        feats = extract_event_shape_features(window_df)
                        row_dict = row.to_dict()
                        row_dict.update(feats)
                        re_extracted_records.append(row_dict)
                    except Exception as e:
                        logger.error(f"Failed re-extraction for event {row['event_id']}: {e}")
                        re_extracted_records.append(row.to_dict())
            except Exception as e:
                logger.error(f"Failed to load or fuse trip {trip_id}: {e}")
                for _, row in group.iterrows():
                    re_extracted_records.append(row.to_dict())
        else:
            for _, row in group.iterrows():
                re_extracted_records.append(row.to_dict())
                
    df_labeled = pd.DataFrame(re_extracted_records)

    logger.info(f"Basis data: {len(df_labeled)} event.")

    # 2. AUGMENTATION (Dihapus karena naif secara fisika)
    # df_augmented = pd.DataFrame()

    # 3. GENERATE ADDITIONAL BACKGROUND
    # ... (Logic background tetap sama) ...
    n_pos = len(df_labeled[df_labeled["label"].isin(["Pothole", "Speed Bump"])])
    n_neg_manual = len(df_labeled[df_labeled["label"] == "Non-Event"])
    
    labeled_trip_ids = df_labeled["trip_id"].unique()
    additional_bg_records = []

    if n_neg_manual < n_pos * BACKGROUND_RATIO:
        n_needed = (n_pos * BACKGROUND_RATIO) - n_neg_manual
        logger.info(f"Mengambil {n_needed} sampel background tambahan...")
        for trip_id in labeled_trip_ids:
            csv_candidates = glob.glob(os.path.join(CSV_FOLDER, f"*{trip_id}*.csv"))
            if not csv_candidates: continue
            try:
                raw_df = pd.read_csv(csv_candidates[0])
                raw_df = apply_sensor_fusion(raw_df)
                event_times = df_events[df_events["trip_id"] == trip_id]["time_s"].values
                duration = (raw_df["timestamp"].iloc[-1] - raw_df["timestamp"].iloc[0]) / 1000.0
                attempts = 0
                while len(additional_bg_records) < n_needed and attempts < 100:
                    attempts += 1
                    t_rand = raw_df["timestamp"].iloc[0]/1000.0 + np.random.uniform(5, duration - 5)
                    
                    # 1. Filter out samples too close to real events
                    if len(event_times) > 0:
                        dist_to_event = np.min(np.abs(event_times - t_rand))
                        if dist_to_event < 3.0: continue
                        
                    # 2. Cruise Speed Check: Must be driving (> 2.0 m/s) to avoid stationary idle engine vibrations
                    t_start = t_rand - 1.0
                    t_end = t_rand + 1.0
                    mask_speed = (raw_df["timestamp"] / 1000.0 >= t_start) & (raw_df["timestamp"] / 1000.0 <= t_end)
                    speed_seg = raw_df[mask_speed]["speed"] if "speed" in raw_df.columns else pd.Series()
                    speed_mean = speed_seg.mean() if len(speed_seg) > 0 else 0.0
                    
                    if speed_mean <= 2.0:
                        continue # Skip because vehicle is idle or slow
                        
                    times_raw = raw_df["timestamp"].astype(float).values / 1000.0
                    mask = (times_raw >= t_rand - (WINDOW_SIZE_S / 2.0)) & (times_raw <= t_rand + (WINDOW_SIZE_S / 2.0))
                    window_df = raw_df[mask]
                    feats = extract_event_shape_features(window_df)
                    if feats["vertical_energy"] > 0:
                        feats.update({
                            "event_id": -1, "time_s": t_rand, "trip_id": trip_id,
                            "label": "Non-Event", "source": "auto_background",
                            "speed_mean": speed_mean
                        })
                        additional_bg_records.append(feats)
                        if len(additional_bg_records) >= n_needed: break
            except Exception as e:
                logger.error(f"Gagal proses background untuk {trip_id}: {e}")

    # 4. COMBINE & SAVE
    df_bg = pd.DataFrame(additional_bg_records)
    
    # Gabungkan semua (Asli + Background)
    dfs_to_concat = [df_labeled]
    if not df_bg.empty:
        dfs_to_concat.append(df_bg)
        
    df_final = pd.concat(dfs_to_concat, ignore_index=True)

    # 3.5 CLEAN LABEL NOISE
    # PERHATIAN: Pembuangan Hard Negatives (Non-Event ekstrem) TELAH DIHENTIKAN.
    # Membuang hard negatives membuat model tidak bisa mengenali guncangan kuat yang bukan pothole.
    # Biarkan model belajar membedakan guncangan ekstrem palsu vs anomali asli.
    before_clean = len(df_final)

    # Simpan dataset
    df_final.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Dataset training disimpan di {OUTPUT_PATH} ({len(df_final)} baris)")
    logger.info("Distribusi Akhir:")
    logger.info(df_final["label"].value_counts().to_string())

if __name__ == "__main__":
    main()

