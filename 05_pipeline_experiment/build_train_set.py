import pandas as pd
import numpy as np
import os
import glob

# Removing tqdm dependency
def tqdm(iterable, **kwargs):
    return iterable

from config import OUT_FOLDER, CSV_FOLDER, get_logger
from sensor_fusion import apply_sensor_fusion
from feature_extraction import extract_event_shape_features

logger = get_logger(__name__)

GT_PATH      = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
EVENTS_PATH  = os.path.join(OUT_FOLDER, "candidates_events.csv")
OUTPUT_PATH  = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Rasio sampling background Non-Event tambahan jika data manual Non-Event sedikit.
BACKGROUND_RATIO = 2 
RANDOM_SEED = 42

import argparse

# --- HELPER: Find CSV for Trip ---
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

def augment_anomalies(df_labeled):
    """
    Generate synthetic variations of minority class samples by scaling raw signals.
    """
    augmented_records = []
    minority_df = df_labeled[df_labeled['label'].isin(["Pothole", "Speed Bump"])]
    
    if minority_df.empty:
        return pd.DataFrame()

    logger.info(f"Augmenting {len(minority_df)} minority class samples...")
    
    grouped = minority_df.groupby("trip_id")
    for trip_id, group in grouped:
        csv_path = get_csv_path_for_trip(trip_id)
        if not csv_path: continue
        
        try:
            raw_df = pd.read_csv(csv_path)
            raw_df = apply_sensor_fusion(raw_df)
            
            # Identify columns to scale (sensors)
            cols_to_scale = [c for c in raw_df.columns if c in ['magnitude', 'a_vertical', 'gx', 'gy', 'gz']]
            
            for _, row in group.iterrows():
                t_event = row['time_s']
                label = row['label']
                
                for factor, suffix in [(1.15, "_up"), (0.85, "_down")]:
                    scaled_df = raw_df.copy()
                    scaled_df[cols_to_scale] *= factor
                    
                    # Re-extract features from scaled signal
                    feats = extract_event_shape_features(scaled_df, t_event)
                    
                    # Ensure legacy features are preserved
                    feats.update({
                        "event_id": f"{row['event_id']}{suffix}",
                        "time_s": t_event,
                        "trip_id": trip_id,
                        "label": label,
                        "source": f"augmented_{suffix[1:]}",
                        "speed_mean": row.get('speed_mean', 0),
                        "peak_mag": row.get('peak_mag', 0) * factor,
                        "peak_vertical_g": row.get('peak_vertical_g', 0) * factor
                    })
                    augmented_records.append(feats)
                    
        except Exception as e:
            logger.error(f"Failed to augment trip {trip_id}: {e}")
            
    return pd.DataFrame(augmented_records)

def main(do_augment=False):
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
    for _, row in df_labeled.iterrows():
        trip_id = row["trip_id"]
        t_event = row["time_s"]
        csv_path = get_csv_path_for_trip(trip_id)
        if csv_path:
            try:
                raw_df = pd.read_csv(csv_path)
                raw_df = apply_sensor_fusion(raw_df)
                feats = extract_event_shape_features(raw_df, t_event)
                # Update with identification and label
                row_dict = row.to_dict()
                row_dict.update(feats)
                re_extracted_records.append(row_dict)
            except Exception as e:
                logger.error(f"Failed re-extraction for event {row['event_id']}: {e}")
                re_extracted_records.append(row.to_dict())
        else:
            re_extracted_records.append(row.to_dict())
    df_labeled = pd.DataFrame(re_extracted_records)

    logger.info(f"Basis data: {len(df_labeled)} event.")

    # 2. AUGMENTATION
    df_augmented = pd.DataFrame()
    if do_augment:
        df_augmented = augment_anomalies(df_labeled)
        if not df_augmented.empty:
            logger.info(f"Berhasil membuat {len(df_augmented)} sampel augmentasi.")

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
                        
                    feats = extract_event_shape_features(raw_df, t_rand)
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
    
    # Gabungkan semua (Asli + Augmentasi + Background)
    dfs_to_concat = [df_labeled]
    if not df_augmented.empty:
        dfs_to_concat.append(df_augmented)
    if not df_bg.empty:
        dfs_to_concat.append(df_bg)
        
    df_final = pd.concat(dfs_to_concat, ignore_index=True)

    # Simpan dataset
    df_final.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Dataset training disimpan di {OUTPUT_PATH} ({len(df_final)} baris)")
    logger.info("Distribusi Akhir:")
    logger.info(df_final["label"].value_counts().to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="Lakukan augmentasi sinyal fisik")
    args = parser.parse_args()
    main(do_augment=args.augment)

