import pandas as pd
import numpy as np
import os
import glob
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(__file__))

from config import OUT_FOLDER, CSV_FOLDER, get_logger
from sensor_fusion import apply_sensor_fusion

logger = get_logger(__name__)

GT_PATH      = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
EVENTS_PATH  = os.path.join(OUT_FOLDER, "candidates_events.csv")
OUTPUT_PATH  = os.path.join(OUT_FOLDER, "shared_background.csv")

BACKGROUND_RATIO = 2 
RANDOM_SEED = 42

def main():
    np.random.seed(RANDOM_SEED)
    
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        logger.error("File ground_truth_labels.csv atau candidates_events.csv tidak ditemukan.")
        return

    df_gt = pd.read_csv(GT_PATH)
    df_events = pd.read_csv(EVENTS_PATH)

    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
    
    if df_labeled.empty:
        logger.warning("Belum ada data yang dilabeli di ground_truth_labels.csv.")
        return

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
                        
                    # We just need to save the timestamps and trip_id. Features will be extracted by the dataset builders.
                    additional_bg_records.append({
                        "event_id": f"bg_{len(additional_bg_records)}",
                        "time_s": t_rand,
                        "trip_id": trip_id,
                        "label": "Non-Event",
                        "source": "auto_background",
                        "speed_mean": speed_mean
                    })
                    if len(additional_bg_records) >= n_needed: break
            except Exception as e:
                logger.error(f"Gagal proses background untuk {trip_id}: {e}")

    df_bg = pd.DataFrame(additional_bg_records)
    df_bg.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Shared background disimpan di {OUTPUT_PATH} ({len(df_bg)} baris)")

if __name__ == "__main__":
    main()
