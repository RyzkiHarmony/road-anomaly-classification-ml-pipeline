import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

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

def main():
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        logger.error("File ground_truth_labels.csv atau candidates_events.csv tidak ditemukan.")
        return

    df_gt = pd.read_csv(GT_PATH)
    df_events = pd.read_csv(EVENTS_PATH)

    # 1. ATTACH LABELS TO CANDIDATES
    # Ini mencakup Pothole, Speed Bump, dan "Non-Event" yang ditolak labeler.
    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
    
    if df_labeled.empty:
        logger.warning("Belum ada data yang dilabeli di ground_truth_labels.csv.")
        return

    logger.info(f"Latih dari {len(df_labeled)} event yang sudah divalidasi manual.")
    logger.info(df_labeled["label"].value_counts().to_string())

    # 2. GENERATE ADDITIONAL BACKGROUND (Optional Balance)
    # Jika jumlah Non-Event manual masih sedikit, kita ambil sampel random dari jalan normal.
    n_pos = len(df_labeled[df_labeled["label"].isin(["Pothole", "Speed Bump"])])
    n_neg_manual = len(df_labeled[df_labeled["label"] == "Non-Event"])
    
    labeled_trip_ids = df_labeled["trip_id"].unique()
    additional_bg_records = []

    if n_neg_manual < n_pos * BACKGROUND_RATIO:
        n_needed = (n_pos * BACKGROUND_RATIO) - n_neg_manual
        logger.info(f"Mengambil {n_needed} sampel background tambahan untuk balancing...")

        # Ambil sampel dari trip yang sudah dilabeli agar distribusi sensor konsisten
        for trip_id in labeled_trip_ids:
            # Cari file CSV asli
            csv_candidates = glob.glob(os.path.join(CSV_FOLDER, f"*{trip_id}*.csv"))
            if not csv_candidates: continue
            
            try:
                raw_df = pd.read_csv(csv_candidates[0])
                raw_df = apply_sensor_fusion(raw_df)
                
                # Cari area yang jauh dari event apapun (min 5 detik)
                event_times = df_events[df_events["trip_id"] == trip_id]["time_s"].values
                duration = (raw_df["timestamp"].iloc[-1] - raw_df["timestamp"].iloc[0]) / 1000.0
                
                # Coba ambil 20 titik random per trip
                attempts = 0
                while len(additional_bg_records) < n_needed and attempts < 50:
                    attempts += 1
                    t_rand = raw_df["timestamp"].iloc[0]/1000.0 + np.random.uniform(5, duration - 5)
                    
                    # Cek jarak ke event terdekat
                    if len(event_times) > 0:
                        dist_to_event = np.min(np.abs(event_times - t_rand))
                        if dist_to_event < 3.0: continue
                    
                    # Ekstrak fitur menggunakan logic yang SAMA dengan event
                    feats = extract_event_shape_features(raw_df, t_rand)
                    if feats["vertical_energy"] > 0:
                        feats.update({
                            "event_id": -1,
                            "time_s": t_rand,
                            "trip_id": trip_id,
                            "label": "Non-Event",
                            "source": "auto_background"
                        })
                        additional_bg_records.append(feats)
                        if len(additional_bg_records) >= n_needed: break
            except Exception as e:
                logger.error(f"Gagal proses background untuk {trip_id}: {e}")

    # 3. COMBINE & SAVE
    df_bg = pd.DataFrame(additional_bg_records)
    if not df_bg.empty:
        # Sinkronkan kolom agar bisa di-concat
        common_cols = [c for c in df_labeled.columns if c in df_bg.columns]
        df_final = pd.concat([df_labeled[common_cols], df_bg[common_cols]], ignore_index=True)
    else:
        df_final = df_labeled

    # Simpan dataset
    df_final.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Dataset training disimpan di {OUTPUT_PATH} ({len(df_final)} baris)")
    logger.info("Distribusi Akhir:")
    logger.info(df_final["label"].value_counts().to_string())

if __name__ == "__main__":
    main()

