import os
import glob
import numpy as np
import pandas as pd

from config import (
    CSV_FOLDER, OUT_FOLDER,
    SPEED_LOW_MS, SPEED_HIGH_MS,
    HIGH_CONF_VERT_G, CANDIDATE_VERT_G,
    PRIORITY_HIGH_THRESHOLD, PRIORITY_MEDIUM_THRESHOLD,
    get_logger,
)
from helpers import load_trip_meta, validate_magnitude
from sensor_fusion import apply_sensor_fusion
from peak_detection import detect_peaks
from feature_extraction import extract_windows_features
from clustering import cluster_peaks
from scoring import score_events
from export import prepare_labeling_file
from label_suggester import apply_label_suggestions

logger = get_logger(__name__)


def main():
    """Entry-point utama pipeline."""

    # ---------- COUNTERS ----------
    event_id_counter = 0
    total_distance_m = 0.0
    total_duration_s = 0.0
    total_trips      = 0
    total_events     = 0
    total_windows    = 0

    all_candidates = []
    all_windows    = []

    # ---------- PROCESS EACH TRIP ----------
    for csv_path in glob.glob(os.path.join(CSV_FOLDER, "*.csv")):
        logger.info("Processing %s", os.path.basename(csv_path))

        df = pd.read_csv(csv_path)

        if "magnitude" not in df.columns:
            df["magnitude"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)

        validate_magnitude(df, os.path.basename(csv_path))

        # Apply sensor fusion to estimate a_vertical and a_horizontal
        try:
            df = apply_sensor_fusion(df)
        except ValueError as e:
            logger.warning("SKIPPED %s: %s", os.path.basename(csv_path), e)
            continue

        _, meta  = load_trip_meta(csv_path)
        trip_id  = meta.get("tripId") if meta else os.path.basename(csv_path)

        if meta:
            total_distance_m += meta.get("distance", 0)
            total_duration_s += meta.get("duration", 0)
            total_trips      += 1

        peaks, accel_thr, gyro_thr, fs = detect_peaks(df)

        sensor_str = "accel+gyro" if gyro_thr is not None else "accel"
        logger.info(
            "  sensors=%s  peaks=%d  accel_thr=%.1f m/s²%s",
            sensor_str, len(peaks), accel_thr,
            f"  gyro_thr={gyro_thr:.2f} rad/s" if gyro_thr else "",
        )

        # Pass raw_df so cluster_peaks can compute per-event jerk
        events, event_id_counter = cluster_peaks(
            peaks, raw_df=df, start_event_id=event_id_counter
        )

        if not events.empty:
            events["trip_id"] = trip_id
            all_candidates.append(events)

        total_events += len(events)

        wfeat            = extract_windows_features(df)
        wfeat["trip_id"] = trip_id
        all_windows.append(wfeat)
        total_windows   += len(wfeat)

    # ---------- POST-CLUSTERING SCORING ----------
    candidates_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    windows_df    = pd.concat(all_windows,    ignore_index=True) if all_windows    else pd.DataFrame()

    # Apply label suggestions first to get shape-based raw labels
    candidates_df = apply_label_suggestions(candidates_df)

    # Apply composite scoring across all events
    candidates_df = score_events(candidates_df)

    # ---------- SAVE ----------
    if not candidates_df.empty:
        candidates_df.to_csv(os.path.join(OUT_FOLDER, "candidates_events.csv"), index=False)

    if not windows_df.empty:
        windows_df.to_csv(os.path.join(OUT_FOLDER, "windows_features.csv"), index=False)

    # Stratified labeling files (always written)
    if not candidates_df.empty:
        high_df = candidates_df[candidates_df["priority"] == "high"]
        med_df  = candidates_df[candidates_df["priority"] == "medium"]
        low_df  = candidates_df[candidates_df["priority"] == "low"]
    else:
        high_df = med_df = low_df = pd.DataFrame()

    prepare_labeling_file(high_df, "labeling_high_conf.csv")
    prepare_labeling_file(
        med_df.sample(min(500, len(med_df)), random_state=42) if not med_df.empty else med_df,
        "labeling_candidate.csv",
    )
    prepare_labeling_file(
        low_df.sample(min(500, len(low_df)), random_state=42) if not low_df.empty else low_df,
        "labeling_normal.csv",
    )

    # ---------- SUMMARY ----------
    _print_summary(candidates_df, windows_df,
                   total_trips, total_distance_m, total_duration_s,
                   total_events, total_windows)


def _print_summary(
    candidates_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    total_trips: int,
    total_distance_m: float,
    total_duration_s: float,
    total_events: int,
    total_windows: int,
) -> None:
    """Log pipeline summary statistics at INFO level."""
    distance_km  = total_distance_m / 1000
    duration_min = total_duration_s / 60

    logger.info("")
    logger.info("=" * 55)
    logger.info(" *** VERTICAL-CENTRIC PIPELINE SUMMARY ***")
    logger.info("=" * 55)
    logger.info(" [+] Total Trips        : %d",  total_trips)
    logger.info(" [+] Total Distance     : %.2f km",  distance_km)
    logger.info(" [+] Total Duration     : %.2f min", duration_min)
    if total_duration_s > 0:
        avg_kmh = (total_distance_m / total_duration_s) * 3.6
        logger.info(" [+] Average Speed      : %.2f km/h", avg_kmh)
    logger.info("-" * 55)

    if not candidates_df.empty:
        pri_counts = candidates_df["priority"].value_counts()
        lvl_counts = candidates_df["level"].value_counts()
        logger.info(" [!] Total Events Detected : %d", total_events)
        if distance_km > 0:
            logger.info(" [!] Events per km         : %.2f", total_events / distance_km)

        logger.info("")
        logger.info(" [ Priority Breakdown ]")
        logger.info("   * High   (>= %.2f) : %d events", PRIORITY_HIGH_THRESHOLD,   int(pri_counts.get("high",   0)))
        logger.info("   * Medium (>= %.2f) : %d events", PRIORITY_MEDIUM_THRESHOLD, int(pri_counts.get("medium", 0)))
        logger.info("   * Low    (<  %.2f) : %d events", PRIORITY_MEDIUM_THRESHOLD, int(pri_counts.get("low",    0)))

        logger.info("")
        logger.info(" [ Legacy Level Breakdown ]")
        logger.info("   * High Conf (abs(vert) >= %.1f G) : %d", HIGH_CONF_VERT_G,  int(lvl_counts.get("high_conf", 0)))
        logger.info("   * Candidate (abs(vert) >= %.1f G) : %d", CANDIDATE_VERT_G,  int(lvl_counts.get("candidate", 0)))
        logger.info("   * Normal                          : %d",                     int(lvl_counts.get("normal",    0)))

        speed_valid = candidates_df["speed_mean"].dropna()
        if not speed_valid.empty:
            n_low  = int((speed_valid < SPEED_LOW_MS).sum())
            n_mid  = int(((speed_valid >= SPEED_LOW_MS) & (speed_valid < SPEED_HIGH_MS)).sum())
            n_high = int((speed_valid >= SPEED_HIGH_MS).sum())
            logger.info("")
            logger.info(" [ Speed Context ]")
            logger.info("   * < %.0f km/h      : %d events", SPEED_LOW_MS  * 3.6, n_low)
            logger.info("   * %.0f-%.0f km/h   : %d events", SPEED_LOW_MS * 3.6, SPEED_HIGH_MS * 3.6, n_mid)
            logger.info("   * >= %.0f km/h     : %d events", SPEED_HIGH_MS * 3.6, n_high)
    else:
        logger.info(" [!] Total Events       : 0 (no candidates detected)")

    logger.info("-" * 55)
    logger.info(" [+] Total Windows Processed: %d", total_windows)

    if not windows_df.empty:
        if "vert_max" in windows_df.columns:
            desc = windows_df["vert_max"].describe()
            logger.info("")
            logger.info(" [~] VERTICAL ACCELERATION (vert_max, m/s²) — mean=%.2f  std=%.2f  max=%.2f",
                        desc["mean"], desc["std"], desc["max"])

        if "gyro_mag_max" in windows_df.columns:
            valid_gyro = windows_df[windows_df["gyro_mag_max"] > 0]["gyro_mag_max"]
            if not valid_gyro.empty:
                desc_g = valid_gyro.describe()
                logger.info(" [~] GYROSCOPE (gyro_mag_max, rad/s)         — mean=%.2f  std=%.2f  max=%.2f",
                            desc_g["mean"], desc_g["std"], desc_g["max"])
                logger.info("   Windows with gyro data : %d / %d", len(valid_gyro), len(windows_df))

    logger.info("=" * 55)


if __name__ == "__main__":
    main()