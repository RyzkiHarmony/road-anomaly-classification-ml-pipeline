# export.py
# Export file labeling terstratifikasi (high/candidate/normal).

import os
import pandas as pd

from config import OUT_FOLDER
from label_suggester import SUGGESTION_COLS

# Column schema for labeling output files (always written, even when empty).
# Superset of old schema — new columns are appended so that downstream scripts
# (manual_labeling_per_trip.py, build_train_set.py) keep working.
LABELING_COLS = [
    "event_id", "time_s", "lat", "lon",
    "peak_vertical", "peak_vertical_g", "peak_gyro_mag",
    "speed_mean", "event_duration", "vert_jrk",
    "num_peaks_accel", "num_peaks_gyro", "peak_interval_mean", "peak_interval_std",
    "asymmetry_score", "vertical_energy", "gyro_energy", "accel_to_gyro_ratio", "local_duration",
    "top2_peak_ratio", "duration_above_threshold", "max_jerk", "peak_to_peak",
    # --- Fitur Domain Frekuensi & Distribusi ---
    "fft_high_low_ratio", "zcr", "kurtosis", "skewness",
    # --- Per-Axis Gyro ---
    "gyro_pitch_energy", "gyro_roll_energy", "gyro_yaw_energy", "gyro_pitch_roll_ratio",
    "score", "priority", "level",
    # --- Suggestion ---
    *SUGGESTION_COLS,
    # --- Labeling fields ---
    "label", "notes", "maps_link",
]


def prepare_labeling_file(df, filename):
    """
    Write a labeling CSV.  If df is empty the file is still created with the
    correct column schema so that downstream tooling does not break.
    """
    out_path = os.path.join(OUT_FOLDER, filename)

    if df.empty:
        pd.DataFrame(columns=LABELING_COLS).to_csv(out_path, index=False)
        return

    out                 = df.copy()
    out["label"]        = ""
    out["notes"]        = ""
    out["maps_link"]    = out.apply(
        lambda r: f"https://www.google.com/maps?q={r['lat']},{r['lon']}", axis=1
    )
    # Retain only defined schema columns that exist
    cols = [c for c in LABELING_COLS if c in out.columns]
    # Sort by composite score (highest first) so labeler works top-down
    sort_col = "score" if "score" in out.columns else "peak_mag"
    out[cols].sort_values(sort_col, ascending=False).to_csv(out_path, index=False)
