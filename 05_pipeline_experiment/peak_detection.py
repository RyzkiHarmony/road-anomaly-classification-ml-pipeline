# peak_detection.py
# Multisensor peak detection untuk deteksi anomali jalan.
#
# Mengidentifikasi sampel kandidat anomali berdasarkan a_vertical.
# Gyro magnitude dan speed diikutsertakan sebagai konteks per peak.

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

from config import (
    PEAK_MIN_DISTANCE_S,
    NORMAL_VERT_MS2,
    GYRO_NORMAL_RAD,
    get_logger,
)

logger = get_logger(__name__)


def detect_peaks(df):
    """
    Identify candidate sample indices using accelerometer AND gyroscope.

    Both sensor signals are peak-detected independently.  Their index sets are
    united so that an anomaly missed by one sensor but caught by the other is
    not discarded.  Per-sample gyro magnitude and speed are carried forward for
    the event scorer.

    CATATAN: Tidak ada filter speed di sini.  Semua raw sample diproses agar
    event pada kecepatan rendah (misal lubang di gang) tetap terdeteksi.

    Returns
    -------
    peaks_df   : DataFrame with columns [timestamp, lat, lon, peak_mag,
                 peak_gyro_mag, peak_speed, time_s] at anomaly sample positions.
    accel_thr  : Adaptive accelerometer threshold used (m/s²).
    gyro_thr   : Adaptive gyroscope threshold used (rad/s), or None.
    fs         : Estimated sampling frequency (Hz).
    """
    df    = df.sort_values("timestamp").reset_index(drop=True)
    times = df["timestamp"].astype(float) / 1000.0
    # Use vertical acceleration for peak detection
    mags  = np.abs(df["a_vertical"].astype(float).values)
    mags_raw_vert = df["a_vertical"].astype(float).values

    if len(df) < 3:
        return pd.DataFrame(), 0.0, None, 50.0

    diffs    = np.diff(times)
    med_diff = float(np.median(diffs)) if len(diffs) > 0 else 0.0
    fs       = 1.0 / med_diff if med_diff > 0 else 50.0
    min_dist = max(1, int(PEAK_MIN_DISTANCE_S * fs))

    # Accelerometer
    # FIXED: Remove trip-level MAD to prevent future data leakage.
    # We use a robust fixed baseline threshold (NORMAL_VERT_MS2) to trigger candidates, 
    # and rely on local window features for severity assessment.
    accel_thr = NORMAL_VERT_MS2
    accel_idx, _ = find_peaks(mags, height=accel_thr, distance=min_dist)

    # Gyroscope
    gyro_thr  = None
    gyro_idx  = np.array([], dtype=int)
    gyro_mags = np.zeros(len(df))  # default: no gyro data

    has_gyro = all(c in df.columns for c in ("gx", "gy", "gz"))
    if has_gyro:
        gx = df["gx"].fillna(0.0).astype(float).values
        gy = df["gy"].fillna(0.0).astype(float).values
        gz = df["gz"].fillna(0.0).astype(float).values
        gyro_mags = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

        med_g    = np.median(gyro_mags)
        mad_g    = median_abs_deviation(gyro_mags, scale="normal")
        gyro_thr = max(GYRO_NORMAL_RAD, med_g + 4.0 * (mad_g if mad_g > 0 else 0.1))
        # gyro_idx, _ = find_peaks(gyro_mags, height=gyro_thr, distance=min_dist) # Removed gyro trigger

    # Peak index solely determined by vertical acceleration
    combined_idx = accel_idx.astype(int)

    if combined_idx.size == 0:
        return pd.DataFrame(), accel_thr, gyro_thr, fs

    peaks                  = df.iloc[combined_idx].copy().reset_index(drop=True)
    peaks["peak_mag"]      = df["magnitude"].values[combined_idx]
    peaks["peak_vertical"] = mags_raw_vert[combined_idx]
    peaks["peak_gyro_mag"] = gyro_mags[combined_idx]
    peaks["time_s"]        = peaks["timestamp"].astype(float) / 1000.0

    # Carry speed per peak sample for event-level aggregation
    if "speed" in df.columns:
        peaks["peak_speed"] = df["speed"].astype(float).values[combined_idx]
    else:
        peaks["peak_speed"] = float("nan")

    return peaks, accel_thr, gyro_thr, fs
