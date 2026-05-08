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
    REGION_WINDOW_S,
    REGION_MAD_MULTIPLIER,
    get_logger,
)

logger = get_logger(__name__)


def detect_peaks(df):
    """
    Identify candidate sample indices using Region-Based Abnormal Motion Detection.

    Instead of relying on a single static peak threshold, this computes a trailing 
    rolling standard deviation (energy) of the signals. Regions that exceed an 
    adaptive baseline (Median + N*MAD) are flagged as active, and the maximum 
    vertical peak within each active region is extracted.

    Returns
    -------
    peaks_df   : DataFrame with columns [timestamp, lat, lon, peak_mag,
                 peak_gyro_mag, peak_speed, time_s] at anomaly sample positions.
    accel_thr  : Adaptive energy threshold used.
    gyro_thr   : Adaptive gyroscope energy threshold used, or None.
    fs         : Estimated sampling frequency (Hz).
    """
    df    = df.sort_values("timestamp").reset_index(drop=True)
    times = df["timestamp"].astype(float) / 1000.0
    mags_raw_vert = df["a_vertical"].astype(float).values

    if len(df) < 3:
        return pd.DataFrame(), 0.0, None, 50.0

    diffs    = np.diff(times)
    med_diff = float(np.median(diffs)) if len(diffs) > 0 else 0.0
    fs       = 1.0 / med_diff if med_diff > 0 else 50.0
    min_dist = max(1, int(PEAK_MIN_DISTANCE_S * fs))
    region_window = max(1, int(REGION_WINDOW_S * fs))

    # 1. Calculate trailing rolling standard deviation for a_vertical
    s_vert = pd.Series(mags_raw_vert)
    rolling_std = s_vert.rolling(window=region_window, min_periods=1).std().fillna(0.0)

    # 2. Causal Adaptive Thresholding (EMA-based)
    # Kita gunakan EMA untuk merepresentasikan "baseline" kondisi jalan saat ini.
    # Ini murni kausal dan valid untuk live deployment.
    alpha_slow = 0.01  # Faktor smoothing untuk baseline (lambat)
    
    # Baseline: Rata-rata deviasi pada jalan "normal"
    ema_std = rolling_std.ewm(alpha=alpha_slow, adjust=False).mean()
    # Deviation: Variansi dari baseline tersebut
    ema_dev = (rolling_std - ema_std).abs().ewm(alpha=alpha_slow, adjust=False).mean()
    
    # Threshold = Baseline + N * Deviation
    # Kita gunakan multiplier yang sedikit lebih tinggi karena EMA lebih sensitif terhadap lokal noise.
    energy_thr = ema_std + (REGION_MAD_MULTIPLIER * 2.5) * ema_dev
    is_active = (rolling_std > energy_thr).values

    # 3. Gyroscope Context & Secondary Trigger (Causal)
    gyro_thr  = None
    gyro_mags = np.zeros(len(df))
    has_gyro = all(c in df.columns for c in ("gx", "gy", "gz"))
    
    if has_gyro:
        gx = df["gx"].fillna(0.0).astype(float).values
        gy = df["gy"].fillna(0.0).astype(float).values
        gz = df["gz"].fillna(0.0).astype(float).values
        gyro_mags = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        
        s_gyro = pd.Series(gyro_mags)
        rolling_gyro = s_gyro.rolling(window=region_window, min_periods=1).mean().fillna(0.0)
        
        ema_g     = rolling_gyro.ewm(alpha=alpha_slow, adjust=False).mean()
        ema_g_dev = (rolling_gyro - ema_g).abs().ewm(alpha=alpha_slow, adjust=False).mean()
        
        gyro_thr_causal = ema_g + (REGION_MAD_MULTIPLIER * 2.5) * ema_g_dev
        # Tetap gunakan min bound 1.0 agar tidak trigger di jalan yang terlalu mulus (noise lantai)
        is_active = is_active | (rolling_gyro > gyro_thr_causal).values | (rolling_gyro > 3.0)
        
        # Untuk logging, kita ambil rata-rata threshold terakhir
        gyro_thr = float(gyro_thr_causal.mean())

    # 4. Group continuous active regions and find local peak
    active_indices = np.where(is_active)[0]
    combined_idx = []
    
    if len(active_indices) > 0:
        breaks = np.where(np.diff(active_indices) > min_dist)[0]
        region_starts = np.insert(active_indices[breaks + 1], 0, active_indices[0])
        region_ends = np.append(active_indices[breaks], active_indices[-1])
        
        for start, end in zip(region_starts, region_ends):
            region_slice = slice(start, end + 1)
            # Find the point of maximum actual magnitude in the active region
            local_peak_idx = start + np.argmax(np.abs(mags_raw_vert[region_slice]))
            combined_idx.append(local_peak_idx)

    combined_idx = np.array(combined_idx, dtype=int)

    # Return mean thresholds for logging
    accel_thr_log = float(energy_thr.mean())
    gyro_thr_log  = float(gyro_thr) if gyro_thr is not None else None

    if combined_idx.size == 0:
        return pd.DataFrame(), accel_thr_log, gyro_thr_log, fs

    peaks                  = df.iloc[combined_idx].copy().reset_index(drop=True)
    peaks["peak_mag"]      = df["magnitude"].values[combined_idx]
    peaks["peak_vertical"] = mags_raw_vert[combined_idx]
    peaks["peak_gyro_mag"] = gyro_mags[combined_idx]
    peaks["time_s"]        = peaks["timestamp"].astype(float) / 1000.0

    if "speed" in df.columns:
        peaks["peak_speed"] = df["speed"].astype(float).values[combined_idx]
    else:
        peaks["peak_speed"] = float("nan")

    return peaks, accel_thr_log, gyro_thr_log, fs
