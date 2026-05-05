import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Handle case where normal_cutoff >= 1.0
    if normal_cutoff >= 1.0:
        return data
        
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Padlen must be less than or equal to len(data) - 1
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    if padlen < 1:
        raise ValueError(f"Data length {len(data)} too short for filter padlen.")
    y = filtfilt(b, a, data, padlen=padlen)
    return y

def apply_sensor_fusion(df, cutoff_hz=0.5):
    """
    Applies sensor fusion to separate gravity from linear acceleration.
    Calculates a_vertical (projection of linear accel on gravity) and a_horizontal.
    Raises ValueError if data is insufficient or too noisy.
    """
    df = df.copy()
    
    # Strict Validation: Check required columns
    required_cols = ["ax", "ay", "az"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Strict Validation: Interpolate small NaNs, fail on too many NaNs
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > len(df) * 0.1: # If more than 10% NaNs, fail
            raise ValueError(f"Too many NaNs in {col} (>{len(df)*0.1})")
        if nan_count > 0:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()

    if len(df) < 50:
        raise ValueError("Insufficient data length for reliable sensor fusion (minimum 50 samples required).")

    times = df["timestamp"].astype(float).values / 1000.0
    dt = np.diff(times)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt) if len(dt) > 0 else 50.0

    # Low-pass filter to estimate gravity
    try:
        gx_est = butter_lowpass_filter(df["ax"].values, cutoff_hz, fs)
        gy_est = butter_lowpass_filter(df["ay"].values, cutoff_hz, fs)
        gz_est = butter_lowpass_filter(df["az"].values, cutoff_hz, fs)
    except Exception as e:
        raise ValueError(f"Sensor fusion filtering failed: {str(e)}")
        
    df["grav_x"] = gx_est
    df["grav_y"] = gy_est
    df["grav_z"] = gz_est
    
    # Linear acceleration
    df["lin_ax"] = df["ax"] - gx_est
    df["lin_ay"] = df["ay"] - gy_est
    df["lin_az"] = df["az"] - gz_est

    # Normalize gravity vector
    g_mag = np.sqrt(gx_est**2 + gy_est**2 + gz_est**2)
    # avoid division by zero
    g_mag[g_mag == 0] = 1.0

    gx_norm = gx_est / g_mag
    gy_norm = gy_est / g_mag
    gz_norm = gz_est / g_mag

    # Project linear acceleration onto gravity vector
    df["a_vertical"] = df["lin_ax"] * gx_norm + df["lin_ay"] * gy_norm + df["lin_az"] * gz_norm
    
    # Horizontal acceleration (magnitude of the remaining linear acceleration vector)
    lin_mag_sq = df["lin_ax"]**2 + df["lin_ay"]**2 + df["lin_az"]**2
    a_horiz_sq = np.maximum(0, lin_mag_sq - df["a_vertical"]**2)
    df["a_horizontal"] = np.sqrt(a_horiz_sq)
    
    # magnitude kept purely for downstream compatibility if needed, but not for logic
    if "magnitude" not in df.columns:
        df["magnitude"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

    return df
