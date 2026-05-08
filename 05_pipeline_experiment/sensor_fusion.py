import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from config import TARGET_HZ

def resample_100hz(df, target_hz=TARGET_HZ):
    """Resample dataframe to TARGET_HZ (e.g. 100Hz = 10ms intervals) using linear interpolation."""
    df_res = df.copy()
    if 'timestamp' not in df_res.columns or len(df_res) < 2:
        return df_res
        
    df_res['timestamp'] = df_res['timestamp'].astype(float)
    df_res = df_res.drop_duplicates(subset=['timestamp'])
    
    # Deteksi gap ekstrim (> 5 detik) untuk mencegah ledakan memori saat resampling
    diffs = np.diff(df_res['timestamp'].values)
    if np.any(diffs > 5000):
        # Jika ada gap besar, kita tetap lakukan resampling tapi limit interpolasi
        # agar tidak 'menghubungkan' dua titik yang terlalu jauh.
        pass

    df_res['datetime'] = pd.to_datetime(df_res['timestamp'], unit='ms')
    df_res = df_res.set_index('datetime')
    
    numeric_cols = df_res.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_res.select_dtypes(exclude=[np.number]).columns
    
    interval = f"{int(1000/target_hz)}ms"
    
    # Resample & Interpolate (limit=10 means max 100ms gap filled)
    df_num = df_res[numeric_cols].resample(interval).mean()
    df_num = df_num.interpolate(method='linear', limit=10)
    
    if len(non_numeric_cols) > 0:
        df_non_num = df_res[non_numeric_cols].resample(interval).ffill(limit=10)
        df_out = pd.concat([df_num, df_non_num], axis=1)
    else:
        df_out = df_num
        
    # Kembalikan baris yang masih NaN (karena gap terlalu besar untuk di-interpolasi)
    # Kita buang agar tidak merusak FFT/Filtering downstream
    df_out = df_out.dropna(subset=['ax', 'ay', 'az'], how='any').copy()
    
    # Ekstrak timestamp secara aman tanpa terpengaruh internal resolution pandas
    df_out['timestamp'] = (df_out.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
    return df_out.reset_index(drop=True)

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Handle case where normal_cutoff >= 1.0
    if normal_cutoff >= 1.0:
        return data
        
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Initialize filter state to avoid start-up transients
    # Since we can't look into the future (causal), we assume the signal
    # starts near the first value to minimize the initial step response.
    zi = lfilter_zi(b, a)
    zi = zi * data[0]
    
    # Use causal lfilter instead of zero-phase filtfilt
    y, _ = lfilter(b, a, data, zi=zi)
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


    
    # Resampling ke grid waktu seragam
    # print("  [DEBUG] Resampling trip data to 100Hz...")
    df = resample_100hz(df)

    if len(df) < 50:
        raise ValueError("Insufficient data length for reliable sensor fusion (minimum 50 samples required).")

    times = df["timestamp"].astype(float).values / 1000.0
    dt = np.diff(times)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt) if len(dt) > 0 else float(TARGET_HZ)

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
