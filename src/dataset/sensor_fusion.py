import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt
from config import TARGET_HZ, get_logger
from filters import butterworth_bandpass_iir

logger = get_logger(__name__)

def resample_100hz(df, target_hz=TARGET_HZ):
    """Resample dataframe to TARGET_HZ (e.g. 100Hz = 10ms intervals) using linear interpolation."""
    df_res = df.copy()
    if 'timestamp' not in df_res.columns or len(df_res) < 2:
        return df_res
        
    df_res['timestamp'] = df_res['timestamp'].astype(float)
    df_res = df_res.drop_duplicates(subset=['timestamp'])
    
    # Deteksi gap ekstrim (> 5 detik)
    # Peringatan Senior ML: Jika ada gap besar, kita TIDAK BOLEH menginterpolasi secara buta.
    # Interpolasi buta akan menciptakan ribuan data palsu yang akan merusak filter LPF.
    diffs = np.diff(df_res['timestamp'].values)
    if np.any(diffs > 5000):
        max_gap_ms = int(np.max(diffs))
        logger.warning(f"Gap >5s detected: max gap = {max_gap_ms}ms. "
                       f"Interpolation limited to 50ms; remaining NaN rows will be dropped.")

    df_res['datetime'] = pd.to_datetime(df_res['timestamp'], unit='ms')
    df_res = df_res.set_index('datetime')
    
    numeric_cols = df_res.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_res.select_dtypes(exclude=[np.number]).columns
    
    interval = f"{int(1000/target_hz)}ms"
    
    # Resample & Interpolate (Preserves peaks much better than .mean())
    # [CRITICAL FIX]: limit=5 means we ONLY interpolate gaps up to 50ms (5 samples).
    # If the gap is larger than 50ms, the physics of the pothole are lost anyway. 
    # Leaving it as NaN ensures it gets dropped instead of hallucinated.
    df_num = df_res[numeric_cols].resample(interval).interpolate(method='linear', limit=5)
    df_num = df_num.ffill(limit=5) # Strictly causal filling for trailing edges
    
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
    return df_out

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Handle case where normal_cutoff >= 1.0
    if normal_cutoff >= 1.0:
        return data
        
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Safely convert to numpy array to avoid pandas index issues
    data_arr = np.asarray(data)
    
    # Initialize filter state to avoid start-up transients
    # Since we can't look into the future (causal), we assume the signal
    # starts near the first value to minimize the initial step response.
    zi = lfilter_zi(b, a)
    zi = zi * data_arr[0]
    
    # Use causal lfilter exclusively. 
    # Do NOT use zero-phase filtfilt to ensure training data matches Android real-time phase delay exactly.
    y, _ = lfilter(b, a, data_arr, zi=zi)
    return y

def apply_sensor_fusion(df, cutoff_hz=2.0):
    """
    Applies sensor fusion to separate gravity from linear acceleration.
    Uses causal filtering (lfilter) exclusively to match Android's real-time hardware fusion phase delay exactly.
    Supports hybrid execution: if Android native hardware fusion columns are present,
    uses them directly. Otherwise, falls back to Butterworth software filter.
    """
    df = df.copy()
    
    # Check if native hardware fusion columns exist
    native_cols = ["lin_ax", "lin_ay", "lin_az", "grav_x", "grav_y", "grav_z"]
    has_native = all(col in df.columns for col in native_cols)
    
    if has_native:
        # Strict Validation for native columns: Interpolate small NaNs, fail on too many NaNs
        for col in native_cols:
            nan_count = df[col].isna().sum()
            if nan_count > len(df) * 0.1:
                raise ValueError(f"Too many NaNs in native column {col} (>{len(df)*0.1})")
            if nan_count > 0:
                df[col] = df[col].interpolate(method='linear', limit=5).bfill(limit=5).ffill(limit=5)
                
        # Resampling trip data to 100Hz
        df = resample_100hz(df)
        
        if len(df) < 50:
            raise ValueError("Insufficient data length for reliable sensor fusion (minimum 50 samples required).")
            
        gx_est = df["grav_x"].values
        gy_est = df["grav_y"].values
        gz_est = df["grav_z"].values
        
        # Calculate gravity magnitude for projection
        g_mag = np.sqrt(gx_est**2 + gy_est**2 + gz_est**2)
        g_mag[g_mag == 0] = 1.0
        
        # Project native linear acceleration onto native gravity vector to get vertical acceleration
        a_vert_raw = df["lin_ax"] * (gx_est / g_mag) + df["lin_ay"] * (gy_est / g_mag) + df["lin_az"] * (gz_est / g_mag)
        
        # Horizontal acceleration (magnitude of the remaining linear acceleration vector)
        lin_mag_sq = df["lin_ax"]**2 + df["lin_ay"]**2 + df["lin_az"]**2
        a_horiz_raw_sq = np.maximum(0, lin_mag_sq - a_vert_raw**2)
        a_horiz_raw = np.sqrt(a_horiz_raw_sq)
        
        fs = float(TARGET_HZ)
        
        # [CRITICAL FIX]: Do NOT apply 6.0Hz lowpass filter on linear acceleration.
        # Potholes are high-frequency impulses (10-15Hz). A 6.0Hz lowpass will destroy them.
        # Only apply the 1-20Hz Bandpass to remove engine noise (>20Hz) and gravity bias (<1Hz).
        df["a_vertical"] = butterworth_bandpass_iir(a_vert_raw.values)
        df["a_horizontal"] = butterworth_bandpass_iir(a_horiz_raw.values)
        
        if "magnitude" not in df.columns:
            df["magnitude"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
            
        df["a_linear_mag"] = np.sqrt(df["a_vertical"]**2 + df["a_horizontal"]**2)
        return df

    # --- Fallback to traditional software separation (for legacy data) ---
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
            df[col] = df[col].interpolate(method='linear', limit=5).bfill(limit=5).ffill(limit=5)

    # Resampling ke grid waktu seragam
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

    
    df["a_vertical"] = butterworth_bandpass_iir(df["a_vertical"].values)
    df["a_horizontal"] = butterworth_bandpass_iir(df["a_horizontal"].values)

    df["a_linear_mag"] = np.sqrt(df["a_vertical"]**2 + df["a_horizontal"]**2)
    return df
