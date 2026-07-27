import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from config import TARGET_HZ, get_logger
from filters import butterworth_bandpass_iir

logger = get_logger(__name__)

MAX_ALLOWED_GAP_MS = 5000


def find_large_timestamp_gaps(df_res, max_gap_ms=MAX_ALLOWED_GAP_MS):
    """Return large timestamp gaps as absolute millisecond intervals."""
    if 'timestamp' not in df_res.columns:
        return []

    timestamps = df_res['timestamp'].astype(float).values
    diffs = np.diff(timestamps)
    if len(diffs) == 0:
        return []

    gap_indices = np.where(diffs > max_gap_ms)[0]
    gaps = []
    for idx in gap_indices:
        start_ms = float(timestamps[idx])
        end_ms = float(timestamps[idx + 1])
        gaps.append({
            'start_ms': start_ms,
            'end_ms': end_ms,
            'gap_ms': end_ms - start_ms,
        })
    return gaps


def split_contiguous_segments(df_res, max_gap_ms=MAX_ALLOWED_GAP_MS):
    """Split a dataframe into continuous segments separated by large gaps."""
    if 'timestamp' not in df_res.columns or len(df_res) == 0:
        return []

    df_sorted = df_res.copy()
    df_sorted['timestamp'] = df_sorted['timestamp'].astype(float)
    df_sorted = df_sorted.drop_duplicates(subset=['timestamp'])
    df_sorted = df_sorted.sort_values('timestamp').reset_index(drop=True)

    timestamps = df_sorted['timestamp'].values
    diffs = np.diff(timestamps)
    if len(diffs) == 0:
        return [df_sorted]

    gap_indices = np.where(diffs > max_gap_ms)[0]
    if len(gap_indices) == 0:
        return [df_sorted]

    segments = []
    start_idx = 0
    for gap_idx in gap_indices:
        segment = df_sorted.iloc[start_idx:gap_idx + 1].copy().reset_index(drop=True)
        if len(segment) > 0:
            segments.append(segment)
        start_idx = gap_idx + 1

    tail = df_sorted.iloc[start_idx:].copy().reset_index(drop=True)
    if len(tail) > 0:
        segments.append(tail)

    return segments


def _validate_timestamp_gaps(df_res):
    diffs = np.diff(df_res['timestamp'].values)
    if len(diffs) == 0:
        return

    max_gap_ms = float(np.max(diffs))
    if max_gap_ms > MAX_ALLOWED_GAP_MS:
        raise ValueError(
            f"Gap >{MAX_ALLOWED_GAP_MS / 1000:.0f}s detected: max gap = {int(max_gap_ms)}ms"
        )

def resample_100hz(df, target_hz=TARGET_HZ):
    """Resample dataframe to TARGET_HZ (e.g. 100Hz = 10ms intervals) using linear interpolation."""
    df_res = df.copy()
    if 'timestamp' not in df_res.columns or len(df_res) < 2:
        return df_res
        
    df_res['timestamp'] = df_res['timestamp'].astype(float)
    df_res = df_res.drop_duplicates(subset=['timestamp'])
    df_res = df_res.sort_values('timestamp').reset_index(drop=True)
    
    # Gap besar harus diputus, bukan dihaluskan lewat interpolasi.
    _validate_timestamp_gaps(df_res)

    df_res['datetime'] = pd.to_datetime(df_res['timestamp'], unit='ms')
    df_res = df_res.set_index('datetime')
    
    numeric_cols = df_res.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_res.select_dtypes(exclude=[np.number]).columns
    
    interval = f"{int(1000/target_hz)}ms"
    
    # Resample & Interpolate (Preserves peaks much better than .mean())
    # [CRITICAL FIX]: limit=5 means we ONLY interpolate gaps up to 50ms (5 samples).
    # If the gap is larger than 50ms, the physics of the pothole are lost anyway. 
    # Leaving it as NaN ensures it gets dropped instead of hallucinated.
    # MUST apply .mean() first to aggregate multiple points per bin!
    df_num = df_res[numeric_cols].resample(interval).mean().interpolate(method='linear', limit=5)
    df_num = df_num.ffill(limit=5) # Causal fill for trailing edges only
    
    if len(non_numeric_cols) > 0:
        # For non-numeric columns like string labels, use 'first' aggregation
        df_non_num = df_res[non_numeric_cols].resample(interval).first().ffill(limit=5)
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
                df[col] = df[col].interpolate(method='linear', limit=5).ffill(limit=5).bfill(limit=5)
                
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
        
        # a_vertical (linear z) is AC-coupled via Bandpass 1-20Hz to remove drift and engine noise.
        df["a_vertical"] = butterworth_bandpass_iir(a_vert_raw.values)
        
        # [CRITICAL FIX]: a_horizontal is a magnitude (strictly positive, high DC). 
        # Bandpassing it destroys the DC baseline and forces it below zero. 
        # We MUST use a Low-Pass Filter to remove engine noise without shifting the baseline.
        df["a_horizontal"] = butter_lowpass_filter(a_horiz_raw.values, cutoff=20.0, fs=float(TARGET_HZ))
        
        if "magnitude" not in df.columns:
            df["magnitude"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
            
        df["a_linear_mag"] = np.sqrt(df["a_vertical"]**2 + df["a_horizontal"]**2)
        return df

    else:
        raise ValueError("Missing native hardware fusion columns (grav_x, lin_ax, dll). Legacy software LPF separation is deprecated to prevent Ghost Acceleration.")
