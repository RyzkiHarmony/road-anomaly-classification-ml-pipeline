# feature_extraction.py
# Ekstraksi fitur event-level (shape, frekuensi, distribusi) dan
# window-level (sliding window) dari sinyal sensor.
#
# DESIGN NOTES:
# - extract_event_shape_features() mengekstrak fitur sekitar pusat event
#   untuk membedakan Pothole vs Speed Bump vs Non-Event.
# - extract_windows_features() mengekstrak fitur per sliding window
#   untuk training set ML.

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation, kurtosis as sp_kurtosis, skew as sp_skew
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d

from config import WINDOW_S, OVERLAP


def extract_event_shape_features(window_df, bg_df=None):
    """
    Ekstrak 10-15 fitur matematis dari sebuah window waktu (misalnya 2 detik).
    Fungsi ini bersifat murni Window-Level (tidak melakukan pemotongan time-series).
    """
    EMPTY = {
        "num_peaks_accel": 0,
        "num_peaks_gyro": 0,
        "peak_interval_mean": 0.0,
        "peak_interval_std": 0.0,
        "asymmetry_score": 0.0,
        "vertical_energy": 0.0,
        "gyro_energy": 0.0,
        "accel_to_gyro_ratio": 0.0,
        "local_duration": 0.0,
        "top2_peak_ratio": 0.0,
        "duration_above_threshold": 0.0,
        "max_jerk": 0.0,
        "peak_to_peak": 0.0,
        "fft_high_low_ratio": 0.0,
        "zcr": 0.0,
        "kurtosis": 0.0,
        "skewness": 0.0,
        "gyro_pitch_energy": 0.0,
        "gyro_roll_energy": 0.0,
        "gyro_yaw_energy": 0.0,
        "gyro_pitch_roll_ratio": 0.0,
        "energy_psd_2_10": 0.0,
        "speed_vert_interaction": 0.0,
        "speed_normalized_p2p": 0.0,
        "horizontal_to_vertical_ratio": 0.0,
        "grav_y_std": 0.0,
        "grav_z_std": 0.0,
        "linear_jerk_3d_max": 0.0,
        "snr_vertical": 0.0,
        "crest_factor": 0.0,
        "hjorth_activity": 0.0,
        "hjorth_mobility": 0.0,
        "hjorth_complexity": 0.0,
        "corr_xy": 0.0,
        "corr_xz": 0.0,
        "corr_yz": 0.0,
        "impulse_factor": 0.0,
        "clearance_factor": 0.0,
        "shape_factor": 0.0,
        "time_center_of_mass": 0.0,
        "min_z_to_max_z_ratio": 0.0,
        "first_peak_polarity": 0.0,
        "rise_time_ratio": 0.0,
        "peak_asymmetry": 0.0,
        "waveform_complexity": 0.0,
        "brake_to_bump_ratio": 0.0,
        "down_up_asymmetry": 0.0,
    }

    if window_df is None or window_df.empty:
        return EMPTY

    seg = window_df
    t = seg["timestamp"].astype(float).values / 1000.0
    mags = seg["magnitude"].astype(float).values
    a_vert = seg["a_vertical"].astype(float).values

    # ---------- Sampling rate ----------
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt) if len(dt) > 0 else 50.0

    # ---------- Gyroscope ----------
    has_gyro = all(c in seg.columns for c in ("gx", "gy", "gz"))
    if has_gyro:
        gx = seg["gx"].fillna(0.0).astype(float).values
        gy = seg["gy"].fillna(0.0).astype(float).values
        gz = seg["gz"].fillna(0.0).astype(float).values
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    else:
        gyro_mag = np.zeros_like(mags)

    # ---------- Remove gravity ----------
    mags_norm = np.abs(a_vert)

    # ---------- Smoothing (CAUSAL) ----------
    # Menggunakan Trailing Moving Average untuk menghindari non-causal lookahead dari Gaussian filter.
    mags_smooth = pd.Series(mags_norm).rolling(window=5, min_periods=1).mean().values

    # ---------- Causal Adaptive Thresholding (EMA) ----------
    # Menggunakan Causal EMA agar selaras dengan peak_detection dan ramah Kotlin (O(1)).
    s_mags = pd.Series(mags_smooth)
    ema_baseline = s_mags.ewm(alpha=0.05, adjust=False).mean()
    ema_dev = (s_mags - ema_baseline).abs().ewm(alpha=0.05, adjust=False).mean()
    # Threshold dinamis per-sampel
    accel_thr = (ema_baseline + 3.0 * ema_dev).values

    min_dist = max(1, int(0.15 * fs))

    accel_peaks, props = find_peaks(
        mags_smooth,
        height=accel_thr,
        prominence=0.5,
        distance=min_dist,
    )

    num_peaks_accel = len(accel_peaks)

    # ---------- Dominant peak as event center ----------
    if num_peaks_accel > 0:
        peak_heights = props["peak_heights"]
        dominant_idx = accel_peaks[np.argmax(peak_heights)]
        t_center = t[dominant_idx]
    else:
        t_center = t[len(t)//2] if len(t) > 0 else 0.0

    t_rel = t - t_center

    # ---------- Gyro peaks ----------
    if has_gyro:
        s_gyro = pd.Series(gyro_mag)
        ema_g_base = s_gyro.ewm(alpha=0.05, adjust=False).mean()
        ema_g_dev = (s_gyro - ema_g_base).abs().ewm(alpha=0.05, adjust=False).mean()
        gyro_thr = (ema_g_base + 2.5 * ema_g_dev).values

        gyro_peaks, _ = find_peaks(
            gyro_mag,
            height=gyro_thr,
            prominence=0.5,
            distance=min_dist,
        )
        num_peaks_gyro = len(gyro_peaks)
    else:
        num_peaks_gyro = 0

    # ---------- Peak interval ----------
    if num_peaks_accel > 1:
        peak_times = t_rel[accel_peaks]
        intervals  = np.diff(peak_times)

        peak_interval_mean = float(np.mean(intervals))
        peak_interval_std  = float(np.std(intervals))
        local_duration     = float(peak_times[-1] - peak_times[0])
    else:
        peak_interval_mean = 0.0
        peak_interval_std  = 0.0
        local_duration     = 0.1 if num_peaks_accel == 1 else 0.0

    # ---------- Top-2 peak ratio ----------
    if num_peaks_accel >= 2:
        sorted_peaks = np.sort(props["peak_heights"])[::-1]
        top2_peak_ratio = float(sorted_peaks[1] / (sorted_peaks[0] + 1e-6))
    else:
        top2_peak_ratio = 0.0

    # ---------- Duration above threshold (Continuous) ----------
    # Mencegah penggabungan multi-pothole dengan mencari segmen kontigu terpanjang 
    # di sekitar pusat event (menghindari label leakage).
    above_thr = mags_smooth > accel_thr
    duration_above_threshold = 0.0
    if np.any(above_thr):
        edges = np.diff(np.concatenate(([0], above_thr.astype(int), [0])))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]
        
        # Cari segmen yang bersinggungan/mengandung puncak utama
        center_idx = dominant_idx if num_peaks_accel > 0 else len(mags_smooth) // 2
        
        valid_durations = []
        for s, e in zip(starts, ends):
            # Jika puncak berada di dalam atau sangat dekat dengan segmen ini
            if s <= center_idx <= e or abs(s - center_idx) < min_dist or abs(e - center_idx) < min_dist:
                valid_durations.append(e - s)
        
        if valid_durations:
            duration_above_threshold = float(max(valid_durations)) / fs
        else:
            duration_above_threshold = float(np.max(ends - starts)) / fs

    # ---------- Asymmetry (trailing narrow window) ----------
    # Window sempit dibagi dua pada sisi trailing untuk menangkap profil impact.
    # Pothole → impact tajam di ujung window.
    narrow_mask_left  = (t_rel >= -0.15) & (t_rel < 0.0)
    narrow_mask_right = (t_rel >= 0.0) & (t_rel <= 0.15)
    left_energy  = float(np.sum(a_vert[narrow_mask_left] ** 2))
    right_energy = float(np.sum(a_vert[narrow_mask_right] ** 2))
    total_energy = left_energy + right_energy + 1e-6

    asymmetry_score = abs(left_energy - right_energy) / total_energy

    # ---------- Energy ----------
    vertical_energy = float(np.sum(a_vert ** 2))
    gyro_energy  = float(np.sum(gyro_mag ** 2))

    accel_to_gyro_ratio = vertical_energy / (gyro_energy + 1e-6)

    # ---------- Max Jerk: max |d(a_vert)/dt| ----------
    # Pothole → jerk sangat tinggi (perubahan mendadak)
    # SpeedBump → jerk lebih rendah (transisi lebih mulus)
    max_jerk = 0.0
    if len(a_vert) > 1 and fs > 0:
        jerk_signal = np.abs(np.diff(a_vert)) * fs  # convert to m/s³
        max_jerk = float(np.max(jerk_signal))

    # ---------- Peak-to-Peak Amplitude ----------
    # Mengukur total swing vertikal dalam window
    # SpeedBump → swing simetris (naik-turun)
    # Pothole → swing asimetris (dominan ke bawah)
    peak_to_peak = float(np.max(a_vert) - np.min(a_vert))

    # ---------- Domain Frekuensi: FFT High/Low Ratio ----------
    # Pothole  → energi dominan di frekuensi tinggi (>15 Hz)
    # SpeedBump → energi dominan di frekuensi rendah (<5 Hz)
    fft_high_low_ratio = 0.0
    if fs > 0 and len(a_vert) >= 8:
        try:
            fft_vals  = np.abs(rfft(a_vert)) ** 2
            fft_freqs = rfftfreq(len(a_vert), d=1.0 / fs)
            energy_low  = float(np.sum(fft_vals[fft_freqs < 5.0]))
            energy_high = float(np.sum(fft_vals[fft_freqs > 15.0]))
            fft_high_low_ratio = energy_high / (energy_low + 1e-6)
        except Exception:
            fft_high_low_ratio = 0.0

    # ---------- Zero Crossing Rate (ZCR) ----------
    # Pothole → ZCR tinggi (sinyal kacau/chaotic)
    # SpeedBump → ZCR rendah (sinyal lebih mulus)
    zcr = 0.0
    if len(a_vert) > 1:
        zcr = float(np.sum(np.diff(np.sign(a_vert)) != 0)) / len(a_vert)

    # ---------- Kurtosis & Skewness ----------
    # Kurtosis tinggi → impulsive (pothole), rendah → smooth (speed bump)
    # Skewness dihitung pada narrow trailing window agar engine noise tidak mendilusi.
    kurtosis_val = float(sp_kurtosis(a_vert, fisher=True)) if len(a_vert) >= 4 else 0.0

    narrow_mask = (t_rel >= -0.15) & (t_rel <= 0.15)
    a_vert_narrow = a_vert[narrow_mask]
    skewness_val = float(sp_skew(a_vert_narrow)) if len(a_vert_narrow) >= 4 else 0.0

    # ---------- Per-Axis Gyro Energy (Pitch / Roll / Yaw) ----------
    # SpeedBump → pitch dominant (motor menunduk-mendongak)
    # Pothole   → roll bisa lebih tinggi (motor oleng ke samping)
    gyro_pitch_energy = 0.0
    gyro_roll_energy  = 0.0
    gyro_yaw_energy   = 0.0
    gyro_pitch_roll_ratio = 0.0
    if has_gyro:
        # Asumsi mounting standar motor:
        #   Y = pitch (depan-belakang miring)
        #   X = roll  (kiri-kanan oleng)
        #   Z = yaw   (belok)
        gy_arr = seg["gy"].fillna(0.0).astype(float).values
        gx_arr = seg["gx"].fillna(0.0).astype(float).values
        gz_arr = seg["gz"].fillna(0.0).astype(float).values
        gyro_pitch_energy = float(np.sum(gy_arr ** 2))
        gyro_roll_energy  = float(np.sum(gx_arr ** 2))
        gyro_yaw_energy   = float(np.sum(gz_arr ** 2))
        gyro_pitch_roll_ratio = gyro_pitch_energy / (gyro_roll_energy + 1e-6)

    # ---------- Fitur Rekomendasi Senior ML: Interaction & PSD ----------
    # 1. Speed Interaction & Normalization
    speed_mean = seg["speed"].mean() if "speed" in seg.columns else 0.0
    speed_vert_interaction = vertical_energy * speed_mean
    speed_normalized_p2p = peak_to_peak / (speed_mean + 1.0) # Avoid div by zero

    # 2. Power Spectral Density (PSD) di rentang resonansi suspensi (2-10 Hz)
    energy_psd_2_10 = 0.0
    if fs > 0 and len(a_vert) >= 16:
        try:
            fft_vals  = np.abs(rfft(a_vert)) ** 2
            fft_freqs = rfftfreq(len(a_vert), d=1.0 / fs)
            # Fokus pada 2-10 Hz di mana suspensi motor biasanya beresonansi
            energy_psd_2_10 = float(np.sum(fft_vals[(fft_freqs >= 2.0) & (fft_freqs <= 10.0)]))
        except Exception:
            pass

    # ---------- Fitur Eksklusif Sensor Native (Senior ML Recommendations) ----------
    # 1. Rasio Energi Horizontal terhadap Vertikal
    a_horiz = seg["a_horizontal"].astype(float).values
    horiz_energy = float(np.sum(a_horiz ** 2))
    horizontal_to_vertical_ratio = horiz_energy / (vertical_energy + 1e-6)

    # 2. Standar Deviasi Gravitasi (Pitch/Rotasi Stabilitas Rangka)
    has_native_grav = all(c in seg.columns for c in ("grav_x", "grav_y", "grav_z"))
    grav_y_std = 0.0
    grav_z_std = 0.0
    if has_native_grav:
        grav_y_std = float(seg["grav_y"].std())
        grav_z_std = float(seg["grav_z"].std())

    # 3. Magnitudo Jerk Linier 3D (Dynamic 3D Jerk)
    linear_jerk_3d_max = 0.0
    has_native_lin = all(c in seg.columns for c in ("lin_ax", "lin_ay", "lin_az"))
    if has_native_lin and len(seg) > 1 and fs > 0:
        d_lax = np.diff(seg["lin_ax"].values) * fs
        d_lay = np.diff(seg["lin_ay"].values) * fs
        d_laz = np.diff(seg["lin_az"].values) * fs
        jerk_3d_mags = np.sqrt(d_lax**2 + d_lay**2 + d_laz**2)
        linear_jerk_3d_max = float(np.max(jerk_3d_mags))

    # ---------- Contextual Features (SNR & Crest Factor) ----------
    # 1. Signal-to-Noise Ratio (SNR)
    # Background Context (-5s to -1s) passed directly if available
    bg_seg = bg_df if bg_df is not None else pd.DataFrame()
    
    if len(bg_seg) > 10 and len(seg) > 5:
        bg_a_vert = bg_seg["a_vertical"].astype(float).values
        bg_energy_rate = float(np.sum(bg_a_vert ** 2)) / len(bg_a_vert)
        event_energy_rate = float(np.sum(a_vert ** 2)) / len(a_vert)
        snr_vertical = event_energy_rate / (bg_energy_rate + 1e-6)
    else:
        snr_vertical = 1.0 # Default fallback if no background found (e.g., at edges)

    # 2. Crest Factor (Peak-to-RMS Ratio) & Other Shape Metrics
    rms_val = float(np.sqrt(np.mean(a_vert**2))) if len(a_vert) > 0 else 0.0
    mean_abs_val = float(np.mean(np.abs(a_vert))) if len(a_vert) > 0 else 0.0
    mean_sqrt_abs_val = float(np.mean(np.sqrt(np.abs(a_vert)))) if len(a_vert) > 0 else 0.0
    max_abs_val = float(np.max(np.abs(a_vert))) if len(a_vert) > 0 else 0.0
    
    crest_factor = max_abs_val / (rms_val + 1e-6)
    impulse_factor = max_abs_val / (mean_abs_val + 1e-6)
    clearance_factor = max_abs_val / ((mean_sqrt_abs_val**2) + 1e-6)
    shape_factor = rms_val / (mean_abs_val + 1e-6)
    
    # Time Center of Mass (Energy concentration relative to the dominant peak)
    time_center_of_mass = 0.0
    if len(a_vert) > 0:
        sum_abs = np.sum(np.abs(a_vert))
        if sum_abs > 0:
            time_center_of_mass = float(np.sum(t_rel * np.abs(a_vert)) / sum_abs)

    # ---------- Z-Accel Polarity & Ratio Features (Direction-Aware) ----------
    min_z_to_max_z_ratio = 0.0
    first_peak_polarity = 0.0
    if len(a_vert) > 0:
        min_z = float(np.min(a_vert))
        max_z = float(np.max(a_vert))
        if max_z > 0:
            min_z_to_max_z_ratio = min_z / (max_z + 1e-6)
            
        idx_min = np.argmin(a_vert)
        idx_max = np.argmax(a_vert)
        # Pothole: drop (min) happens before bounce (max)
        first_peak_polarity = 1.0 if idx_min < idx_max else -1.0

    # ---------- 3. Hjorth Parameters (Activity, Mobility, Complexity) ----------
    hjorth_activity = 0.0
    hjorth_mobility = 0.0
    hjorth_complexity = 0.0
    
    if len(a_vert) > 2:
        hjorth_activity = float(np.var(a_vert))
        
        diff1 = np.diff(a_vert)
        diff2 = np.diff(diff1)
        
        var_y = np.var(a_vert)
        var_d1 = np.var(diff1)
        var_d2 = np.var(diff2)
        
        if var_y > 0:
            hjorth_mobility = float(np.sqrt(var_d1 / var_y))
            if var_d1 > 0:
                hjorth_complexity = float(np.sqrt(var_d2 / var_d1) / hjorth_mobility)
                
    # ---------- 4. Cross-Correlation (X, Y, Z) ----------
    corr_xy = 0.0
    corr_xz = 0.0
    corr_yz = 0.0
    
    if len(seg) > 5 and has_native_lin:
        lax = seg["lin_ax"].astype(float).values
        lay = seg["lin_ay"].astype(float).values
        laz = seg["lin_az"].astype(float).values
        
        try:
            corr_xy = float(np.corrcoef(lax, lay)[0, 1])
            corr_xz = float(np.corrcoef(lax, laz)[0, 1])
            corr_yz = float(np.corrcoef(lay, laz)[0, 1])
            
            # Handle NaNs from constant signals
            if np.isnan(corr_xy): corr_xy = 0.0
            if np.isnan(corr_xz): corr_xz = 0.0
            if np.isnan(corr_yz): corr_yz = 0.0
        except Exception:
            pass

    # ---------- 5. Shape-Aware Temporal Features ----------
    # These features capture the MORPHOLOGY (shape) of the waveform,
    # not just statistical aggregates. Critical for Pothole vs Speed Bump.

    # 5a. Rise Time Ratio: time_to_peak / time_from_peak
    # Speed Bump: gradual rise → ratio ~1.0 (symmetric hill)
    # Pothole: instant drop then slow recovery → ratio << 1.0 or >> 1.0
    rise_time_ratio = 0.0
    if len(a_vert) > 5:
        abs_peak_idx = np.argmax(np.abs(a_vert))
        time_to_peak = abs_peak_idx  # samples from start to abs peak
        time_from_peak = len(a_vert) - 1 - abs_peak_idx  # samples from abs peak to end
        rise_time_ratio = float(time_to_peak) / (float(time_from_peak) + 1e-6)

    # 5b. Peak Asymmetry: energy ratio before vs after the absolute peak
    # Speed Bump: energy is roughly equal on both sides (~0.5)
    # Pothole: energy concentrated on one side (drop or bounce)
    peak_asymmetry = 0.0
    if len(a_vert) > 5:
        abs_peak_idx = np.argmax(np.abs(a_vert))
        energy_before = float(np.sum(a_vert[:abs_peak_idx] ** 2))
        energy_after = float(np.sum(a_vert[abs_peak_idx + 1:] ** 2))
        total = energy_before + energy_after + 1e-6
        peak_asymmetry = (energy_after - energy_before) / total  # [-1, +1]

    # 5c. Waveform Complexity: arc_length / straight_line_distance
    # Smooth hill (Speed Bump): complexity ~1.0
    # Chaotic spikes (Pothole): complexity >> 1.0
    # O(N), trivial to implement in Kotlin: sum of abs(diff)
    waveform_complexity = 0.0
    if len(a_vert) > 2:
        arc_length = float(np.sum(np.abs(np.diff(a_vert))))
        straight_dist = float(np.abs(a_vert[-1] - a_vert[0]))
        waveform_complexity = arc_length / (straight_dist + 1e-6)

    # 6. EXCLUSIVE FIXED-MOUNT DIRECTIONAL FEATURES (Hard Negative Killers)
    brake_to_bump_ratio = 0.0
    if has_native_lin and vertical_energy > 0:
        ay_arr = seg["lin_ay"].astype(float).values
        brake_energy = float(np.sum(ay_arr ** 2))
        brake_to_bump_ratio = brake_energy / (vertical_energy + 1e-6)
        
    down_up_asymmetry = 0.0
    if len(a_vert) > 0:
        down_energy = float(np.sum(a_vert[a_vert < 0] ** 2))
        up_energy = float(np.sum(a_vert[a_vert > 0] ** 2))
        down_up_asymmetry = down_energy / (up_energy + 1e-6)

    return {
        "num_peaks_accel": num_peaks_accel,
        "num_peaks_gyro": num_peaks_gyro,
        "peak_interval_mean": peak_interval_mean,
        "peak_interval_std": peak_interval_std,
        "asymmetry_score": asymmetry_score,
        "vertical_energy": vertical_energy,
        "gyro_energy": gyro_energy,
        "accel_to_gyro_ratio": accel_to_gyro_ratio,
        "local_duration": local_duration,
        "top2_peak_ratio": top2_peak_ratio,
        "duration_above_threshold": duration_above_threshold,
        "max_jerk": max_jerk,
        "peak_to_peak": peak_to_peak,
        "fft_high_low_ratio": fft_high_low_ratio,
        "zcr": zcr,
        "kurtosis": kurtosis_val,
        "skewness": skewness_val,
        "gyro_pitch_energy": gyro_pitch_energy,
        "gyro_roll_energy":  gyro_roll_energy,
        "gyro_yaw_energy":   gyro_yaw_energy,
        "gyro_pitch_roll_ratio": gyro_pitch_roll_ratio,
        "energy_psd_2_10": energy_psd_2_10,
        "speed_vert_interaction": speed_vert_interaction,
        "speed_normalized_p2p": speed_normalized_p2p,
        "horizontal_to_vertical_ratio": horizontal_to_vertical_ratio,
        "grav_y_std": grav_y_std,
        "grav_z_std": grav_z_std,
        "linear_jerk_3d_max": linear_jerk_3d_max,
        "snr_vertical": snr_vertical,
        "crest_factor": crest_factor,
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
        "corr_xy": corr_xy,
        "corr_xz": corr_xz,
        "corr_yz": corr_yz,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
        "shape_factor": shape_factor,
        "time_center_of_mass": time_center_of_mass,
        "min_z_to_max_z_ratio": min_z_to_max_z_ratio,
        "first_peak_polarity": first_peak_polarity,
        "rise_time_ratio": rise_time_ratio,
        "peak_asymmetry": peak_asymmetry,
        "waveform_complexity": waveform_complexity,
        "brake_to_bump_ratio": brake_to_bump_ratio,
        "down_up_asymmetry": down_up_asymmetry,
    }


def extract_windows_features(df):
    """
    Sliding-window feature extraction over accelerometer and gyroscope signals.

    Accelerometer : mag_mean, mag_std, mag_max, mag_rms, mag_jrk
    Gyroscope     : gyro_mag_mean, gyro_mag_max, gyro_mag_jrk,
                    gyro_x_std, gyro_y_std, gyro_z_std
    Contextual    : lat_mean, lon_mean, speed_mean
    """
    df    = df.sort_values("timestamp").reset_index(drop=True)
    times = df["timestamp"].astype(float) / 1000.0
    t0, t1 = times.iloc[0], times.iloc[-1]
    step   = WINDOW_S * (1 - OVERLAP)

    has_gyro = all(c in df.columns for c in ("gx", "gy", "gz"))

    windows = []
    start   = t0

    while start + WINDOW_S <= t1 + 1e-6:
        end = start + WINDOW_S
        seg = df[(times >= start) & (times < end)]

        if len(seg) >= 3:
            mags = seg["magnitude"].astype(float).values
            a_vert = seg["a_vertical"].astype(float).values

            feat = {
                "window_start": start,
                "window_end":   end,
                "mag_mean":     float(np.mean(mags)),
                "mag_std":      float(np.std(mags)),
                "mag_max":      float(np.max(mags)),
                "mag_rms":      float(np.sqrt(np.mean(mags ** 2))),
                "mag_jrk":      float(np.mean(np.abs(np.diff(mags)))),
                "vert_mean":    float(np.mean(a_vert)),
                "vert_std":     float(np.std(a_vert)),
                "vert_max":     float(np.max(a_vert)),
                "vert_min":     float(np.min(a_vert)),
                "vert_rms":     float(np.sqrt(np.mean(a_vert ** 2))),
                "lat_mean":     float(seg["lat"].mean()),
                "lon_mean":     float(seg["lon"].mean()),
                "speed_mean":   float(seg["speed"].mean()) if "speed" in seg.columns else float("nan"),
            }

            if has_gyro:
                gx = seg["gx"].fillna(0.0).astype(float).values
                gy = seg["gy"].fillna(0.0).astype(float).values
                gz = seg["gz"].fillna(0.0).astype(float).values
                gyro_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

                feat["gyro_mag_mean"] = float(np.mean(gyro_mag))
                feat["gyro_mag_max"]  = float(np.max(gyro_mag))
                feat["gyro_mag_jrk"]  = float(np.mean(np.abs(np.diff(gyro_mag)))) if len(gyro_mag) > 1 else 0.0
                feat["gyro_x_std"]    = float(np.std(gx))
                feat["gyro_y_std"]    = float(np.std(gy))
                feat["gyro_z_std"]    = float(np.std(gz))
            else:
                for k in ("gyro_mag_mean", "gyro_mag_max", "gyro_mag_jrk",
                          "gyro_x_std", "gyro_y_std", "gyro_z_std"):
                    feat[k] = 0.0

            windows.append(feat)

        start += step

    return pd.DataFrame(windows)
