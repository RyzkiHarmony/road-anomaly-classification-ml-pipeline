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


def extract_event_shape_features(raw_df, event_time_s, window_s=1.0):
    """
    Final refined version:
    Robust, noise-resistant, and classification-oriented shape features.
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
        # --- Fitur Baru: Domain Frekuensi & Distribusi ---
        "fft_high_low_ratio": 0.0,
        "zcr": 0.0,
        "kurtosis": 0.0,
        "skewness": 0.0,
        # --- Fitur Baru: Per-Axis Gyro (Pitch/Roll/Yaw) ---
        "gyro_pitch_energy": 0.0,
        "gyro_roll_energy": 0.0,
        "gyro_yaw_energy": 0.0,
        "gyro_pitch_roll_ratio": 0.0,
    }

    if raw_df is None or raw_df.empty:
        return EMPTY

    times = raw_df["timestamp"].astype(float).values / 1000.0

    t_start = event_time_s - window_s
    t_end   = event_time_s + window_s

    mask = (times >= t_start) & (times <= t_end)
    seg  = raw_df[mask]

    if len(seg) < 5:
        return EMPTY

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

    # ---------- Smoothing ----------
    mags_smooth = gaussian_filter1d(mags_norm, sigma=2.0)

    # ---------- Adaptive threshold (ROBUST) ----------
    med = np.median(mags_smooth)
    mad = median_abs_deviation(mags_smooth, scale="normal")
    accel_thr = med + 3.0 * (mad if mad > 0 else 1.0)

    min_dist = max(1, int(0.15 * fs))

    accel_peaks, props = find_peaks(
        mags_smooth,
        height=accel_thr,
        prominence=0.8,
        distance=min_dist,
    )

    num_peaks_accel = len(accel_peaks)

    # ---------- Dominant peak as event center ----------
    if num_peaks_accel > 0:
        peak_heights = props["peak_heights"]
        dominant_idx = accel_peaks[np.argmax(peak_heights)]
        t_center = t[dominant_idx]
    else:
        t_center = event_time_s

    t_rel = t - t_center

    # ---------- Gyro peaks ----------
    if has_gyro:
        med_g = np.median(gyro_mag)
        mad_g = median_abs_deviation(gyro_mag, scale="normal")
        gyro_thr = med_g + 2.5 * (mad_g if mad_g > 0 else 0.1)

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

    # ---------- Duration above threshold ----------
    above_thr = mags_smooth > accel_thr
    if np.any(above_thr):
        duration_above_threshold = float(np.sum(above_thr) / fs)
    else:
        duration_above_threshold = 0.0

    # ---------- Asymmetry (narrow window: ±0.3s dari peak) ----------
    # Window sempit agar engine vibration tidak mendilusi sinyal event.
    # Pothole → asimetris (energi terkonsentrasi di satu sisi)
    # SpeedBump → simetris (naik-turun seimbang)
    narrow_mask_left  = (t_rel >= -0.3) & (t_rel < 0)
    narrow_mask_right = (t_rel > 0) & (t_rel <= 0.3)
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
    # Skewness dihitung pada narrow window ±0.3s agar engine noise tidak
    # mendilusi karakter event (pothole seharusnya negatif/ke bawah)
    kurtosis_val = float(sp_kurtosis(a_vert, fisher=True)) if len(a_vert) >= 4 else 0.0

    narrow_mask = (t_rel >= -0.3) & (t_rel <= 0.3)
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
