# labeling.py
# Multisensor road anomaly candidate generation pipeline for motorcycle data.
#
# DESIGN NOTES (for skripsi methodology):
# - Data dikumpulkan dengan smartphone pada sepeda motor.
# - Motor memiliki engine vibration yang tinggi, terutama saat idle/kecepatan
#   rendah.  Oleh karena itu, speed TIDAK digunakan sebagai hard filter untuk
#   membuang raw sample sebelum peak detection.
# - Speed digunakan sebagai KONTEKS pada tahap event scoring setelah
#   clustering, sehingga event pada kecepatan rendah mendapat skor lebih
#   rendah tanpa otomatis dihapus.  Ini menjaga recall tetap tinggi yang
#   penting pada tahap labeling manual.
# - Candidate events diprioritaskan berdasarkan composite score agar labeler
#   bisa mengerjakan event paling meyakinkan terlebih dahulu.
#
# Pipeline stages:
#   1. Peak detection  – adaptive threshold (median + MAD) per sensor
#   2. Clustering      – temporal + spatial grouping of proximate peaks
#   3. Event scoring   – composite score using accel, gyro, jerk, speed, duration
#   4. Export          – stratified labeling files with priority ordering

import os
import glob
import json
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter1d
from label_suggester import apply_label_suggestions, SUGGESTION_COLS
from sensor_fusion import apply_sensor_fusion

# ---------- CONFIG ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER  = os.path.join(_SCRIPT_DIR, "data/csv")
META_FOLDER = os.path.join(_SCRIPT_DIR, "data/meta")
OUT_FOLDER  = os.path.join(_SCRIPT_DIR, "out")

WINDOW_S            = 1.0
OVERLAP             = 0.5
PEAK_MIN_DISTANCE_S = 0.2
CLUSTER_TIME_S      = 2.0
CLUSTER_SPATIAL_M   = 10.0

# Accelerometer severity thresholds (G-force, specifically for a_vertical)
NORMAL_VERT_G    = 3.0
CANDIDATE_VERT_G = 5.0
HIGH_CONF_VERT_G = 8.0

G_TO_MS2       = 9.80665
NORMAL_VERT_MS2     = NORMAL_VERT_G    * G_TO_MS2
CANDIDATE_VERT_MS2  = CANDIDATE_VERT_G * G_TO_MS2
HIGH_CONF_VERT_MS2  = HIGH_CONF_VERT_G * G_TO_MS2

# Gyroscope severity thresholds (rad/s)
GYRO_NORMAL_RAD    = 3.0   # minimum to trigger peak detection
GYRO_CANDIDATE_RAD = 4.0   # event escalated to candidate
GYRO_HIGH_CONF_RAD = 8.0   # event escalated to high_conf

# Speed context thresholds (m/s).
# Pada motor, kecepatan rendah BUKAN berarti event tidak valid — bisa jadi
# lubang saat belok pelan atau polisi tidur di gang.  Threshold ini hanya
# mempengaruhi skor, bukan membuang data.
SPEED_LOW_MS  = 2.0    # ≈ 7.2 km/h – skor sedikit diturunkan (mungkin idle)
SPEED_HIGH_MS = 8.0    # ≈ 28.8 km/h – skor sedikit dinaikkan (impact lebih kuat)

# Composite score weights – dipakai di score_events()
W_ACCEL    = 0.35
W_GYRO     = 0.20
W_JERK     = 0.15
W_SPEED    = 0.15
W_DURATION = 0.15

os.makedirs(OUT_FOLDER, exist_ok=True)

# ---------- COUNTERS ----------
GLOBAL_EVENT_ID  = 0
total_distance_m = 0.0
total_duration_s = 0.0
total_trips      = 0
total_events     = 0
total_windows    = 0

# ---------- HELPERS ----------

def load_trip_meta(csv_path):
    base      = os.path.basename(csv_path).rsplit(".", 1)[0]
    json_path = os.path.join(META_FOLDER, base + ".json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json_path, json.load(f)
    return None, None


def haversine(lat1, lon1, lat2, lon2):
    R    = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def validate_magnitude(df, csv_name):
    """
    Confirm that the magnitude column contains values in m/s² (not raw ADC counts).
    Warns if the data appears to be outside a physically plausible range for road
    vibration with a consumer smartphone (expected: 5 – 200 m/s²).
    """
    mags = df["magnitude"].astype(float)
    if mags.max() < 2.0:
        print(f"  [WARN] {csv_name}: mag_max={mags.max():.3f} – values look too small. "
              "Check units (expected m/s²).")
    elif mags.max() > 500.0:
        print(f"  [WARN] {csv_name}: mag_max={mags.max():.1f} – values look too large. "
              "Check units (expected m/s², not raw ADC).")


def _normalise_0_1(values):
    """Min-max normalise an array to [0, 1].  Returns zeros if range is zero."""
    mn, mx = np.min(values), np.max(values)
    rng = mx - mn
    if rng < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mn) / rng

def _robust_normalise(values, min_val, max_val):
    """Clip values to [min_val, max_val] and normalise to [0, 1]."""
    clipped = np.clip(values, min_val, max_val)
    rng = max_val - min_val
    if rng < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (clipped - min_val) / rng


# ---------- MULTISENSOR PEAK DETECTION ----------

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
    med_a     = np.median(mags)
    mad_a     = median_abs_deviation(mags, scale="normal")
    accel_thr = max(NORMAL_VERT_MS2, med_a + 4.0 * (mad_a if mad_a > 0 else 1.0))
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

# ---------- SHAPE & TEMPORAL FEATURE EXTRACTION ----------

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

    # ---------- Asymmetry ----------
    left_energy  = float(np.sum(a_vert[t_rel < 0] ** 2))
    right_energy = float(np.sum(a_vert[t_rel > 0] ** 2))
    total_energy = left_energy + right_energy + 1e-6

    asymmetry_score = abs(left_energy - right_energy) / total_energy

    # ---------- Energy ----------
    vertical_energy = float(np.sum(a_vert ** 2))
    gyro_energy  = float(np.sum(gyro_mag ** 2))

    accel_to_gyro_ratio = vertical_energy / (gyro_energy + 1e-6)

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
    }
    
# ---------- MULTISENSOR CLUSTERING ----------

def cluster_peaks(peaks, raw_df=None):
    """
    Group temporally and spatially proximate peaks into single events.

    Clustering rule  : a new sample joins the current cluster when its time
                       distance from the *last sample in the cluster* is within
                       CLUSTER_TIME_S AND its haversine distance from the last
                       sample is within CLUSTER_SPATIAL_M.

    raw_df is the original trip DataFrame (with 'magnitude' and optionally
    'speed'); it is used to compute per-event jerk around the event centre.
    """
    global GLOBAL_EVENT_ID

    if peaks.empty:
        return pd.DataFrame()

    # Pre-compute raw helpers for jerk calculation
    raw_times = raw_mags = None
    if raw_df is not None and len(raw_df) > 0:
        raw_sorted = raw_df.sort_values("timestamp").reset_index(drop=True)
        raw_times  = raw_sorted["timestamp"].astype(float).values / 1000.0
        raw_mags   = raw_sorted["magnitude"].astype(float).values

    events  = []
    current = None

    for _, row in peaks.iterrows():
        if current is None:
            current = dict(
                event_id   = GLOBAL_EVENT_ID,
                times      = [row.time_s],
                lats       = [row.lat],
                lons       = [row.lon],
                mags       = [row.peak_mag],
                mags_v     = [row.peak_vertical],
                gyro_mags  = [row.peak_gyro_mag],
                speeds     = [row.peak_speed],
            )
            GLOBAL_EVENT_ID += 1
            continue

        # Compare against the LAST sample in the cluster (sequential)
        dt = abs(row.time_s - current["times"][-1])
        if dt <= CLUSTER_TIME_S:
            d = haversine(
                current["lats"][-1], current["lons"][-1],
                row.lat, row.lon,
            )
            if d <= CLUSTER_SPATIAL_M:
                current["times"].append(row.time_s)
                current["lats"].append(row.lat)
                current["lons"].append(row.lon)
                current["mags"].append(row.peak_mag)
                current["mags_v"].append(row.peak_vertical)
                current["gyro_mags"].append(row.peak_gyro_mag)
                current["speeds"].append(row.peak_speed)
                continue

        events.append(current)
        current = dict(
            event_id   = GLOBAL_EVENT_ID,
            times      = [row.time_s],
            lats       = [row.lat],
            lons       = [row.lon],
            mags       = [row.peak_mag],
            mags_v     = [row.peak_vertical],
            gyro_mags  = [row.peak_gyro_mag],
            speeds     = [row.peak_speed],
        )
        GLOBAL_EVENT_ID += 1

    if current:
        events.append(current)

    rows = []
    for e in events:
        max_accel_ms2 = float(np.max(e["mags"]))
        max_vert_ms2  = float(e["mags_v"][np.argmax(np.abs(e["mags_v"]))])
        max_gyro_rads = float(np.max(e["gyro_mags"]))
        accel_g       = max_accel_ms2 / G_TO_MS2
        vert_g        = max_vert_ms2 / G_TO_MS2

        # Event duration (seconds between first and last peak in cluster)
        event_duration = float(np.max(e["times"]) - np.min(e["times"]))

        # Speed mean across peaks in this event (NaN-safe)
        valid_speeds = [s for s in e["speeds"] if not (s != s)]  # filter NaN
        speed_mean   = float(np.mean(valid_speeds)) if valid_speeds else float("nan")

        # Jerk: mean |diff(a_vertical)| within ±0.5 s of event centre from raw signal
        vert_jrk = 0.0
        t_centre = float(np.mean(e["times"]))
        if raw_times is not None:
            jrk_mask = (raw_times >= t_centre - 0.5) & (raw_times <= t_centre + 0.5)
            jrk_seg  = raw_df.loc[jrk_mask, "a_vertical"].astype(float).values
            if len(jrk_seg) > 1:
                vert_jrk = float(np.mean(np.abs(np.diff(jrk_seg))))

        # Secondary shape features extraction from raw signal
        shape_feats = extract_event_shape_features(raw_df, t_centre)

        # Multisensor fusion level (OR logic) — updated to use vertical
        if abs(vert_g) >= HIGH_CONF_VERT_G or max_gyro_rads >= GYRO_HIGH_CONF_RAD:
            level = "high_conf"
        elif abs(vert_g) >= CANDIDATE_VERT_G or max_gyro_rads >= GYRO_CANDIDATE_RAD:
            level = "candidate"
        else:
            level = "normal"

        rows.append({
            "event_id":       e["event_id"],
            "time_s":         t_centre,
            "lat":            float(e["lats"][-1]),
            "lon":            float(e["lons"][-1]),
            "peak_mag":       max_accel_ms2,
            "peak_vertical":  max_vert_ms2,
            "peak_mag_g":     accel_g,
            "peak_vertical_g": vert_g,
            "peak_gyro_mag":  max_gyro_rads,
            "speed_mean":     speed_mean,
            "event_duration": event_duration,
            "vert_jrk":       vert_jrk,
            "num_peaks_accel": shape_feats["num_peaks_accel"],
            "num_peaks_gyro":  shape_feats["num_peaks_gyro"],
            "peak_interval_mean": shape_feats["peak_interval_mean"],
            "peak_interval_std":  shape_feats["peak_interval_std"],
            "asymmetry_score": shape_feats["asymmetry_score"],
            "vertical_energy": shape_feats["vertical_energy"],
            "gyro_energy":    shape_feats["gyro_energy"],
            "accel_to_gyro_ratio": shape_feats["accel_to_gyro_ratio"],
            "local_duration": shape_feats["local_duration"],
            "level":          level,
        })

    return pd.DataFrame(rows)


# ---------- EVENT SCORING ----------
# Speed-aware composite scoring.  Tujuannya adalah memprioritaskan event
# untuk labeling manual.  Event pada kecepatan rendah TIDAK dibuang,
# melainkan mendapat speed_factor lebih rendah sehingga skornya turun
# tetapi event tersebut tetap ada di output.

def score_events(events_df):
    """
    Compute a composite priority score for each candidate event.

    Score components (all normalised to [0, 1] before weighting):
      - accel   : peak_mag_g         – semakin tinggi, semakin meyakinkan
      - gyro    : peak_gyro_mag      – konfirmasi dari sensor kedua
      - jerk    : mag_jrk            – perubahan mendadak = impact event
      - speed   : speed_factor       – konteks, bukan filter
      - duration: event_duration     – event lebih panjang = lebih signifikan

    Speed factor logic (motor-specific):
      - speed >= SPEED_HIGH_MS  → 1.0  (impact lebih kuat pada kecepatan tinggi)
      - SPEED_LOW_MS <= speed < SPEED_HIGH_MS → interpolasi linier [0.5, 1.0]
      - speed < SPEED_LOW_MS   → 0.3  (mungkin engine vibration, tapi tetap dipertahankan)
      - speed NaN              → 0.5  (netral, tidak menghukum)

    Priority levels berdasarkan skor:
      - high   : score >= 0.65
      - medium : score >= 0.35
      - low    : score <  0.35

    Returns the input DataFrame with added columns: speed_factor, score, priority.
    """
    if events_df.empty:
        events_df["speed_factor"] = []
        events_df["score"]        = []
        events_df["priority"]     = []
        return events_df

    df = events_df.copy()

    # --- Speed factor (motor-specific context) ---
    def _speed_factor(spd):
        if spd != spd:  # NaN
            return 0.5
        if spd >= SPEED_HIGH_MS:
            return 1.0
        if spd >= SPEED_LOW_MS:
            # Linear interpolation: [SPEED_LOW_MS, SPEED_HIGH_MS] -> [0.5, 1.0]
            return 0.5 + 0.5 * (spd - SPEED_LOW_MS) / (SPEED_HIGH_MS - SPEED_LOW_MS)
        # Below SPEED_LOW_MS: likely idle, but don't reject — just penalise
        return 0.3

    df["speed_factor"] = df["speed_mean"].apply(_speed_factor)

    # --- Normalise each component to [0, 1] using robust fixed bounds ---
    # Outliers (like 10G or 20G peaks) will no longer squash the scores of normal 4G-5G potholes
    n_accel = _robust_normalise(np.abs(df["peak_vertical_g"]).values, 3.0, 8.0)
    n_gyro  = _robust_normalise(df["peak_gyro_mag"].values, 0.0, 6.0)
    n_jerk  = _robust_normalise(df["vert_jrk"].values, 0.0, 15.0)
    n_speed = df["speed_factor"].values  # already [0, 1]
    n_dur   = _robust_normalise(df["event_duration"].values, 0.0, 2.0)

    # --- Weighted composite score ---
    df["score"] = (
        W_ACCEL    * n_accel
        + W_GYRO   * n_gyro
        + W_JERK   * n_jerk
        + W_SPEED  * n_speed
        + W_DURATION * n_dur
    )

    # --- Shape Features Boost / Penalty (DISABLED) ---
    # def _apply_boost(row):
    #     lbl = row.get("suggested_label", "")
    #     boost = 0.0
    #     if lbl in ("Pothole", "Speed Bump"):
    #         boost = 0.25
    #     elif lbl == "Non-Event":
    #         boost = -0.25
    #     return row["score"] + boost

    # if "suggested_raw_label" in df.columns:
    #     df["score"] = df.apply(_apply_boost, axis=1)
    #     df["score"] = df["score"].clip(0.0, 1.0)

    # --- Priority label ---
    def _priority(s):
        if s >= 0.60:
            return "high"
        if s >= 0.30:
            return "medium"
        return "low"

    df["priority"] = df["score"].apply(_priority)

    return df


# ---------- WINDOW FEATURE EXTRACTION ----------

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


# ---------- LABELING FILE EXPORT ----------

# Column schema for labeling output files (always written, even when empty).
# Superset of old schema — new columns are appended so that downstream scripts
# (manual_labeling_per_trip.py, build_train_set.py) keep working.
_LABELING_COLS = [
    "event_id", "time_s", "lat", "lon",
    "peak_vertical", "peak_vertical_g", "peak_gyro_mag",
    "speed_mean", "event_duration", "vert_jrk",
    "num_peaks_accel", "num_peaks_gyro", "peak_interval_mean", "peak_interval_std",
    "asymmetry_score", "vertical_energy", "gyro_energy", "accel_to_gyro_ratio", "local_duration",
    "score", "priority", "level",
    "trip_id", "label", "notes", "maps_link",
    "peak_mag", "peak_mag_g"
] + SUGGESTION_COLS

def prepare_labeling_file(df, filename):
    """
    Write a labeling CSV.  If df is empty the file is still created with the
    correct column schema so that downstream tooling does not break.
    """
    out_path = os.path.join(OUT_FOLDER, filename)

    if df.empty:
        pd.DataFrame(columns=_LABELING_COLS).to_csv(out_path, index=False)
        return

    out                 = df.copy()
    out["label"]        = ""
    out["notes"]        = ""
    out["maps_link"]    = out.apply(
        lambda r: f"https://www.google.com/maps?q={r['lat']},{r['lon']}", axis=1
    )
    # Retain only defined schema columns that exist
    cols = [c for c in _LABELING_COLS if c in out.columns]
    # Sort by composite score (highest first) so labeler works top-down
    sort_col = "score" if "score" in out.columns else "peak_mag"
    out[cols].sort_values(sort_col, ascending=False).to_csv(out_path, index=False)


# ---------- MAIN ----------
all_candidates = []
all_windows    = []

for csv_path in glob.glob(os.path.join(CSV_FOLDER, "*.csv")):
    print(f"Processing {os.path.basename(csv_path)}")

    df = pd.read_csv(csv_path)

    if "magnitude" not in df.columns:
        df["magnitude"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)

    validate_magnitude(df, os.path.basename(csv_path))
    
    # Apply sensor fusion to estimate a_vertical and a_horizontal
    try:
        df = apply_sensor_fusion(df)
    except ValueError as e:
        print(f"  [SKIPPED] {os.path.basename(csv_path)}: {e}")
        continue

    _, meta  = load_trip_meta(csv_path)
    trip_id  = meta.get("tripId") if meta else os.path.basename(csv_path)

    if meta:
        total_distance_m += meta.get("distance", 0)
        total_duration_s += meta.get("duration", 0)
        total_trips      += 1

    peaks, accel_thr, gyro_thr, fs = detect_peaks(df)

    sensor_str = "accel+gyro" if gyro_thr is not None else "accel"
    print(
        f"  sensors={sensor_str}  peaks={len(peaks)}"
        f"  accel_thr={accel_thr:.1f} m/s²"
        + (f"  gyro_thr={gyro_thr:.2f} rad/s" if gyro_thr else "")
    )

    # Pass raw_df so cluster_peaks can compute per-event jerk
    events = cluster_peaks(peaks, raw_df=df)

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
distance_km  = total_distance_m / 1000
duration_min = total_duration_s / 60

print("\n" + "="*55)
print(" *** VERTICAL-CENTRIC PIPELINE SUMMARY ***")
print("="*55)
print(f" [+] Total Trips        : {total_trips}")
print(f" [+] Total Distance     : {distance_km:.2f} km")
print(f" [+] Total Duration     : {duration_min:.2f} min")
if total_duration_s > 0:
    print(f" [+] Average Speed      : {(total_distance_m / total_duration_s) * 3.6:.2f} km/h")
print("-" * 55)

if not candidates_df.empty:
    pri_counts = candidates_df["priority"].value_counts()
    lvl_counts = candidates_df["level"].value_counts()
    print(f" [!] Total Events Detected : {total_events}")
    if distance_km > 0:
        print(f" [!] Events per km         : {total_events / distance_km:.2f}")
    
    print("\n [ Priority Breakdown ]")
    print(f"   * High   (>= 0.60) : {int(pri_counts.get('high', 0))} events")
    print(f"   * Medium (>= 0.30) : {int(pri_counts.get('medium', 0))} events")
    print(f"   * Low    (<  0.30) : {int(pri_counts.get('low', 0))} events")
    
    print("\n [ Legacy Level Breakdown ]")
    print(f"   * High Conf (abs(vert) >= {HIGH_CONF_VERT_G} G) : {int(lvl_counts.get('high_conf', 0))}")
    print(f"   * Candidate (abs(vert) >= {CANDIDATE_VERT_G} G) : {int(lvl_counts.get('candidate', 0))}")
    print(f"   * Normal                         : {int(lvl_counts.get('normal', 0))}")

    # Speed context breakdown
    speed_valid = candidates_df["speed_mean"].dropna()
    if not speed_valid.empty:
        n_low  = int((speed_valid < SPEED_LOW_MS).sum())
        n_mid  = int(((speed_valid >= SPEED_LOW_MS) & (speed_valid < SPEED_HIGH_MS)).sum())
        n_high = int((speed_valid >= SPEED_HIGH_MS).sum())
        print("\n [ Speed Context ]")
        print(f"   * < {SPEED_LOW_MS*3.6:.0f} km/h     : {n_low} events")
        print(f"   * {SPEED_LOW_MS*3.6:.0f}-{SPEED_HIGH_MS*3.6:.0f} km/h : {n_mid} events")
        print(f"   * >= {SPEED_HIGH_MS*3.6:.0f} km/h    : {n_high} events")
else:
    print(" [!] Total Events       : 0 (no candidates detected)")

print("-" * 55)
print(f" [+] Total Windows Processed: {total_windows}")

# ---------- FEATURE DISTRIBUTIONS ----------
if not windows_df.empty:
    if "vert_max" in windows_df.columns:
        print("\n [~] VERTICAL ACCELERATION (vert_max, m/s²)")
        print(" " + "-"*40)
        print(windows_df["vert_max"].describe().to_string())

    if "gyro_mag_max" in windows_df.columns:
        valid_gyro = windows_df[windows_df["gyro_mag_max"] > 0]["gyro_mag_max"]
        if not valid_gyro.empty:
            print("\n [~] GYROSCOPE (gyro_mag_max, rad/s)")
            print(" " + "-"*40)
            print(valid_gyro.describe().to_string())
            print(f"   Windows with gyro data : {len(valid_gyro)} / {len(windows_df)}")

print("="*55 + "\n")