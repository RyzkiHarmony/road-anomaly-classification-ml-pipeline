# config.py
# Konfigurasi dan konstanta untuk pipeline deteksi anomali jalan.
#
# Semua threshold, weight, dan parameter pipeline dikumpulkan di satu file
# agar mudah di-tune tanpa mengubah logic di modul lain.

import os
import logging

# ---------- PATHS ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER  = os.path.join(_SCRIPT_DIR, "new-data", "csv")
META_FOLDER = os.path.join(_SCRIPT_DIR, "new-data", "meta")
OUT_FOLDER  = os.path.join(_SCRIPT_DIR, "out")
LABELS_FOLDER = os.path.join(_SCRIPT_DIR, "labels")

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

# ---------- LOGGING ----------
# Logging disetup sekali di sini dan dipakai oleh seluruh modul pipeline.
# Level default: INFO (tampilkan progress tanpa debug noise).
# Ubah ke logging.DEBUG untuk troubleshooting, atau logging.WARNING untuk quiet mode.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger untuk modul pipeline.

    Cara pakai::

        from config import get_logger
        logger = get_logger(__name__)
        logger.info("Processing trip %s", trip_id)
    """
    return logging.getLogger(name)

#Pipeline05
# ---------- WINDOW & PEAK PARAMETERS ----------
WINDOW_S              = 2.0   # detik per sliding window
OVERLAP               = 0.5   # fraksi overlap antar window
PEAK_MIN_DISTANCE_S   = 0.2   # jarak minimum antar peak (detik)
REGION_WINDOW_S       = 0.2   # Trailing window untuk rolling energy (detik)
REGION_MAD_MULTIPLIER = 1.2   # Adaptive threshold multiplier untuk region abnormal
CLUSTER_TIME_S        = 0.8   # Maksimum gap waktu dalam satu cluster (detik) untuk latency < 1s
CLUSTER_SPATIAL_M     = 10.0  # maksimum jarak GPS dalam satu cluster (meter)

# ---------- ACCELEROMETER SEVERITY THRESHOLDS (G-force, a_vertical) ----------
# Threshold berbasis G-force untuk konsistensi lintas perangkat.
# Diturunkan untuk mengakomodasi atenuasi amplitudo dari filter kausal & resampling 100Hz
NORMAL_VERT_G    = 1.0   # ambang bawah kandidat event
CANDIDATE_VERT_G = 1.4   # event diprioritaskan untuk labeling
HIGH_CONF_VERT_G = 1.8   # event sangat meyakinkan

G_TO_MS2           = 9.80665
NORMAL_VERT_MS2    = NORMAL_VERT_G    * G_TO_MS2
CANDIDATE_VERT_MS2 = CANDIDATE_VERT_G * G_TO_MS2
HIGH_CONF_VERT_MS2 = HIGH_CONF_VERT_G * G_TO_MS2

# ---------- GYROSCOPE SEVERITY THRESHOLDS (rad/s) ----------
GYRO_NORMAL_RAD    = 1.5   # minimum untuk trigger peak detection
GYRO_CANDIDATE_RAD = 4.0   # event dinaikan ke level candidate
GYRO_HIGH_CONF_RAD = 6.0   # event dinaikan ke level high_conf

# ---------- SPEED CONTEXT THRESHOLDS (m/s) ----------
# Pada motor, kecepatan rendah BUKAN berarti event tidak valid — bisa jadi
# lubang saat belok pelan atau polisi tidur di gang.  Threshold ini hanya
# mempengaruhi skor, bukan membuang data.
SPEED_LOW_MS  = 2.0   # ≈  7.2 km/h – mungkin idle / berhenti
SPEED_HIGH_MS = 8.0   # ≈ 28.8 km/h – kecepatan cruising normal

# ---------- COMPOSITE SCORE WEIGHTS ----------
# Total weight harus = 1.0
# Dipakai di scoring.score_events()
W_ACCEL    = 0.45
W_GYRO     = 0.20
W_JERK     = 0.20
W_DURATION = 0.15

# ---------- COMPOSITE SCORE NORMALISATION BOUNDS ----------
# Batas untuk robust_normalise() di scoring.score_events().
# Nilai di bawah SCORE_*_MIN dikip ke nol; di atas SCORE_*_MAX dikip ke satu.
# Bounds dipilih berdasarkan distribusi empiris dari 12 trip (92.75 km).
SCORE_ACCEL_MIN_G = 1.0    # sama dengan NORMAL_VERT_G
SCORE_ACCEL_MAX_G = 1.8    # Di-adjust dari 6.0 karena causal 100Hz signal smoothing
SCORE_GYRO_MAX    = 6.0    # rad/s — gyro > 6 sangat jarang, dianggap saturasi
SCORE_JERK_MAX    = 400.0  # m/s³ (Direkalibrasi dari 25.0, karena fs 100Hz membuat turunan jerk sangat tinggi)
SCORE_DUR_MAX_S   = 0.15   # detik (Direkalibrasi dari 2.0s karena perhitungan kontigu membuat durasi jauh lebih presisi/pendek)

# ---------- PRIORITY THRESHOLDS ----------
PRIORITY_HIGH_THRESHOLD   = 0.50
PRIORITY_MEDIUM_THRESHOLD = 0.30

# ---------- PREPROCESSING PARAMETERS ----------
TARGET_HZ  = 100      # Target resampling frequency (10ms)

# ---------- ML MODEL FEATURES (SINGLE SOURCE OF TRUTH) ----------
BEST_FEATURES = [
    "event_duration", 
    "gyro_roll_energy", 
    "peak_interval_std", 
    "hjorth_activity", 
    "linear_jerk_3d_max", 
    "snr_vertical", 
    "num_peaks_accel", 
    "peak_interval_mean", 
    "corr_xy", 
    "crest_factor",
    "skewness",
    "min_z_to_max_z_ratio",
    "first_peak_polarity",
    "corr_xz",
    "corr_yz"
]
