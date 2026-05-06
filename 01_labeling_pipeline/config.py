# config.py
# Konfigurasi dan konstanta untuk pipeline deteksi anomali jalan.
#
# Semua threshold, weight, dan parameter pipeline dikumpulkan di satu file
# agar mudah di-tune tanpa mengubah logic di modul lain.

import os
import logging

# ---------- PATHS ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER  = os.path.join(_SCRIPT_DIR, "data", "csv")
META_FOLDER = os.path.join(_SCRIPT_DIR, "data", "meta")
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


# ---------- WINDOW & PEAK PARAMETERS ----------
WINDOW_S            = 1.0   # detik per sliding window
OVERLAP             = 0.5   # fraksi overlap antar window
PEAK_MIN_DISTANCE_S = 0.2   # jarak minimum antar peak (detik)
CLUSTER_TIME_S      = 2.0   # maksimum gap waktu dalam satu cluster
CLUSTER_SPATIAL_M   = 10.0  # maksimum jarak GPS dalam satu cluster (meter)

# ---------- ACCELEROMETER SEVERITY THRESHOLDS (G-force, a_vertical) ----------
# Threshold berbasis G-force untuk konsistensi lintas perangkat.
NORMAL_VERT_G    = 3.0   # ambang bawah kandidat event
CANDIDATE_VERT_G = 5.0   # event diprioritaskan untuk labeling
HIGH_CONF_VERT_G = 8.0   # event sangat meyakinkan

G_TO_MS2           = 9.80665
NORMAL_VERT_MS2    = NORMAL_VERT_G    * G_TO_MS2
CANDIDATE_VERT_MS2 = CANDIDATE_VERT_G * G_TO_MS2
HIGH_CONF_VERT_MS2 = HIGH_CONF_VERT_G * G_TO_MS2

# ---------- GYROSCOPE SEVERITY THRESHOLDS (rad/s) ----------
GYRO_NORMAL_RAD    = 3.0   # minimum untuk trigger peak detection
GYRO_CANDIDATE_RAD = 4.0   # event dinaikan ke level candidate
GYRO_HIGH_CONF_RAD = 8.0   # event dinaikan ke level high_conf

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
SCORE_ACCEL_MIN_G = 3.0    # sama dengan NORMAL_VERT_G
SCORE_ACCEL_MAX_G = 8.0    # sama dengan HIGH_CONF_VERT_G
SCORE_GYRO_MAX    = 6.0    # rad/s — gyro > 6 sangat jarang, dianggap saturasi
SCORE_JERK_MAX    = 15.0   # m/s³ — jerk > 15 adalah impact yang sangat keras
SCORE_DUR_MAX_S   = 2.0    # detik — event > 2s biasanya multi-event atau slip

# ---------- PRIORITY THRESHOLDS ----------
PRIORITY_HIGH_THRESHOLD   = 0.60
PRIORITY_MEDIUM_THRESHOLD = 0.30
