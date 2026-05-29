# scoring.py
# Composite scoring untuk memprioritaskan event labeling.
#
# Speed-aware composite scoring.  Tujuannya adalah memprioritaskan event
# untuk labeling manual.  Event pada kecepatan rendah TIDAK dibuang,
# melainkan tetap ada di output.
#
# Semua bounds normalisation dan threshold priority didefinisikan di config.py
# agar dapat di-tune tanpa mengubah logic di sini.

import numpy as np
import pandas as pd

from config import (
    W_ACCEL, W_GYRO, W_JERK, W_DURATION,
    SCORE_ACCEL_MIN_G, SCORE_ACCEL_MAX_G,
    SCORE_GYRO_MAX, SCORE_JERK_MAX, SCORE_DUR_MAX_S,
    PRIORITY_HIGH_THRESHOLD, PRIORITY_MEDIUM_THRESHOLD,
    get_logger,
)
from helpers import robust_normalise

logger = get_logger(__name__)


def score_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite priority score for each candidate event.

    Score components (all normalised to [0, 1] before weighting):

    - **accel**    ``peak_vertical_g``    – G-force vertikal; semakin tinggi semakin meyakinkan
    - **gyro**     ``peak_gyro_mag``      – konfirmasi sensor kedua (rad/s)
    - **jerk**     ``vert_jrk``           – perubahan mendadak = impact tajam
    - **duration** ``event_duration``     – event lebih panjang = lebih signifikan

    Normalisation bounds diambil dari ``config.py`` (SCORE_*) berdasarkan
    distribusi empiris 12 trip (92.75 km).

    Priority levels:
    - ``high``   : score >= ``PRIORITY_HIGH_THRESHOLD``   (0.60)
    - ``medium`` : score >= ``PRIORITY_MEDIUM_THRESHOLD`` (0.30)
    - ``low``    : score <  ``PRIORITY_MEDIUM_THRESHOLD``

    Returns
    -------
    DataFrame
        Input DataFrame dengan tambahan kolom: ``speed_factor``, ``score``, ``priority``.
    """
    if events_df.empty:
        logger.debug("score_events: received empty DataFrame, returning as-is")
        events_df["speed_factor"] = []
        events_df["score"]        = []
        events_df["priority"]     = []
        return events_df

    df = events_df.copy()

    # --- Fisika Kecepatan (Low-Speed Boost) ---
    # Guncangan tinggi di kecepatan rendah membutuhkan energi/impact yang jauh lebih masif (lubang sangat dalam).
    # Oleh karena itu, kita memberikan "Boost" eksponensial untuk event dengan kecepatan rendah,
    # dan menetapkan kecepatan normal/tinggi sebagai baseline (1.0).
    speed_kph = df["speed_mean"].values * 3.6 if "speed_mean" in df.columns else np.full(len(df), 30.0)
    
    # Fungsi eksponensial terbalik: 
    # v = 0 km/j -> boost maksimal (x1.5)
    # v >= 30 km/j -> mendekati baseline (x1.0)
    speed_boost = np.where(speed_kph < 30.0, 1.0 + 0.5 * np.exp(-speed_kph / 10.0), 1.0)
    df["speed_factor"] = speed_boost

    # Normalise each component to [0, 1] using robust fixed bounds from config
    n_accel = robust_normalise(
        np.abs(df["peak_vertical_g"]).values, SCORE_ACCEL_MIN_G, SCORE_ACCEL_MAX_G
    )
    n_gyro = robust_normalise(df["peak_gyro_mag"].values, 0.0, SCORE_GYRO_MAX)
    n_jerk = robust_normalise(df["vert_jrk"].values, 0.0, SCORE_JERK_MAX)
    n_dur  = robust_normalise(df["event_duration"].values, 0.0, SCORE_DUR_MAX_S)

    # Base score [0, 1]
    base_score = (
        W_ACCEL    * n_accel
        + W_GYRO   * n_gyro
        + W_JERK   * n_jerk
        + W_DURATION * n_dur
    )
    
    # Kalikan dengan speed_boost dan batasi maksimal 1.0 agar tetap ternormalisasi
    df["score"] = np.clip(base_score * df["speed_factor"], 0.0, 1.0)

    def _to_priority(score: float) -> str:
        if score >= PRIORITY_HIGH_THRESHOLD:
            return "high"
        if score >= PRIORITY_MEDIUM_THRESHOLD:
            return "medium"
        return "low"

    df["priority"] = df["score"].apply(_to_priority)

    high_n   = (df["priority"] == "high").sum()
    medium_n = (df["priority"] == "medium").sum()
    low_n    = (df["priority"] == "low").sum()
    logger.debug(
        "score_events: %d events scored — high=%d medium=%d low=%d",
        len(df), high_n, medium_n, low_n,
    )

    return df
