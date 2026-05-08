# clustering.py
# Pengelompokan peak anomali menjadi event berdasarkan kedekatan
# temporal dan spasial.
#
# DESIGN NOTES:
# - Clustering rule: peak baru bergabung ke cluster jika time distance
#   dari peak terakhir dalam cluster <= CLUSTER_TIME_S DAN haversine
#   distance <= CLUSTER_SPATIAL_M.
# - Setiap event menghasilkan fitur agregat (max accel, max gyro, jerk,
#   speed, duration, shape features) untuk scoring dan labeling.

import numpy as np
import pandas as pd

from config import (
    CLUSTER_TIME_S,
    CLUSTER_SPATIAL_M,
    G_TO_MS2,
    HIGH_CONF_VERT_G,
    CANDIDATE_VERT_G,
    GYRO_HIGH_CONF_RAD,
    GYRO_CANDIDATE_RAD,
    get_logger,
)
from helpers import haversine
from feature_extraction import extract_event_shape_features

logger = get_logger(__name__)


def _make_cluster(event_id: int, row) -> dict:
    """Initialise a new cluster dict from the first peak row."""
    return dict(
        event_id  = event_id,
        times     = [row.time_s],
        lats      = [row.lat],
        lons      = [row.lon],
        mags      = [row.peak_mag],
        mags_vert = [row.peak_vertical],
        gyro_mags = [row.peak_gyro_mag],
        speeds    = [row.peak_speed],
    )


def _append_to_cluster(cluster: dict, row) -> None:
    """Add a peak row into an existing cluster in-place."""
    cluster["times"].append(row.time_s)
    cluster["lats"].append(row.lat)
    cluster["lons"].append(row.lon)
    cluster["mags"].append(row.peak_mag)
    cluster["mags_vert"].append(row.peak_vertical)
    cluster["gyro_mags"].append(row.peak_gyro_mag)
    cluster["speeds"].append(row.peak_speed)


def _classify_level(vert_g: float, gyro_rads: float) -> str:
    """Return severity level string based on sensor magnitudes."""
    if abs(vert_g) >= HIGH_CONF_VERT_G or gyro_rads >= GYRO_HIGH_CONF_RAD:
        return "high_conf"
    if abs(vert_g) >= CANDIDATE_VERT_G or gyro_rads >= GYRO_CANDIDATE_RAD:
        return "candidate"
    return "normal"




def _cluster_to_event_row(cluster: dict, raw_df, raw_times) -> dict:
    """Convert a cluster dict into one event feature row."""
    # Gunakan peak vertikal terkuat sebagai representasi waktu utama event
    dominant_vert_idx = int(np.argmax(np.abs(cluster["mags_vert"])))
    max_vert_ms2      = float(cluster["mags_vert"][dominant_vert_idx])
    time_dominant     = float(cluster["times"][dominant_vert_idx])
    
    max_accel_ms2     = float(np.max(cluster["mags"]))
    max_gyro_rads     = float(np.max(cluster["gyro_mags"]))

    accel_g = max_accel_ms2 / G_TO_MS2
    vert_g  = max_vert_ms2  / G_TO_MS2

    # Ekstraksi fitur shape berpusat pada peak terkuat
    shape_features = extract_event_shape_features(raw_df, time_dominant)

    valid_speeds = [s for s in cluster["speeds"] if not np.isnan(s)]
    speed_mean   = float(np.mean(valid_speeds)) if valid_speeds else float("nan")

    level = _classify_level(vert_g, max_gyro_rads)

    row = {
        "event_id":         cluster["event_id"],
        "time_s":           time_dominant,
        "lat":              float(cluster["lats"][dominant_vert_idx]),
        "lon":              float(cluster["lons"][dominant_vert_idx]),
        "peak_mag":         max_accel_ms2,
        "peak_vertical":    max_vert_ms2,
        "peak_mag_g":       accel_g,
        "peak_vertical_g":  vert_g,
        "peak_gyro_mag":    max_gyro_rads,
        "speed_mean":       speed_mean,
        "event_duration":   shape_features["duration_above_threshold"],
        "vert_jrk":         shape_features["max_jerk"], # Konsisten dengan extraction logic
        "level":            level,
    }
    # Merge all shape features (21 columns) into the row
    row.update(shape_features)
    return row


def cluster_peaks(peaks: pd.DataFrame, raw_df=None, start_event_id: int = 0):
    """Group temporally and spatially proximate peaks into single events.

    Parameters
    ----------
    peaks : DataFrame
        Output from detect_peaks().
    raw_df : DataFrame, optional
        Raw trip DataFrame used to compute per-event jerk and shape features.
    start_event_id : int
        Starting event ID.  Pass the last returned ``next_event_id`` across
        trips to get globally unique IDs without global mutable state.

    Returns
    -------
    events_df : DataFrame
        One row per event with aggregated features.
    next_event_id : int
        The event ID to use for the next call (= last assigned ID + 1).
    """
    if peaks.empty:
        logger.debug("cluster_peaks: no peaks to cluster, returning empty DataFrame")
        return pd.DataFrame(), start_event_id

    # Pre-compute raw timestamps for jerk calculation
    raw_times = None
    if raw_df is not None and not raw_df.empty:
        raw_sorted = raw_df.sort_values("timestamp").reset_index(drop=True)
        raw_times  = raw_sorted["timestamp"].astype(float).values / 1000.0

    # Sequential greedy clustering
    event_id       = start_event_id
    finished_clusters: list[dict] = []
    active_cluster: dict | None = None

    for _, peak_row in peaks.iterrows():
        if active_cluster is None:
            active_cluster = _make_cluster(event_id, peak_row)
            event_id += 1
            continue

        time_gap = abs(peak_row.time_s - active_cluster["times"][-1])
        if time_gap <= CLUSTER_TIME_S:
            spatial_dist = haversine(
                active_cluster["lats"][-1], active_cluster["lons"][-1],
                peak_row.lat, peak_row.lon,
            )
            if spatial_dist <= CLUSTER_SPATIAL_M:
                _append_to_cluster(active_cluster, peak_row)
                continue

        # Current peak does not belong to active cluster → start new one
        finished_clusters.append(active_cluster)
        active_cluster = _make_cluster(event_id, peak_row)
        event_id += 1

    if active_cluster is not None:
        finished_clusters.append(active_cluster)

    # Convert each cluster into an event feature row
    event_rows = [
        _cluster_to_event_row(cluster, raw_df, raw_times)
        for cluster in finished_clusters
    ]

    logger.debug(
        "cluster_peaks: %d peaks → %d events (IDs %d–%d)",
        len(peaks), len(event_rows), start_event_id, event_id - 1,
    )

    return pd.DataFrame(event_rows), event_id
