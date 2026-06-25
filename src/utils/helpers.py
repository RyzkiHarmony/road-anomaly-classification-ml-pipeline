# helpers.py
# Utility functions yang dipakai di berbagai modul pipeline.

import os
import json
import math
import numpy as np

from config import META_FOLDER, get_logger

logger = get_logger(__name__)


def load_trip_meta(csv_path):
    """Load trip metadata JSON yang berpasangan dengan file CSV."""
    base      = os.path.basename(csv_path).rsplit(".", 1)[0]
    json_path = os.path.join(META_FOLDER, base + ".json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json_path, json.load(f)
    return None, None


def haversine(lat1, lon1, lat2, lon2):
    """Hitung jarak haversine (meter) antara dua koordinat GPS."""
    R    = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def validate_magnitude(df, csv_name: str) -> None:
    """
    Confirm that the magnitude column contains values in m/s² (not raw ADC counts).
    Warns if the data appears to be outside a physically plausible range for road
    vibration with a consumer smartphone (expected: 5 – 200 m/s²).
    """
    mags = df["magnitude"].astype(float)
    if mags.max() < 2.0:
        logger.warning(
            "%s: mag_max=%.3f — values look too small. Check units (expected m/s²).",
            csv_name, mags.max(),
        )
    elif mags.max() > 500.0:
        logger.warning(
            "%s: mag_max=%.1f — values look too large. Check units (expected m/s², not raw ADC).",
            csv_name, mags.max(),
        )


def normalise_0_1(values):
    """Min-max normalise an array to [0, 1].  Returns zeros if range is zero."""
    mn, mx = np.min(values), np.max(values)
    rng = mx - mn
    if rng < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mn) / rng


def robust_normalise(values, min_val, max_val):
    """Clip values to [min_val, max_val] and normalise to [0, 1]."""
    clipped = np.clip(values, min_val, max_val)
    rng = max_val - min_val
    if rng < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (clipped - min_val) / rng
