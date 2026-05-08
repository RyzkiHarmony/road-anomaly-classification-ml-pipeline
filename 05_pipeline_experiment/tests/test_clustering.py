# test_clustering.py
# Unit tests for clustering.cluster_peaks().
#
# Run with:
#   cd labeling
#   python -m pytest tests/ -v

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clustering import cluster_peaks


def _make_peaks(**kwargs) -> pd.DataFrame:
    """Build a minimal peaks DataFrame with all columns required by cluster_peaks()."""
    n = len(kwargs.get("time_s", [0.0]))
    defaults = {
        "time_s":        [0.0] * n,
        "lat":           [-7.0] * n,
        "lon":           [110.0] * n,
        "peak_mag":      [30.0] * n,
        "peak_vertical": [30.0] * n,
        "peak_gyro_mag": [2.0]  * n,
        "peak_speed":    [5.0]  * n,
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestClusterPeaks:
    """Tests for the temporal/spatial peak clustering function."""

    def test_empty_peaks_returns_empty(self):
        """Empty input must return empty DataFrame and unchanged event_id."""
        df, next_id = cluster_peaks(pd.DataFrame(), start_event_id=0)
        assert df.empty
        assert next_id == 0

    def test_single_peak_becomes_one_event(self):
        """A single peak must produce exactly one event."""
        peaks = _make_peaks(time_s=[1.0])
        df, next_id = cluster_peaks(peaks, start_event_id=0)
        assert len(df) == 1
        assert next_id == 1

    def test_two_nearby_peaks_merge_into_one_event(self):
        """Peaks within CLUSTER_TIME_S (2s) and same GPS point should merge."""
        peaks = _make_peaks(time_s=[0.0, 0.5])   # 0.5s apart → same cluster
        df, next_id = cluster_peaks(peaks, start_event_id=0)
        assert len(df) == 1
        assert next_id == 1

    def test_two_distant_peaks_become_two_events(self):
        """Peaks more than CLUSTER_TIME_S (2s) apart should become separate events."""
        peaks = _make_peaks(time_s=[0.0, 5.0])   # 5s apart → different clusters
        df, next_id = cluster_peaks(peaks, start_event_id=0)
        assert len(df) == 2
        assert next_id == 2

    def test_event_id_counter_increments(self):
        """Event IDs must start from start_event_id and be unique."""
        peaks = _make_peaks(time_s=[0.0, 5.0, 10.0])
        df, next_id = cluster_peaks(peaks, start_event_id=100)
        assert len(df) == 3
        assert set(df["event_id"].tolist()) == {100, 101, 102}
        assert next_id == 103

    def test_start_event_id_respected(self):
        """start_event_id parameter must offset all assigned event IDs."""
        peaks = _make_peaks(time_s=[0.0])
        df, _ = cluster_peaks(peaks, start_event_id=42)
        assert df["event_id"].iloc[0] == 42

    def test_output_has_required_columns(self):
        """Output DataFrame must contain the core columns needed downstream."""
        required_cols = {
            "event_id", "time_s", "lat", "lon",
            "peak_mag", "peak_vertical", "peak_vertical_g",
            "peak_gyro_mag", "speed_mean", "level",
        }
        peaks = _make_peaks(time_s=[0.0])
        df, _ = cluster_peaks(peaks, start_event_id=0)
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_level_classification(self):
        """Peaks with a_vertical > HIGH_CONF_VERT_G (8G) should get level 'high_conf'."""
        # 8G * 9.80665 = 78.45 m/s²
        high_g_ms2 = 8.5 * 9.80665
        peaks = _make_peaks(
            time_s         =[0.0],
            peak_vertical  =[high_g_ms2],
            peak_gyro_mag  =[0.0],
        )
        df, _ = cluster_peaks(peaks, start_event_id=0)
        assert df["level"].iloc[0] == "high_conf"

    def test_chain_cluster_continuity(self):
        """next_event_id returned from one call should be valid input for next call."""
        peaks_trip1 = _make_peaks(time_s=[0.0, 5.0])
        peaks_trip2 = _make_peaks(time_s=[100.0, 105.0])

        df1, next_id = cluster_peaks(peaks_trip1, start_event_id=0)
        df2, final_id = cluster_peaks(peaks_trip2, start_event_id=next_id)

        all_ids = set(df1["event_id"].tolist()) | set(df2["event_id"].tolist())
        assert len(all_ids) == 4, "All event IDs must be unique across trips"
        assert final_id == 4
