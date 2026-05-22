# test_scoring.py
# Unit tests for scoring.score_events().
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

from scoring import score_events
from config import PRIORITY_HIGH_THRESHOLD, PRIORITY_MEDIUM_THRESHOLD


def _make_events(**kwargs) -> pd.DataFrame:
    """Build a minimal events DataFrame with required columns for score_events()."""
    defaults = {
        "peak_vertical_g": [5.0],
        "peak_gyro_mag":   [2.0],
        "vert_jrk":        [5.0],
        "event_duration":  [0.5],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestScoreEvents:
    """Tests for the composite scoring function."""

    def test_empty_input_returns_empty(self):
        """score_events on empty DataFrame must add columns and return empty."""
        empty = pd.DataFrame(columns=["peak_vertical_g", "peak_gyro_mag",
                                      "vert_jrk", "event_duration"])
        result = score_events(empty)
        assert "score"        in result.columns
        assert "priority"     in result.columns
        assert "speed_factor" in result.columns
        assert len(result) == 0

    def test_score_is_between_0_and_1(self):
        """Score must always be in [0, 1] for any valid input."""
        df = _make_events(
            peak_vertical_g=[3.0, 5.5, 8.0, 15.0],
            peak_gyro_mag  =[0.0, 2.0, 6.0, 20.0],
            vert_jrk       =[0.0, 5.0, 15.0, 50.0],
            event_duration =[0.0, 0.5, 2.0, 5.0],
        )
        result = score_events(df)
        assert (result["score"] >= 0.0).all(), "Score must be >= 0"
        assert (result["score"] <= 1.0).all(), "Score must be <= 1"

    def test_high_severity_event_gets_high_priority(self):
        """An event with max accel, high gyro, high jerk → priority 'high'."""
        df = _make_events(
            peak_vertical_g=[8.5],   # above SCORE_ACCEL_MAX_G = 6.0
            peak_gyro_mag  =[7.0],   # above SCORE_GYRO_MAX = 6.0
            vert_jrk       =[30.0],  # above SCORE_JERK_MAX = 25.0
            event_duration =[3.0],   # above SCORE_DUR_MAX_S = 2.0
        )
        result = score_events(df)
        assert result["priority"].iloc[0] == "high"
        assert result["score"].iloc[0] == pytest.approx(1.0)

    def test_minimal_event_gets_low_priority(self):
        """An event barely above detection threshold → priority 'low'."""
        df = _make_events(
            peak_vertical_g=[3.01],  # just above SCORE_ACCEL_MIN_G = 3.0
            peak_gyro_mag  =[0.0],
            vert_jrk       =[0.0],
            event_duration =[0.0],
        )
        result = score_events(df)
        assert result["priority"].iloc[0] == "low"

    def test_priority_levels_are_consistent_with_thresholds(self):
        """All 'high' events must have score >= PRIORITY_HIGH_THRESHOLD, etc."""
        df = _make_events(
            peak_vertical_g=[3.0, 5.0, 8.0, 10.0] * 5,
            peak_gyro_mag  =[0.0, 1.0, 3.0, 6.0] * 5,
            vert_jrk       =[0.0, 3.0, 8.0, 15.0] * 5,
            event_duration =[0.0, 0.5, 1.0, 2.0] * 5,
        )
        result = score_events(df)
        for _, row in result.iterrows():
            if row["priority"] == "high":
                assert row["score"] >= PRIORITY_HIGH_THRESHOLD
            elif row["priority"] == "medium":
                assert PRIORITY_MEDIUM_THRESHOLD <= row["score"] < PRIORITY_HIGH_THRESHOLD
            else:
                assert row["score"] < PRIORITY_MEDIUM_THRESHOLD

    def test_speed_factor_is_always_one(self):
        """speed_factor must be 1.0 (speed penalty removed)."""
        df = _make_events()
        result = score_events(df)
        assert (result["speed_factor"] == 1.0).all()

    def test_original_columns_preserved(self):
        """score_events must not drop any original columns."""
        df = _make_events(extra_col=[99.0])
        result = score_events(df)
        assert "extra_col" in result.columns

    def test_higher_accel_gives_higher_score(self):
        """Ceteris paribus, higher vertical G must yield a higher score."""
        low_accel  = _make_events(peak_vertical_g=[3.5], peak_gyro_mag=[0.0],
                                  vert_jrk=[0.0], event_duration=[0.0])
        high_accel = _make_events(peak_vertical_g=[7.5], peak_gyro_mag=[0.0],
                                  vert_jrk=[0.0], event_duration=[0.0])
        s_low  = score_events(low_accel)["score"].iloc[0]
        s_high = score_events(high_accel)["score"].iloc[0]
        assert s_high > s_low
