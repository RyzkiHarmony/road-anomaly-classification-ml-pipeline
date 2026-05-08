# test_helpers.py
# Unit tests for helpers.py utility functions.
#
# Run with:
#   cd labeling
#   python -m pytest tests/ -v

import math
import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure the labeling package root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import haversine, normalise_0_1, robust_normalise


class TestHaversine:
    """Tests for haversine distance calculation."""

    def test_same_point_is_zero(self):
        """Distance between a point and itself must be zero."""
        assert haversine(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_jakarta_to_bandung(self):
        """Approximate distance Jakarta–Bandung ≈ 120 km."""
        # Jakarta: -6.2088, 106.8456
        # Bandung: -6.9175, 107.6191
        dist_m = haversine(-6.2088, 106.8456, -6.9175, 107.6191)
        assert 115_000 < dist_m < 130_000, f"Expected ~120 km, got {dist_m/1000:.1f} km"

    def test_short_distance_cluster_threshold(self):
        """Two GPS points 5 metres apart should be < CLUSTER_SPATIAL_M (10 m)."""
        # Roughly 0.00005° latitude ≈ 5.5 m
        dist_m = haversine(-7.0, 110.0, -7.00005, 110.0)
        assert dist_m < 10.0

    def test_symmetry(self):
        """Distance A→B must equal distance B→A."""
        d1 = haversine(-6.0, 107.0, -7.0, 108.0)
        d2 = haversine(-7.0, 108.0, -6.0, 107.0)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_returns_metres(self):
        """1 degree latitude ≈ 111 km."""
        dist_m = haversine(0.0, 0.0, 1.0, 0.0)
        assert 110_000 < dist_m < 112_000


class TestNormalise01:
    """Tests for min-max normalise_0_1."""

    def test_range_is_zero_to_one(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalise_0_1(values)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_zero_range_returns_zeros(self):
        """All-same values → output should be all zeros (no div-by-zero)."""
        values = np.array([3.0, 3.0, 3.0])
        result = normalise_0_1(values)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_single_element(self):
        result = normalise_0_1(np.array([42.0]))
        assert result[0] == pytest.approx(0.0)

    def test_preserves_order(self):
        values = np.array([10.0, 5.0, 20.0, 1.0])
        result = normalise_0_1(values)
        # Relative ordering must be preserved
        sorted_original = np.argsort(values)
        sorted_result   = np.argsort(result)
        np.testing.assert_array_equal(sorted_original, sorted_result)


class TestRobustNormalise:
    """Tests for clipped/bounded normalise."""

    def test_within_bounds(self):
        values = np.array([3.0, 5.0, 8.0])
        result = robust_normalise(values, 3.0, 8.0)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_below_min_clips_to_zero(self):
        values = np.array([-999.0, 0.0])
        result = robust_normalise(values, 3.0, 8.0)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_above_max_clips_to_one(self):
        values = np.array([999.0, 100.0])
        result = robust_normalise(values, 3.0, 8.0)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_midpoint_is_half(self):
        values = np.array([5.5])  # midpoint of [3, 8]
        result = robust_normalise(values, 3.0, 8.0)
        assert result[0] == pytest.approx(0.5)

    def test_zero_range_returns_zeros(self):
        values = np.array([5.0, 5.0, 5.0])
        result = robust_normalise(values, 5.0, 5.0)
        np.testing.assert_array_equal(result, np.zeros(3))
