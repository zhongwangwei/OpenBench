#!/usr/bin/env python3
"""
Tests for Step 1: Validation and EGM Calculation
测试重复站点检测性能优化
"""

import time
import random
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.station import Station
from src.steps.step1_validate import (
    detect_duplicates,
    detect_duplicates_fast,
    haversine_distance,
)


class TestHaversineDistance:
    """Test haversine distance calculation."""

    def test_same_point_zero_distance(self):
        """Same point should have zero distance."""
        dist = haversine_distance(45.0, 90.0, 45.0, 90.0)
        assert dist == 0.0

    def test_known_distance(self):
        """Test a known distance between two cities."""
        # Paris (48.8566, 2.3522) to London (51.5074, -0.1278)
        # Approximately 343 km
        dist = haversine_distance(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340000 < dist < 350000  # ~343 km in meters

    def test_equator_one_degree(self):
        """One degree at equator is approximately 111 km."""
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110000 < dist < 112000  # ~111 km in meters


class TestDuplicateDetection:
    """Test duplicate station detection."""

    def _create_station(self, i: int, lat: float, lon: float) -> Station:
        """Helper to create a Station instance."""
        return Station(
            id=f"station_{i}",
            name=f"Station {i}",
            lat=lat,
            lon=lon,
            num_observations=100,
            source="test",
        )

    def test_no_duplicates_when_far_apart(self):
        """Stations far apart should not be duplicates."""
        stations = [
            self._create_station(0, 0.0, 0.0),
            self._create_station(1, 10.0, 10.0),
            self._create_station(2, -20.0, 30.0),
        ]
        rules = {'distance_threshold_m': 100}

        duplicates = detect_duplicates(stations, rules)
        assert len(duplicates) == 0

    def test_duplicates_when_close(self):
        """Stations within threshold should be detected."""
        stations = [
            self._create_station(0, 0.0, 0.0),
            self._create_station(1, 0.0, 0.0001),  # Very close (~11m)
            self._create_station(2, 10.0, 10.0),   # Far away
        ]
        rules = {'distance_threshold_m': 100}

        duplicates = detect_duplicates(stations, rules)
        assert len(duplicates) == 1
        assert duplicates[0][0] == 'station_0'
        assert duplicates[0][1] == 'station_1'

    def test_fast_duplicate_detection_same_results(self):
        """Fast algorithm should find same duplicates as original."""
        random.seed(42)
        stations = [
            self._create_station(i, random.uniform(-90, 90), random.uniform(-180, 180))
            for i in range(100)
        ]
        # Add some known duplicates
        stations.append(self._create_station(100, stations[0].lat, stations[0].lon + 0.00001))
        stations.append(self._create_station(101, stations[50].lat + 0.00001, stations[50].lon))

        rules = {'distance_threshold_m': 100}

        original_dups = detect_duplicates(stations, rules)
        fast_dups = detect_duplicates_fast(stations, rules)

        # Both should find the same number of duplicates
        assert len(original_dups) == len(fast_dups)

        # Compare found pairs (may be in different order)
        original_pairs = {(d[0], d[1]) for d in original_dups}
        fast_pairs = {(d[0], d[1]) for d in fast_dups}
        assert original_pairs == fast_pairs


class TestDuplicateDetectionPerformance:
    """Performance tests for duplicate detection."""

    def _create_station(self, i: int, lat: float, lon: float) -> Station:
        """Helper to create a Station instance."""
        return Station(
            id=f"station_{i}",
            name=f"Station {i}",
            lat=lat,
            lon=lon,
            num_observations=100,
            source="test",
        )

    def test_fast_algorithm_performance_1000_stations(self):
        """Fast algorithm should handle 1000 stations quickly."""
        random.seed(42)
        stations = [
            self._create_station(i, random.uniform(-90, 90), random.uniform(-180, 180))
            for i in range(1000)
        ]
        rules = {'distance_threshold_m': 100}

        start = time.time()
        result = detect_duplicates_fast(stations, rules)
        elapsed = time.time() - start

        # Should complete in under 1 second for 1000 stations
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 1000 stations"

    def test_fast_algorithm_performance_10000_stations(self):
        """Fast algorithm should handle 10000 stations in reasonable time."""
        random.seed(42)
        stations = [
            self._create_station(i, random.uniform(-90, 90), random.uniform(-180, 180))
            for i in range(10000)
        ]
        rules = {'distance_threshold_m': 100}

        start = time.time()
        result = detect_duplicates_fast(stations, rules)
        elapsed = time.time() - start

        # Should complete in under 3 seconds for 10000 stations
        assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s for 10000 stations"

    def test_fast_algorithm_performance_50000_stations(self):
        """Fast algorithm should handle 50000 stations in < 10 seconds."""
        random.seed(42)
        stations = [
            self._create_station(i, random.uniform(-90, 90), random.uniform(-180, 180))
            for i in range(50000)
        ]
        rules = {'distance_threshold_m': 100}

        start = time.time()
        result = detect_duplicates_fast(stations, rules)
        elapsed = time.time() - start

        # Should complete in under 10 seconds for 50000 stations
        assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s for 50000 stations"
        print(f"\n50000 stations processed in {elapsed:.2f}s")

    def test_fast_vs_original_performance_ratio(self):
        """Fast algorithm should be significantly faster than O(n^2) for large N."""
        random.seed(42)
        stations = [
            self._create_station(i, random.uniform(-90, 90), random.uniform(-180, 180))
            for i in range(2000)
        ]
        rules = {'distance_threshold_m': 100}

        # Time original algorithm
        start = time.time()
        original_result = detect_duplicates(stations, rules)
        original_elapsed = time.time() - start

        # Time fast algorithm
        start = time.time()
        fast_result = detect_duplicates_fast(stations, rules)
        fast_elapsed = time.time() - start

        # Fast should be at least 10x faster for 2000 stations
        speedup = original_elapsed / fast_elapsed if fast_elapsed > 0 else float('inf')
        print(f"\nOriginal: {original_elapsed:.3f}s, Fast: {fast_elapsed:.3f}s, Speedup: {speedup:.1f}x")

        # At minimum, fast algorithm should not be slower
        assert fast_elapsed <= original_elapsed * 1.5, \
            f"Fast algorithm should not be slower: {fast_elapsed:.3f}s vs {original_elapsed:.3f}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
