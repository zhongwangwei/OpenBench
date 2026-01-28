#!/usr/bin/env python3
"""
Unit tests for AllocateVS core functions
Tests the logic without requiring full CaMa data files
"""

import numpy as np
import math
import sys

# Import the class
from AllocateVS import AllocateVS


def test_D8_directions():
    """Test D8 direction encoding
    Note: D8 uses angle-based thresholds (tan(22.5°) and tan(67.5°))
    For exact diagonals like (1,1), the angle falls in the middle range
    """
    allocator = AllocateVS()

    # Test cardinal directions and near-cardinal angles
    # D8 graph:
    # |-----------|
    # | 8 | 1 | 2 |
    # |-----------|
    # | 7 | 0 | 3 |
    # |-----------|
    # | 6 | 5 | 4 |
    # |-----------|
    #
    # The function uses angle thresholds:
    # - tan(22.5°) = 0.4142135
    # - tan(67.5°) = 2.4142135
    # For exact 45° (dy/dx = 1), falls in middle range

    # For the Southeast quadrant (dx>0, dy>0):
    # - tan(22.5°) ≈ 0.4142, tan(67.5°) ≈ 2.4142
    # - tval in (0, 0.4142]: returns 4
    # - tval in (0.4142, 2.4142]: returns 5
    # - tval > 2.4142: returns 6
    # For (2,1): tval = 0.5, so returns 5 (not 4!)
    # For (3,1): tval = 0.33, so returns 4
    test_cases = [
        ((0, -1), 1),   # North (cardinal)
        ((2, -1), 2),   # Northeast (angle 26.6° - in range for 2)
        ((1, 0), 3),    # East (cardinal)
        ((3, 1), 4),    # Southeast (tval=0.33 < 0.4142)
        ((0, 1), 5),    # South (cardinal)
        ((1, 1), 5),    # 45° diagonal (tval=1.0) - falls in South range
        ((2, 1), 5),    # tval=0.5, in (0.4142, 2.4142] -> returns 5
        ((-1, 1), 6),   # Southwest
        ((-1, 0), 7),   # West (cardinal)
        ((-1, -1), 8),  # Northwest
    ]

    print("Testing D8 direction encoding...")
    passed = 0
    for (dx, dy), expected in test_cases:
        result = allocator.D8(dx, dy)
        if result == expected:
            passed += 1
            print(f"  D8({dx}, {dy}) = {result} ✓")
        else:
            print(f"  D8({dx}, {dy}) = {result}, expected {expected} ✗")

    print(f"D8 tests: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


def test_next_D8():
    """Test next_D8 direction decoding"""
    allocator = AllocateVS()

    # Test reverse mapping
    test_cases = [
        (1, (0, -1)),   # North
        (2, (1, -1)),   # Northeast
        (3, (1, 0)),    # East
        (4, (1, 1)),    # Southeast
        (5, (0, 1)),    # South
        (6, (-1, 1)),   # Southwest
        (7, (-1, 0)),   # West
        (8, (-1, -1)),  # Northwest
    ]

    print("Testing next_D8 direction decoding...")
    passed = 0
    for dval, expected in test_cases:
        result = allocator.next_D8(dval)
        if result == expected:
            passed += 1
            print(f"  next_D8({dval}) = {result} ✓")
        else:
            print(f"  next_D8({dval}) = {result}, expected {expected} ✗")

    print(f"next_D8 tests: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


def test_hubeny_distance():
    """Test Hubeny distance calculation"""
    allocator = AllocateVS()

    # Test case: Tokyo to Osaka (approximately 400km)
    # Tokyo: 35.6762° N, 139.6503° E
    # Osaka: 34.6937° N, 135.5023° E
    dist = allocator.hubeny_real(35.6762, 139.6503, 34.6937, 135.5023)
    expected = 400000  # approximately 400km

    print("Testing Hubeny distance calculation...")
    # Allow 5% error
    if abs(dist - expected) / expected < 0.05:
        print(f"  Tokyo-Osaka distance: {dist/1000:.1f} km (expected ~400 km) ✓")
        return True
    else:
        print(f"  Tokyo-Osaka distance: {dist/1000:.1f} km (expected ~400 km) ✗")
        return False


def test_set_name():
    """Test tile name generation"""
    allocator = AllocateVS()

    test_cases = [
        ((0, 0), "n00e000"),
        ((-180, -60), "s60w180"),
        ((170, 80), "n80e170"),
        ((-10, -10), "s10w010"),
        ((100, 50), "n50e100"),
    ]

    print("Testing set_name tile naming...")
    passed = 0
    for (lon, lat), expected in test_cases:
        result = allocator.set_name(lon, lat)
        if result == expected:
            passed += 1
            print(f"  set_name({lon}, {lat}) = '{result}' ✓")
        else:
            print(f"  set_name({lon}, {lat}) = '{result}', expected '{expected}' ✗")

    print(f"set_name tests: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


def test_search_range():
    """Verify search range is set to 60 (matching Fortran)"""
    print("Testing search range parameters...")

    # Read the source file and check nn values
    with open('AllocateVS.py', 'r') as f:
        content = f.read()

    # Check find_nearest_river has nn = 60
    if 'nn = 60' in content:
        print("  Search range nn = 60 found ✓")
        return True
    else:
        print("  Search range nn = 60 NOT found ✗")
        return False


def test_mock_find_nearest_river():
    """Test find_nearest_river with mock data"""
    print("Testing find_nearest_river with mock data...")

    allocator = AllocateVS()

    # Setup mock data
    nx, ny = 100, 100
    allocator.nx = nx
    allocator.ny = ny

    # Create mock arrays
    allocator.visual = np.zeros((ny, nx), dtype=np.int32)
    allocator.catmXX = np.ones((ny, nx), dtype=np.int32)
    allocator.catmYY = np.ones((ny, nx), dtype=np.int32)

    # Place a river centerline at (50, 50)
    allocator.visual[49, 49] = 10  # River centerline (0-based index)

    # Test from position (55, 55) - should find river at (50, 50)
    kx, ky, lag = allocator.find_nearest_river(55, 55)

    expected_kx, expected_ky = 50, 50
    expected_lag = math.sqrt(5**2 + 5**2)

    if kx == expected_kx and ky == expected_ky:
        print(f"  Found river at ({kx}, {ky}), lag={lag:.2f} ✓")
        return True
    else:
        print(f"  Found river at ({kx}, {ky}), expected ({expected_kx}, {expected_ky}) ✗")
        return False


def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("AllocateVS Unit Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("D8 directions", test_D8_directions()))
    results.append(("next_D8", test_next_D8()))
    results.append(("Hubeny distance", test_hubeny_distance()))
    results.append(("set_name", test_set_name()))
    results.append(("search_range", test_search_range()))
    results.append(("find_nearest_river (mock)", test_mock_find_nearest_river()))

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
