# tests/test_unit_converter.py
import numpy as np
from src.utils.unit_converter import convert_discharge, convert_area, convert_mmd_to_m3s

def test_convert_discharge_m3s():
    """m3/s to m3/s is identity."""
    assert convert_discharge(10.0, "m3/s") == 10.0

def test_convert_discharge_ft3s():
    result = convert_discharge(100.0, "ft3/s")
    assert abs(result - 2.83168) < 0.001

def test_convert_discharge_ls():
    assert convert_discharge(1000.0, "L/s") == 1.0

def test_convert_mmd_to_m3s():
    """1 mm/d over 86.4 km2 = 1 m3/s."""
    result = convert_mmd_to_m3s(1.0, area_km2=86.4)
    assert abs(result - 1.0) < 0.001

def test_convert_mmd_to_m3s_array():
    q_mmd = np.array([1.0, 2.0, np.nan])
    result = convert_mmd_to_m3s(q_mmd, area_km2=86.4)
    assert abs(result[0] - 1.0) < 0.001
    assert abs(result[1] - 2.0) < 0.001
    assert np.isnan(result[2])

def test_convert_area_km2():
    assert convert_area(1.0, "km2") == 1.0

def test_convert_area_m2():
    assert convert_area(1e6, "m2") == 1.0

def test_convert_area_ha():
    assert convert_area(100.0, "ha") == 1.0
