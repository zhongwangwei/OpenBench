# tests/test_crs_converter.py
import numpy as np
from src.utils.crs_converter import reproject_to_wgs84, validate_wgs84

def test_wgs84_passthrough():
    lon, lat = reproject_to_wgs84(10.0, 50.0, "WGS84")
    assert lon == 10.0
    assert lat == 50.0

def test_epsg25832_to_wgs84():
    """Known point: UTM32N (500000, 5500000) ~ (9.0E, 49.6N)."""
    lon, lat = reproject_to_wgs84(500000.0, 5500000.0, "EPSG:25832")
    assert abs(lon - 9.0) < 0.5
    assert abs(lat - 49.6) < 0.5

def test_validate_wgs84_valid():
    assert validate_wgs84(10.0, 50.0) is True

def test_validate_wgs84_invalid_lat():
    assert validate_wgs84(10.0, 100.0) is False

def test_validate_wgs84_invalid_lon():
    assert validate_wgs84(200.0, 50.0) is False

def test_batch_reproject():
    lons = np.array([474625.6, 500000.0])
    lats = np.array([6329837.5, 5500000.0])
    out_lons, out_lats = reproject_to_wgs84(lons, lats, "EPSG:25832")
    assert len(out_lons) == 2
    assert all(-180 <= lon <= 180 for lon in out_lons)
    assert all(-90 <= lat <= 90 for lat in out_lats)
