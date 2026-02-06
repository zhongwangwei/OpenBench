# tests/test_models.py
import numpy as np
from src.models import StationTimeSeries, StationMetadata, StationDataset

def test_station_time_series():
    ts = StationTimeSeries(
        station_id="GRDC_1234",
        discharge=np.array([1.0, 2.0, np.nan]),
        time=np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64"),
        latitude=50.0, longitude=10.0, upstream_area=100.0,
    )
    assert ts.station_id == "GRDC_1234"
    assert len(ts.discharge) == 3
    assert ts.valid_count == 2

def test_station_metadata():
    meta = StationMetadata(
        name="Test Station", country="DE",
        source_file="test.nc", source_variable="Q", source_unit="m3/s",
    )
    assert meta.source_crs == "WGS84"  # default

def test_station_dataset():
    ts = StationTimeSeries(
        station_id="A", discharge=np.array([1.0]),
        time=np.array(["2020-01-01"], dtype="datetime64"),
        latitude=50.0, longitude=10.0, upstream_area=100.0,
    )
    meta = StationMetadata(name="A", source_file="a.nc", source_variable="Q", source_unit="m3/s")
    ds = StationDataset(
        source_name="test", time_resolution="daily",
        timezone_type="utc", timezone_utc_offset=0.0, timezone_definition="UTC",
        stations=[ts], metadata={"A": meta},
    )
    assert ds.station_count == 1
