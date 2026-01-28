import pytest
from src.core.station import Station, StationList

def test_station_creation():
    s = Station(
        id="ST001",
        name="Test Station",
        lon=100.5,
        lat=30.2,
        source="hydroweb"
    )
    assert s.id == "ST001"
    assert s.lon == 100.5
    assert s.source == "hydroweb"

def test_station_validation():
    s = Station(id="ST001", name="Test", lon=200, lat=30, source="test")
    assert not s.is_valid()  # lon out of range

def test_station_list():
    stations = StationList()
    stations.add(Station("S1", "A", 100, 30, "hydroweb"))
    stations.add(Station("S2", "B", 101, 31, "cgls"))
    assert len(stations) == 2
    assert len(stations.filter_by_source("hydroweb")) == 1
