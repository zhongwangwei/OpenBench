# tests/test_netcdf_writer.py
import pytest
from src.writers.netcdf_writer import NetCDFWriter
from src.core.station import Station, StationList


class TestNetCDFWriterFilter:
    """Test station filtering logic."""

    def test_filter_by_uparea(self):
        """Should filter stations with uparea > min_uparea"""
        writer = NetCDFWriter({'min_uparea': 100.0})

        stations = StationList()

        # Station with uparea > 100 (should pass)
        s1 = Station(id='1', name='S1', lon=0, lat=0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}
        stations.add(s1)

        # Station with uparea < 100 (should fail)
        s2 = Station(id='2', name='S2', lon=1, lat=1, source='cgls')
        s2.cama_results = {'glb_03min': {'flag': 20, 'uparea': 50.0}}
        stations.add(s2)

        # Station with no CaMa result (should fail)
        s3 = Station(id='3', name='S3', lon=2, lat=2, source='icesat')
        s3.cama_results = {}
        stations.add(s3)

        # Station with flag=0 (should fail)
        s4 = Station(id='4', name='S4', lon=3, lat=3, source='hydrosat')
        s4.cama_results = {'glb_03min': {'flag': 0, 'uparea': 200.0}}
        stations.add(s4)

        filtered = writer._filter_stations(stations)

        assert len(filtered) == 1
        assert filtered[0].id == '1'

    def test_filter_any_resolution_passes(self):
        """Station passes if ANY resolution has valid uparea"""
        writer = NetCDFWriter({'min_uparea': 100.0})

        stations = StationList()
        s1 = Station(id='1', name='S1', lon=0, lat=0, source='hydroweb')
        s1.cama_results = {
            'glb_01min': {'flag': 0, 'uparea': 50.0},   # fail
            'glb_03min': {'flag': 20, 'uparea': 150.0}, # pass
        }
        stations.add(s1)

        filtered = writer._filter_stations(stations)
        assert len(filtered) == 1
