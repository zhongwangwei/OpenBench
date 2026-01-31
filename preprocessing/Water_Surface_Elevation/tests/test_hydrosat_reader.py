# tests/test_hydrosat_reader.py
"""
Tests for HydroSatReader.read_timeseries() method.

This test module covers the read_timeseries functionality that enables
NetCDF export for HydroSat data sources.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from src.readers.hydrosat_reader import HydroSatReader


class TestHydroSatReaderTimeseries:
    """Test HydroSatReader.read_timeseries()"""

    def test_read_timeseries_returns_list(self, tmp_path):
        """read_timeseries should return a list of dicts with datetime/elevation"""
        # Create test file
        test_file = tmp_path / "test_station.txt"
        test_file.write_text("""# hydrosat_no.: 12345
# object: Amazon
# latitude: -3.5
# longitude: -60.0
# DATA
2020,1,15,100.5,0.1
2020,2,20,101.2,0.2
2020,3,25,99.8,0.15
""")
        reader = HydroSatReader()
        ts = reader.read_timeseries(str(test_file))

        assert isinstance(ts, list)
        assert len(ts) == 3
        assert 'datetime' in ts[0]
        assert 'elevation' in ts[0]
        assert ts[0]['elevation'] == 100.5

    def test_read_timeseries_file_not_found(self, tmp_path):
        """read_timeseries should return empty list for missing file"""
        reader = HydroSatReader()
        ts = reader.read_timeseries(str(tmp_path / "nonexistent.txt"))
        assert ts == []

    def test_read_timeseries_includes_uncertainty(self, tmp_path):
        """read_timeseries should include uncertainty field"""
        test_file = tmp_path / "test_uncertainty.txt"
        test_file.write_text("""# hydrosat_no.: 12345
# object: Amazon
# latitude: -3.5
# longitude: -60.0
# DATA
2020,1,15,100.5,0.1
""")
        reader = HydroSatReader()
        ts = reader.read_timeseries(str(test_file))

        assert len(ts) == 1
        assert 'uncertainty' in ts[0]
        assert ts[0]['uncertainty'] == 0.1

    def test_read_timeseries_empty_file(self, tmp_path):
        """read_timeseries should return empty list for file with no data"""
        test_file = tmp_path / "empty_data.txt"
        test_file.write_text("""# hydrosat_no.: 12345
# object: Amazon
# latitude: -3.5
# longitude: -60.0
# No data lines
""")
        reader = HydroSatReader()
        ts = reader.read_timeseries(str(test_file))
        assert ts == []

    def test_read_timeseries_datetime_format(self, tmp_path):
        """read_timeseries should return proper datetime objects"""
        from datetime import datetime

        test_file = tmp_path / "test_datetime.txt"
        test_file.write_text("""# hydrosat_no.: 12345
# object: Amazon
# DATA
2020,6,15,100.5,0.1
""")
        reader = HydroSatReader()
        ts = reader.read_timeseries(str(test_file))

        assert len(ts) == 1
        assert ts[0]['datetime'] == datetime(2020, 6, 15)

    def test_read_timeseries_permission_error(self):
        """read_timeseries should return empty list for permission denied"""
        reader = HydroSatReader()
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            ts = reader.read_timeseries("/some/path.txt")
        assert ts == []
