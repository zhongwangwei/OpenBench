# tests/test_icesat_reader.py
"""Tests for ICESatReader.read_timeseries()"""
import pytest
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

from src.readers.icesat_reader import ICESatReader


class TestICESatReaderTimeseries:
    """Test ICESatReader.read_timeseries()"""

    def test_read_timeseries_text_format(self, tmp_path):
        """read_timeseries should parse text format files (nXXeXXX.txt)"""
        # Create test file with text format
        # Format: lon lat ? year month day ? elevation
        test_file = tmp_path / "n00e005.txt"
        test_file.write_text("""5.0 0.5 0 2020 1 15 0 100.5
5.0 0.5 0 2020 2 20 0 101.2
5.0 0.5 0 2020 3 25 0 99.8
""")
        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))

        assert isinstance(ts, list)
        assert len(ts) == 3
        assert 'datetime' in ts[0]
        assert 'elevation' in ts[0]
        # First observation: 2020-01-15, 100.5m
        assert ts[0]['elevation'] == 100.5

    def test_read_timeseries_file_not_found(self, tmp_path):
        """read_timeseries should return empty list for missing file"""
        reader = ICESatReader()
        ts = reader.read_timeseries(str(tmp_path / "nonexistent.txt"))
        assert ts == []

    def test_read_timeseries_unknown_format(self, tmp_path):
        """read_timeseries should return empty list for unknown file format"""
        # Create file with unknown naming pattern
        test_file = tmp_path / "random_file.txt"
        test_file.write_text("some content")

        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))
        assert ts == []

    def test_read_timeseries_returns_datetime_format(self, tmp_path):
        """read_timeseries should return proper datetime objects"""
        # Format: lon lat ? year month day ? elevation
        test_file = tmp_path / "s10w060.txt"
        test_file.write_text("""-60.0 -10.0 0 2021 6 15 0 50.5
""")
        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))

        assert len(ts) == 1
        dt = ts[0]['datetime']
        # The reader returns datetime, not date
        from datetime import datetime
        assert dt == datetime(2021, 6, 15)

    def test_read_timeseries_empty_file(self, tmp_path):
        """read_timeseries should return empty list for empty file"""
        test_file = tmp_path / "n45e090.txt"
        test_file.write_text("")

        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))
        assert ts == []

    def test_read_timeseries_header_only(self, tmp_path):
        """read_timeseries should skip header lines starting with #"""
        test_file = tmp_path / "n30w120.txt"
        test_file.write_text("""# Comment line 1
# Comment line 2
# Comment line 3
""")
        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))
        assert ts == []

    @patch.object(ICESatReader, '_read_glah14_observations')
    def test_read_timeseries_glah14_format(self, mock_read, tmp_path):
        """read_timeseries should call _read_glah14_observations for GLAH14 files"""
        # Create a fake GLAH14 file (just need the filename pattern)
        test_file = tmp_path / "GLAH14_634_2131_002_0071_0_01_0001.H5"
        test_file.write_text("")  # Content doesn't matter, we're mocking

        # Mock the GLAH14 reader to return sample data
        mock_read.return_value = [
            {'date': date(2005, 3, 15), 'elevation': 200.5, 'error': 0.5},
            {'date': date(2005, 3, 16), 'elevation': 201.0, 'error': 0.4},
        ]

        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))

        mock_read.assert_called_once()
        assert len(ts) == 2
        assert ts[0]['elevation'] == 200.5
        assert ts[0]['uncertainty'] == 0.5

    @patch.object(ICESatReader, '_read_atl13_observations')
    def test_read_timeseries_atl13_format(self, mock_read, tmp_path):
        """read_timeseries should call _read_atl13_observations for ATL13 files"""
        # Create a fake ATL13 file
        test_file = tmp_path / "ATL13_20200315120000_12345678_006_01.h5"
        test_file.write_text("")

        # Mock the ATL13 reader
        mock_read.return_value = [
            {'date': date(2020, 3, 15), 'elevation': 150.0, 'error': 0.2},
        ]

        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))

        mock_read.assert_called_once()
        assert len(ts) == 1
        assert ts[0]['elevation'] == 150.0
        assert ts[0]['uncertainty'] == 0.2

    def test_read_timeseries_malformed_data(self, tmp_path):
        """read_timeseries should handle malformed lines gracefully"""
        # Format: lon lat ? year month day ? elevation
        test_file = tmp_path / "n00e000.txt"
        test_file.write_text("""0.0 0.5 0 2020 1 15 0 100.5
invalid line
too few parts
0.0 0.5 0 2020 3 25 0 99.8
""")
        reader = ICESatReader()
        ts = reader.read_timeseries(str(test_file))

        # Should still get 2 valid observations (skip malformed lines)
        assert len(ts) == 2
