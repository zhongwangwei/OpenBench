# tests/test_netcdf_writer.py
import pytest
from datetime import date, datetime
import numpy as np
import netCDF4 as nc
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


class TestNetCDFWriterTimeAxis:
    """Test time axis building."""

    def test_build_time_axis_daily(self):
        """Should create daily time axis from start to end date"""
        writer = NetCDFWriter({
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
            'time_reference': '1800-01-01',
        })

        time_axis, time_values = writer._build_time_axis()

        # 5 days: Jan 1-5
        assert len(time_axis) == 5
        assert time_axis[0] == date(2020, 1, 1)
        assert time_axis[-1] == date(2020, 1, 5)

        # Days since 1800-01-01
        # 2020-01-01 is 80353 days after 1800-01-01
        assert time_values[0] == 80353

    def test_date_to_days_since_ref(self):
        """Should convert date to days since reference"""
        writer = NetCDFWriter({'time_reference': '1800-01-01'})

        # 1800-01-01 -> 0
        assert writer._date_to_days(date(1800, 1, 1)) == 0

        # 1800-01-02 -> 1
        assert writer._date_to_days(date(1800, 1, 2)) == 1

        # 2000-01-01 -> 73048
        assert writer._date_to_days(date(2000, 1, 1)) == 73048


class TestNetCDFWriterCreate:
    """Test NetCDF file creation."""

    def test_create_netcdf_structure(self, tmp_path):
        """Should create NetCDF with correct dimensions and variables"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-10',
        })

        # Create mock station list
        stations = StationList()
        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb',
                    elevation=100.0, num_observations=50)
        s1.egm08 = 30.0
        s1.egm96 = 29.5
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0,
                          'lon_cama': 10.1, 'lat_cama': 20.1}}
        stations.add(s1)

        filtered = [s1]
        time_axis, time_values = writer._build_time_axis()

        writer._create_netcdf(filtered, time_axis, time_values)

        # Verify file structure
        with nc.Dataset(output_file, 'r') as ds:
            # Check dimensions
            assert 'station' in ds.dimensions
            assert 'time' in ds.dimensions
            assert len(ds.dimensions['station']) == 1
            assert len(ds.dimensions['time']) == 10

            # Check variables exist
            assert 'wse' in ds.variables
            assert 'lat' in ds.variables
            assert 'lon' in ds.variables
            assert 'station_id' in ds.variables
            assert 'time' in ds.variables
            assert 'EGM08' in ds.variables

            # Check global attributes
            assert 'CF-1.8' in ds.Conventions


class TestNetCDFWriterTimeseries:
    """Test time series writing."""

    def test_write_station_timeseries(self, tmp_path):
        """Should write time series data to correct positions"""
        import numpy as np

        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        # Setup
        stations = StationList()
        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}
        s1.metadata = {'filepath': 'dummy.txt'}
        stations.add(s1)

        filtered = [s1]
        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf(filtered, time_axis, time_values)

        # Mock timeseries
        timeseries = [
            {'datetime': date(2020, 1, 2), 'elevation': 100.5},
            {'datetime': date(2020, 1, 4), 'elevation': 101.2},
        ]

        # Build time index
        time_index = {d: i for i, d in enumerate(time_axis)}

        # Open dataset and write timeseries
        with nc.Dataset(output_file, 'a') as ds:
            writer._write_station_timeseries(ds, 0, s1, timeseries, time_index)

        # Verify
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            # Index 0 (Jan 1) should be masked/NaN (no data)
            assert np.ma.is_masked(wse[0]) or np.isnan(wse[0])
            # Index 1 (Jan 2) should be 100.5
            assert wse[1] == pytest.approx(100.5)
            # Index 3 (Jan 4) should be 101.2
            assert wse[3] == pytest.approx(101.2)


class TestNetCDFWriterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_multiple_observations_same_day(self, tmp_path):
        """Last observation on same day should overwrite previous"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        # Setup
        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}

        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf([s1], time_axis, time_values)

        # Multiple observations on same day
        timeseries = [
            {'datetime': date(2020, 1, 2), 'elevation': 100.0},
            {'datetime': date(2020, 1, 2), 'elevation': 105.0},  # Same day, different value
            {'datetime': date(2020, 1, 2), 'elevation': 110.0},  # Last value should win
        ]

        time_index = {d: i for i, d in enumerate(time_axis)}

        with nc.Dataset(output_file, 'a') as ds:
            writer._write_station_timeseries(ds, 0, s1, timeseries, time_index)

        # Verify last value wins
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            assert wse[1] == pytest.approx(110.0)

    def test_datetime_with_time_component(self, tmp_path):
        """Datetime with time component should be truncated to date"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}

        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf([s1], time_axis, time_values)

        # Timeseries with datetime objects (not date)
        timeseries = [
            {'datetime': datetime(2020, 1, 2, 10, 30, 45), 'elevation': 100.5},
            {'datetime': datetime(2020, 1, 3, 15, 0, 0), 'elevation': 101.2},
        ]

        time_index = {d: i for i, d in enumerate(time_axis)}

        with nc.Dataset(output_file, 'a') as ds:
            writer._write_station_timeseries(ds, 0, s1, timeseries, time_index)

        # Verify data was written correctly
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            assert wse[1] == pytest.approx(100.5)  # Jan 2
            assert wse[2] == pytest.approx(101.2)  # Jan 3

    def test_empty_timeseries(self, tmp_path):
        """Empty timeseries should leave all values as NaN"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}

        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf([s1], time_axis, time_values)

        # Empty timeseries
        timeseries = []

        time_index = {d: i for i, d in enumerate(time_axis)}

        with nc.Dataset(output_file, 'a') as ds:
            writer._write_station_timeseries(ds, 0, s1, timeseries, time_index)

        # Verify all values are NaN/masked
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            for i in range(len(wse)):
                assert np.ma.is_masked(wse[i]) or np.isnan(wse[i])

    def test_observation_outside_time_range(self, tmp_path):
        """Observations outside time range should be skipped"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}

        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf([s1], time_axis, time_values)

        # Observations with some outside time range
        timeseries = [
            {'datetime': date(2019, 12, 31), 'elevation': 99.0},  # Before range
            {'datetime': date(2020, 1, 2), 'elevation': 100.5},   # In range
            {'datetime': date(2020, 1, 10), 'elevation': 102.0},  # After range
        ]

        time_index = {d: i for i, d in enumerate(time_axis)}

        with nc.Dataset(output_file, 'a') as ds:
            writer._write_station_timeseries(ds, 0, s1, timeseries, time_index)

        # Verify only in-range observation is written
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            assert wse[1] == pytest.approx(100.5)  # Jan 2
            # Other positions should be NaN
            assert np.ma.is_masked(wse[0]) or np.isnan(wse[0])

    def test_observation_with_none_elevation(self, tmp_path):
        """Observations with None elevation should be skipped"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}

        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf([s1], time_axis, time_values)

        # Observations with None elevation
        timeseries = [
            {'datetime': date(2020, 1, 2), 'elevation': None},
            {'datetime': date(2020, 1, 3), 'elevation': 101.2},
        ]

        time_index = {d: i for i, d in enumerate(time_axis)}

        with nc.Dataset(output_file, 'a') as ds:
            writer._write_station_timeseries(ds, 0, s1, timeseries, time_index)

        # Verify None elevation is skipped
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            assert np.ma.is_masked(wse[1]) or np.isnan(wse[1])  # Jan 2 - None
            assert wse[2] == pytest.approx(101.2)  # Jan 3

    def test_observation_with_none_datetime(self, tmp_path):
        """Observations with None datetime should be skipped"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}

        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf([s1], time_axis, time_values)

        # Observations with None datetime
        timeseries = [
            {'datetime': None, 'elevation': 100.0},
            {'datetime': date(2020, 1, 3), 'elevation': 101.2},
        ]

        time_index = {d: i for i, d in enumerate(time_axis)}

        with nc.Dataset(output_file, 'a') as ds:
            writer._write_station_timeseries(ds, 0, s1, timeseries, time_index)

        # Verify None datetime is skipped, valid one is written
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            assert wse[2] == pytest.approx(101.2)  # Jan 3
