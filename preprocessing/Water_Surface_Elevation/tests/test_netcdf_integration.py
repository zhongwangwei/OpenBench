# tests/test_netcdf_integration.py
"""Integration test for NetCDF output."""
import pytest
from pathlib import Path
import netCDF4 as nc

from src.core.station import Station, StationList
from src.writers.netcdf_writer import NetCDFWriter


class TestNetCDFIntegration:
    """End-to-end NetCDF output test."""

    def test_full_write_workflow(self, tmp_path):
        """Test complete workflow: filter -> create -> write"""
        # Setup
        output_file = tmp_path / "test_wse.nc"

        config = {
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-10',
            'min_uparea': 100.0,
        }

        writer = NetCDFWriter(config)

        # Create test stations
        stations = StationList()

        # Station 1: passes filter
        s1 = Station(id='HW001', name='Amazon_Station', lon=-60.0, lat=-3.5,
                    source='hydroweb', elevation=50.0, num_observations=100)
        s1.egm08 = 20.0
        s1.egm96 = 19.5
        s1.cama_results = {
            'glb_03min': {'flag': 20, 'uparea': 500000.0, 'lon_cama': -60.1, 'lat_cama': -3.6},
            'glb_15min': {'flag': 20, 'uparea': 490000.0, 'lon_cama': -60.0, 'lat_cama': -3.5},
        }
        s1.metadata = {'filepath': '/mock/path.txt'}
        stations.add(s1)

        # Station 2: fails filter (uparea < 100)
        s2 = Station(id='HW002', name='Small_Stream', lon=-61.0, lat=-4.0,
                    source='hydroweb', elevation=100.0, num_observations=50)
        s2.cama_results = {'glb_03min': {'flag': 20, 'uparea': 50.0}}
        stations.add(s2)

        # Mock data_paths (won't be used since filepath doesn't exist)
        data_paths = {'hydroweb': '/mock/data'}

        # Run
        result = writer.write(stations, data_paths)

        # Verify
        assert result == output_file
        assert output_file.exists()

        with nc.Dataset(output_file, 'r') as ds:
            # Only 1 station should pass
            assert len(ds.dimensions['station']) == 1

            # Check metadata
            assert ds.variables['station_id'][0] == 'HW001'
            assert ds.variables['lat'][0] == pytest.approx(-3.5)
            assert ds.variables['lon'][0] == pytest.approx(-60.0)
            assert ds.variables['EGM08'][0] == pytest.approx(20.0)

            # Check CaMa results
            assert ds.variables['cama_uparea_03min'][0] == pytest.approx(500000.0)

            # Check time dimension
            assert len(ds.dimensions['time']) == 10

            # Check global attributes
            assert 'CF-1.8' in ds.Conventions
