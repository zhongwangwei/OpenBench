"""Tests for the CAMELS-DK reader (EPSG:25832 reprojection, mm/d -> m3/s)."""

import pytest
import numpy as np
from pathlib import Path

from src.readers.camels_dk_reader import CamelsDKReader, SOURCE_CRS
from src.models import StationDataset

DATA_PATH = Path("/Users/zhongwangwei/Downloads/Streamflow/CAMELS-DK")

skip_no_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="CAMELS-DK data not available"
)


def _make_config(max_stations=None):
    """Build a minimal config dict for CamelsDKReader.read_all."""
    cfg = {
        "source": {"path": str(DATA_PATH)},
        "options": {},
    }
    if max_stations is not None:
        cfg["options"]["max_stations"] = max_stations
    return cfg


# ---------------------------------------------------------------------------
# Unit / logic tests (always run)
# ---------------------------------------------------------------------------


class TestCamelsDKConstants:
    """Verify reader constants and class attributes."""

    def test_source_crs_is_epsg25832(self):
        assert SOURCE_CRS == "EPSG:25832"

    def test_source_name(self):
        reader = CamelsDKReader()
        assert reader.source_name == "camels_dk"


# ---------------------------------------------------------------------------
# Integration tests (skip when data is not available)
# ---------------------------------------------------------------------------


@skip_no_data
class TestCamelsDKIntegration:
    """Integration tests that require the actual CAMELS-DK data on disk."""

    @pytest.fixture(scope="class")
    def datasets(self):
        """Read CAMELS-DK data once (max_stations=5) and share across tests."""
        reader = CamelsDKReader()
        return reader.read_all(_make_config(max_stations=5))

    def test_read_returns_one_dataset(self, datasets):
        """read_all should return a list with exactly 1 StationDataset."""
        assert isinstance(datasets, list)
        assert len(datasets) == 1

    def test_time_resolution_is_daily(self, datasets):
        ds = datasets[0]
        assert ds.time_resolution == "daily"

    def test_source_name_is_camels_dk(self, datasets):
        ds = datasets[0]
        assert ds.source_name == "camels_dk"

    def test_station_count(self, datasets):
        ds = datasets[0]
        assert ds.station_count > 0
        assert ds.station_count <= 5  # limited by max_stations

    def test_coordinates_are_wgs84(self, datasets):
        """After EPSG:25832 reprojection, coordinates must be valid WGS84."""
        ds = datasets[0]
        for stn in ds.stations:
            assert -90.0 <= stn.latitude <= 90.0, (
                f"Latitude out of range for {stn.station_id}: {stn.latitude}"
            )
            assert -180.0 <= stn.longitude <= 180.0, (
                f"Longitude out of range for {stn.station_id}: {stn.longitude}"
            )
            # Denmark roughly: lat 54-58, lon 8-16
            assert 53.0 <= stn.latitude <= 59.0, (
                f"Latitude not in Denmark range for {stn.station_id}: {stn.latitude}"
            )
            assert 7.0 <= stn.longitude <= 16.0, (
                f"Longitude not in Denmark range for {stn.station_id}: {stn.longitude}"
            )

    def test_area_converted_to_km2(self, datasets):
        """Area must be positive and in km2 (not m2)."""
        ds = datasets[0]
        for stn in ds.stations:
            assert stn.upstream_area > 0, (
                f"Area should be > 0 for {stn.station_id}"
            )
            # CAMELS-DK areas range ~0.02 to ~1100 km2;
            # if still in m2 they would be > 1e6
            assert stn.upstream_area < 50000, (
                f"Area suspiciously large (still in m2?) for {stn.station_id}: "
                f"{stn.upstream_area}"
            )

    def test_discharge_units_converted_to_m3s(self, datasets):
        """Discharge should be converted from mm/d to m3/s."""
        ds = datasets[0]
        for stn in ds.stations:
            valid = stn.discharge[~np.isnan(stn.discharge)]
            if len(valid) > 0:
                assert valid.dtype == np.float64
                # After mm/d -> m3/s conversion, values should be reasonable
                assert valid.max() < 1e7, (
                    f"Discharge suspiciously large for {stn.station_id}"
                )

    def test_metadata_source_unit_is_mmd(self, datasets):
        """Metadata should record the original source unit as mm/d."""
        ds = datasets[0]
        for sid, meta in ds.metadata.items():
            assert meta.source_unit == "mm/d", (
                f"Expected source_unit 'mm/d', got '{meta.source_unit}' for {sid}"
            )

    def test_metadata_source_crs_is_epsg25832(self, datasets):
        """Metadata should record the original source CRS."""
        ds = datasets[0]
        for sid, meta in ds.metadata.items():
            assert meta.source_crs == "EPSG:25832", (
                f"Expected source_crs 'EPSG:25832', got '{meta.source_crs}' for {sid}"
            )

    def test_metadata_country_is_dk(self, datasets):
        ds = datasets[0]
        for sid, meta in ds.metadata.items():
            assert meta.country == "DK"

    def test_timezone_settings(self, datasets):
        ds = datasets[0]
        assert ds.timezone_type == "fixed_offset"
        assert ds.timezone_utc_offset == 1.0
        assert ds.timezone_definition == "CET (UTC+1)"

    def test_metadata_keys_match_stations(self, datasets):
        """Metadata dict keys must match station IDs."""
        ds = datasets[0]
        station_ids = {s.station_id for s in ds.stations}
        meta_ids = set(ds.metadata.keys())
        assert station_ids == meta_ids

    def test_all_304_gauged_stations(self):
        """Without max_stations, all 304 gauged stations should be loaded."""
        reader = CamelsDKReader()
        datasets = reader.read_all(_make_config(max_stations=None))
        ds = datasets[0]
        assert ds.station_count == 304, (
            f"Expected 304 gauged stations, got {ds.station_count}"
        )
