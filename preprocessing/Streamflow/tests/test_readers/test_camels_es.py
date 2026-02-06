"""Tests for CAMELS-ES reader (269 Spanish catchments, mm/d -> m3/s)."""

from pathlib import Path

import numpy as np
import pytest

DATA_PATH = Path("/Users/zhongwangwei/Downloads/Streamflow/CAMELS-ES")

# Import the reader module to trigger registration
from src.readers.camels_es_reader import CAMELSESReader, CAMELSES_PREFIX


def _make_config(max_stations=None):
    """Build a minimal config dict for CAMELSESReader.read_all."""
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


class TestCAMELSESConstants:
    """Verify constants and class attributes."""

    def test_prefix(self):
        assert CAMELSES_PREFIX == "camelses"

    def test_source_name(self):
        reader = CAMELSESReader()
        assert reader.source_name == "camels_es"

    def test_reader_registered(self):
        from src.readers import READERS
        assert "camels_es" in READERS
        assert READERS["camels_es"] is CAMELSESReader


# ---------------------------------------------------------------------------
# Integration tests (skip when data is not available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DATA_PATH.exists(), reason="CAMELS-ES data not available")
class TestCAMELSESIntegration:
    """Integration tests that require the actual CAMELS-ES data on disk."""

    def test_read_returns_one_dataset(self):
        """read_all should return a list with exactly 1 StationDataset (daily)."""
        reader = CAMELSESReader()
        config = _make_config(max_stations=3)
        datasets = reader.read_all(config)

        assert isinstance(datasets, list)
        assert len(datasets) == 1

        ds = datasets[0]
        assert ds.time_resolution == "daily"
        assert ds.source_name == "camels_es"
        assert ds.station_count > 0

    def test_station_has_valid_coords(self):
        """Coordinates should be valid WGS84 and area > 0."""
        reader = CAMELSESReader()
        config = _make_config(max_stations=5)
        datasets = reader.read_all(config)

        ds = datasets[0]
        for stn in ds.stations:
            assert -90.0 <= stn.latitude <= 90.0, (
                f"Latitude out of range for {stn.station_id}: {stn.latitude}"
            )
            assert -180.0 <= stn.longitude <= 180.0, (
                f"Longitude out of range for {stn.station_id}: {stn.longitude}"
            )
            assert stn.upstream_area > 0, (
                f"Area should be > 0 for {stn.station_id}: {stn.upstream_area}"
            )

    def test_discharge_units_m3s(self):
        """Source unit in metadata is 'mm/d'; discharge values should be in m3/s."""
        reader = CAMELSESReader()
        config = _make_config(max_stations=3)
        datasets = reader.read_all(config)

        ds = datasets[0]
        # Metadata should record original source unit
        for sid, meta in ds.metadata.items():
            assert meta.source_unit == "mm/d", (
                f"Expected source_unit 'mm/d', got '{meta.source_unit}' for {sid}"
            )

        # Discharge values: after mm/d -> m3/s conversion, values should be finite.
        for stn in ds.stations:
            valid = stn.discharge[~np.isnan(stn.discharge)]
            if len(valid) > 0:
                assert valid.max() < 1e7, (
                    f"Discharge suspiciously large for {stn.station_id}"
                )
                assert valid.dtype == np.float64 or valid.dtype == np.float32

    def test_timezone_fixed_offset_cet(self):
        """CAMELS-ES uses CET (UTC+1) fixed offset."""
        reader = CAMELSESReader()
        config = _make_config(max_stations=1)
        datasets = reader.read_all(config)

        ds = datasets[0]
        assert ds.timezone_type == "fixed_offset"
        assert ds.timezone_utc_offset == 1.0
        assert "CET" in ds.timezone_definition

    def test_max_stations_option(self):
        """The max_stations option should cap the number of stations returned."""
        reader = CAMELSESReader()
        config = _make_config(max_stations=2)
        datasets = reader.read_all(config)

        ds = datasets[0]
        assert ds.station_count <= 2

    def test_station_ids_have_camelses_prefix(self):
        """All station IDs should start with 'camelses_'."""
        reader = CAMELSESReader()
        config = _make_config(max_stations=5)
        datasets = reader.read_all(config)

        ds = datasets[0]
        for stn in ds.stations:
            assert stn.station_id.startswith("camelses_"), (
                f"Unexpected station_id format: {stn.station_id}"
            )

    def test_metadata_country_is_spain(self):
        """All stations should be in Spain."""
        reader = CAMELSESReader()
        config = _make_config(max_stations=3)
        datasets = reader.read_all(config)

        ds = datasets[0]
        for sid, meta in ds.metadata.items():
            assert meta.country == "Spain", (
                f"Expected country 'Spain', got '{meta.country}' for {sid}"
            )
