"""Tests for Caravan-core reader (5 sub-datasets, mm/d -> m3/s)."""

from pathlib import Path

import numpy as np
import pytest

DATA_PATH = Path("/Users/zhongwangwei/Downloads/Streamflow/Caravan-core")

# Import the reader module to trigger registration, and the constant list
from src.readers.caravan_core_reader import CaravanCoreReader, CARAVAN_SUBDATASETS


def _make_config(max_stations=None, subdatasets=None):
    """Build a minimal config dict for CaravanCoreReader.read_all."""
    cfg = {
        "source": {"path": str(DATA_PATH)},
        "options": {},
    }
    if max_stations is not None:
        cfg["options"]["max_stations"] = max_stations
    if subdatasets is not None:
        cfg["options"]["subdatasets"] = subdatasets
    return cfg


# ---------------------------------------------------------------------------
# Unit / logic tests (always run)
# ---------------------------------------------------------------------------


class TestCaravanSubdatasets:
    """Verify the allowed subdataset list."""

    def test_excludes_camelsbr_camelsgb(self):
        """Only 5 sub-datasets; camelsbr and camelsgb must NOT be present."""
        assert "camelsbr" not in CARAVAN_SUBDATASETS
        assert "camelsgb" not in CARAVAN_SUBDATASETS

    def test_includes_expected_subdatasets(self):
        expected = {"camels", "camelsaus", "camelscl", "hysets", "lamah"}
        assert set(CARAVAN_SUBDATASETS) == expected

    def test_subdataset_count(self):
        assert len(CARAVAN_SUBDATASETS) == 5


# ---------------------------------------------------------------------------
# Integration tests (skip when data is not available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Caravan-core data not available")
class TestCaravanCoreIntegration:
    """Integration tests that require the actual Caravan-core data on disk."""

    def test_read_returns_datasets(self):
        """read_all should return a list with at least 1 StationDataset, daily."""
        reader = CaravanCoreReader()
        config = _make_config(max_stations=2, subdatasets=["camels"])
        datasets = reader.read_all(config)

        assert isinstance(datasets, list)
        assert len(datasets) >= 1

        ds = datasets[0]
        assert ds.time_resolution == "daily"
        assert ds.source_name == "caravan_core"
        assert ds.station_count > 0

    def test_station_has_valid_coords(self):
        """Coordinates should be valid WGS84 and area > 0."""
        reader = CaravanCoreReader()
        config = _make_config(max_stations=5, subdatasets=["camels"])
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
        reader = CaravanCoreReader()
        config = _make_config(max_stations=3, subdatasets=["camels"])
        datasets = reader.read_all(config)

        ds = datasets[0]
        # Metadata should record original source unit
        for sid, meta in ds.metadata.items():
            assert meta.source_unit == "mm/d", (
                f"Expected source_unit 'mm/d', got '{meta.source_unit}' for {sid}"
            )

        # Discharge values: typical rivers in m3/s range ~0.001 to ~100000.
        # mm/d values are typically < 50, but after conversion with area they
        # should be larger. We just verify they are finite floats.
        for stn in ds.stations:
            valid = stn.discharge[~np.isnan(stn.discharge)]
            if len(valid) > 0:
                assert valid.max() < 1e7, (
                    f"Discharge suspiciously large for {stn.station_id}"
                )
                # With area conversion, most values should exceed pure mm/d range
                # (though very small catchments could yield small m3/s values)
                assert valid.dtype == np.float64 or valid.dtype == np.float32

    def test_timezone_type_is_local(self):
        """Caravan-core stores per-station local timezone info."""
        reader = CaravanCoreReader()
        config = _make_config(max_stations=1, subdatasets=["camels"])
        datasets = reader.read_all(config)

        ds = datasets[0]
        assert ds.timezone_type == "local"

    def test_multiple_subdatasets(self):
        """Reading from two subdatasets should combine stations."""
        reader = CaravanCoreReader()
        config = _make_config(max_stations=3, subdatasets=["camels", "camelsaus"])
        datasets = reader.read_all(config)

        ds = datasets[0]
        # Should have stations from both (up to max_stations total)
        assert ds.station_count > 0
