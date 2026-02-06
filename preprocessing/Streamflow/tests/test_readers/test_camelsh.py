"""Tests for the CAMELSH reader (hourly, per-station NC in ZIP)."""

from pathlib import Path

import numpy as np
import pytest

from src.readers.camelsh_reader import CAMELSHReader
from src.models import StationDataset

DATA_PATH = Path("/Volumes/Data01/StreamFlow-zip/CAMELSH")

skip_no_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="CAMELSH data not available"
)


def _make_config(max_stations=None):
    """Build a minimal config dict for CAMELSHReader.read_all."""
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


class TestCAMELSHReaderUnit:
    """Unit tests that do not need real data on disk."""

    def test_source_name(self):
        reader = CAMELSHReader()
        assert reader.source_name == "camelsh"

    def test_reader_registered(self):
        from src.readers import READERS
        assert "camelsh" in READERS

    def test_read_all_missing_path_returns_empty(self, tmp_path):
        """When the data path does not exist, read_all returns []."""
        reader = CAMELSHReader()
        config = {
            "source": {"path": str(tmp_path / "nonexistent")},
            "options": {},
        }
        result = reader.read_all(config)
        assert result == []

    def test_read_all_no_zip_returns_empty(self, tmp_path):
        """When attributes CSV exists but ZIP does not, returns []."""
        attrs = tmp_path / "attributes_gageii_BasinID.csv"
        attrs.write_text("STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE\n")
        reader = CAMELSHReader()
        config = {
            "source": {"path": str(tmp_path)},
            "options": {},
        }
        result = reader.read_all(config)
        assert result == []


# ---------------------------------------------------------------------------
# Integration tests (skip when data is not available)
# ---------------------------------------------------------------------------


@skip_no_data
class TestCAMELSHIntegration:
    """Integration tests against real CAMELSH data (skip if absent)."""

    @pytest.fixture(scope="class")
    def datasets(self):
        """Read CAMELSH data once (max_stations=5) and share across tests."""
        reader = CAMELSHReader()
        return reader.read_all(_make_config(max_stations=5))

    def test_read_returns_one_dataset(self, datasets):
        """Reader must return exactly 1 StationDataset with resolution 'hourly'."""
        assert len(datasets) == 1
        ds = datasets[0]
        assert isinstance(ds, StationDataset)
        assert ds.time_resolution == "hourly"

    def test_source_name_is_camelsh(self, datasets):
        """source_name on the dataset must be 'camelsh'."""
        assert datasets[0].source_name == "camelsh"

    def test_station_count(self, datasets):
        """Should have loaded the requested number of stations."""
        ds = datasets[0]
        assert ds.station_count > 0
        assert ds.station_count <= 5

    def test_station_ids_are_8digit(self, datasets):
        """Station IDs should be 8-digit zero-padded strings."""
        for stn in datasets[0].stations:
            assert len(stn.station_id) == 8, (
                f"Expected 8-digit ID, got '{stn.station_id}'"
            )
            assert stn.station_id.isdigit(), (
                f"Station ID should be numeric: '{stn.station_id}'"
            )

    def test_stations_have_valid_coordinates(self, datasets):
        """Every station must have valid lat/lon."""
        for stn in datasets[0].stations:
            assert -90.0 <= stn.latitude <= 90.0, (
                f"{stn.station_id}: lat={stn.latitude}"
            )
            assert -180.0 <= stn.longitude <= 180.0, (
                f"{stn.station_id}: lon={stn.longitude}"
            )

    def test_upstream_area_positive(self, datasets):
        """Upstream area should be positive."""
        for stn in datasets[0].stations:
            assert stn.upstream_area > 0, (
                f"{stn.station_id}: area={stn.upstream_area}"
            )

    def test_discharge_dtype_float64(self, datasets):
        """Discharge arrays must be float64."""
        for stn in datasets[0].stations:
            assert stn.discharge.dtype == np.float64, (
                f"{stn.station_id}: dtype={stn.discharge.dtype}"
            )

    def test_no_inf_in_discharge(self, datasets):
        """No inf values should remain in discharge arrays."""
        for stn in datasets[0].stations:
            finite_or_nan = np.isnan(stn.discharge) | np.isfinite(stn.discharge)
            assert np.all(finite_or_nan), (
                f"{stn.station_id} contains non-finite values"
            )

    def test_units_already_m3s(self, datasets):
        """Metadata source_unit must be 'm3/s' for every station."""
        ds = datasets[0]
        for sid, meta in ds.metadata.items():
            assert meta.source_unit == "m3/s", (
                f"Station {sid}: expected 'm3/s', got '{meta.source_unit}'"
            )

    def test_metadata_keys_match_stations(self, datasets):
        """Metadata dict keys must match station IDs."""
        ds = datasets[0]
        station_ids = {s.station_id for s in ds.stations}
        meta_ids = set(ds.metadata.keys())
        assert station_ids == meta_ids

    def test_timezone_type_is_local(self, datasets):
        """timezone_type should be 'local'."""
        assert datasets[0].timezone_type == "local"

    def test_time_array_not_empty(self, datasets):
        """Every station must have at least some time values."""
        for stn in datasets[0].stations:
            assert len(stn.time) > 0, (
                f"{stn.station_id}: empty time array"
            )
