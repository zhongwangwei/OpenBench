"""Tests for the GSHA reader (monthly consolidated NetCDF, ~21k stations)."""

import pytest
import numpy as np
from pathlib import Path

from src.readers.gsha_reader import GSHAReader, _FILL_VALUE
from src.models import StationDataset

DATA_PATH = Path("/Volumes/Data01/StreamFlow-zip/GSHA")

skip_no_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="GSHA data not available"
)


def _make_config(max_stations=None):
    """Build a minimal config dict for GSHAReader.read_all."""
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


class TestGSHAReaderUnit:
    """Unit tests that do not require data on disk."""

    def test_source_name(self):
        reader = GSHAReader()
        assert reader.source_name == "gsha"

    def test_fill_value_constant(self):
        assert _FILL_VALUE == -9999.0

    def test_read_all_missing_directory(self, tmp_path):
        """Reader should return [] when the data directory does not exist."""
        reader = GSHAReader()
        config = {"source": {"path": str(tmp_path / "nonexistent")}, "options": {}}
        result = reader.read_all(config)
        assert result == []

    def test_read_all_missing_file(self, tmp_path):
        """Reader should return [] when directory exists but file is absent."""
        reader = GSHAReader()
        config = {"source": {"path": str(tmp_path)}, "options": {}}
        result = reader.read_all(config)
        assert result == []


# ---------------------------------------------------------------------------
# Integration tests (skip when data is not available)
# ---------------------------------------------------------------------------


@skip_no_data
class TestGSHAIntegration:
    """Integration tests against real GSHA data (skip if absent)."""

    @pytest.fixture(scope="class")
    def datasets(self):
        """Read GSHA data once (max_stations=10) and share across tests."""
        reader = GSHAReader()
        return reader.read_all(_make_config(max_stations=10))

    def test_read_returns_monthly(self, datasets):
        """Reader must return exactly one StationDataset with resolution 'monthly'."""
        assert len(datasets) == 1
        ds = datasets[0]
        assert isinstance(ds, StationDataset)
        assert ds.time_resolution == "monthly"

    def test_source_name_is_gsha(self, datasets):
        for ds in datasets:
            assert ds.source_name == "gsha"

    def test_timezone_is_utc(self, datasets):
        ds = datasets[0]
        assert ds.timezone_type == "utc"
        assert ds.timezone_utc_offset == 0.0

    def test_station_count_respects_max(self, datasets):
        ds = datasets[0]
        assert ds.station_count <= 10

    def test_fill_value_handled(self, datasets):
        """No -9999.0 sentinel values should remain in discharge arrays."""
        for ds in datasets:
            for station in ds.stations:
                assert not np.any(
                    station.discharge == _FILL_VALUE
                ), f"Station {station.station_id} still contains -9999.0"

    def test_discharge_dtype_float64(self, datasets):
        """Discharge arrays must be float64."""
        for ds in datasets:
            for station in ds.stations:
                assert station.discharge.dtype == np.float64, (
                    f"{station.station_id}: dtype={station.discharge.dtype}"
                )

    def test_stations_have_coordinates(self, datasets):
        """Every station must have valid lat/lon (or NaN, but not -9999)."""
        for ds in datasets:
            for station in ds.stations:
                # Should not be fill value
                assert station.latitude != _FILL_VALUE
                assert station.longitude != _FILL_VALUE
                # If not NaN, must be in valid range
                if not np.isnan(station.latitude):
                    assert -90 <= station.latitude <= 90, (
                        f"{station.station_id}: lat={station.latitude}"
                    )
                if not np.isnan(station.longitude):
                    assert -180 <= station.longitude <= 180, (
                        f"{station.station_id}: lon={station.longitude}"
                    )

    def test_metadata_keys_match_stations(self, datasets):
        """Metadata dict keys must match station IDs."""
        for ds in datasets:
            station_ids = {s.station_id for s in ds.stations}
            meta_ids = set(ds.metadata.keys())
            assert station_ids == meta_ids

    def test_units_already_m3s(self, datasets):
        """Metadata source_unit must be 'm3/s' for every station."""
        for ds in datasets:
            for sid, meta in ds.metadata.items():
                assert meta.source_unit == "m3/s", (
                    f"Station {sid}: expected 'm3/s', got '{meta.source_unit}'"
                )

    def test_station_ids_are_strings(self, datasets):
        """Station IDs should be non-empty strings from the dataset."""
        for ds in datasets:
            for station in ds.stations:
                assert isinstance(station.station_id, str)
                assert len(station.station_id) > 0

    def test_time_array_length(self, datasets):
        """All stations should share the same time dimension."""
        ds = datasets[0]
        if ds.stations:
            expected_len = len(ds.stations[0].time)
            for station in ds.stations:
                assert len(station.time) == expected_len, (
                    f"{station.station_id}: time length {len(station.time)} != {expected_len}"
                )
