"""Tests for the GRDC reader."""

import pytest
import numpy as np
from pathlib import Path

from src.readers.grdc_reader import GRDCReader
from src.models import StationDataset

DATA_PATH = Path("/Users/zhongwangwei/Downloads/Streamflow/GRDC")

skip_no_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="GRDC data not available"
)


def _make_config(max_zips=1):
    """Build a minimal config dict for the reader."""
    return {
        "source": {"path": str(DATA_PATH)},
        "options": {"max_zips": max_zips},
    }


@skip_no_data
class TestGRDCReader:
    """Integration tests against real GRDC data (skip if absent)."""

    @pytest.fixture(scope="class")
    def datasets(self):
        """Read GRDC data once (max_zips=1) and share across tests."""
        reader = GRDCReader()
        return reader.read_all(_make_config(max_zips=1))

    # ------------------------------------------------------------------
    # 1. test_read_returns_daily
    # ------------------------------------------------------------------
    def test_read_returns_daily(self, datasets):
        """Reader must return at least one StationDataset with resolution 'daily'."""
        assert len(datasets) >= 1
        resolutions = [ds.time_resolution for ds in datasets]
        assert "daily" in resolutions, f"Expected 'daily' in {resolutions}"

    # ------------------------------------------------------------------
    # 2. test_fill_value_handled
    # ------------------------------------------------------------------
    def test_fill_value_handled(self, datasets):
        """No -999.0 sentinel values should remain in discharge arrays."""
        for ds in datasets:
            for station in ds.stations:
                assert not np.any(
                    station.discharge == -999.0
                ), f"Station {station.station_id} still contains -999.0"

    # ------------------------------------------------------------------
    # 3. test_station_ids_prefixed
    # ------------------------------------------------------------------
    def test_station_ids_prefixed(self, datasets):
        """All station IDs must start with 'GRDC_'."""
        for ds in datasets:
            for station in ds.stations:
                assert station.station_id.startswith(
                    "GRDC_"
                ), f"Station ID {station.station_id} missing GRDC_ prefix"

    # ------------------------------------------------------------------
    # 4. test_units_already_m3s
    # ------------------------------------------------------------------
    def test_units_already_m3s(self, datasets):
        """Metadata source_unit must be 'm3/s' for every station."""
        for ds in datasets:
            for sid, meta in ds.metadata.items():
                assert (
                    meta.source_unit == "m3/s"
                ), f"Station {sid}: expected 'm3/s', got '{meta.source_unit}'"

    # ------------------------------------------------------------------
    # Extra: basic sanity checks
    # ------------------------------------------------------------------
    def test_stations_have_coordinates(self, datasets):
        """Every station must have valid lat/lon."""
        for ds in datasets:
            for station in ds.stations:
                assert -90 <= station.latitude <= 90, (
                    f"{station.station_id}: lat={station.latitude}"
                )
                assert -180 <= station.longitude <= 180, (
                    f"{station.station_id}: lon={station.longitude}"
                )

    def test_discharge_dtype_float(self, datasets):
        """Discharge arrays must be float64."""
        for ds in datasets:
            for station in ds.stations:
                assert station.discharge.dtype == np.float64, (
                    f"{station.station_id}: dtype={station.discharge.dtype}"
                )

    def test_metadata_keys_match_stations(self, datasets):
        """Metadata dict keys must match station IDs."""
        for ds in datasets:
            station_ids = {s.station_id for s in ds.stations}
            meta_ids = set(ds.metadata.keys())
            assert station_ids == meta_ids

    def test_source_name_is_grdc(self, datasets):
        """source_name on every dataset must be 'grdc'."""
        for ds in datasets:
            assert ds.source_name == "grdc"
