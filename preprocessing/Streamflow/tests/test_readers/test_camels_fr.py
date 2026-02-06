"""Tests for the CAMELS_FR reader (semicolon CSV, L/s, hydro day)."""

from pathlib import Path

import numpy as np
import pytest

from src.readers.camels_fr_reader import CAMELSFRReader
from src.constants import DISCHARGE_CONVERSIONS
from src.utils.unit_converter import convert_mmd_to_m3s
from src.models import StationDataset

DATA_PATH = Path("/Users/zhongwangwei/Downloads/Streamflow/CAMELS_FR")

skip_no_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="CAMELS_FR data not available"
)


def _make_config(max_stations=None, resolutions=None):
    """Build a minimal config dict for CAMELSFRReader.read_all."""
    cfg = {
        "source": {"path": str(DATA_PATH)},
        "options": {},
    }
    if max_stations is not None:
        cfg["options"]["max_stations"] = max_stations
    if resolutions is not None:
        cfg["options"]["resolutions"] = resolutions
    return cfg


# ---------------------------------------------------------------------------
# Unit / logic tests (always run -- no real data needed)
# ---------------------------------------------------------------------------


class TestLsToM3sConversion:
    """Verify the L/s -> m3/s conversion factor and arithmetic."""

    def test_conversion_factor_exists(self):
        """DISCHARGE_CONVERSIONS must contain 'L/s' = 0.001."""
        assert "L/s" in DISCHARGE_CONVERSIONS
        assert DISCHARGE_CONVERSIONS["L/s"] == pytest.approx(0.001)

    def test_scalar_conversion(self):
        """1000 L/s should equal 1.0 m3/s."""
        ls_value = 1000.0
        m3s_value = ls_value * DISCHARGE_CONVERSIONS["L/s"]
        assert m3s_value == pytest.approx(1.0)

    def test_array_conversion(self):
        """Array conversion: [500, 2000, 0] L/s -> [0.5, 2.0, 0.0] m3/s."""
        ls_arr = np.array([500.0, 2000.0, 0.0])
        m3s_arr = ls_arr * DISCHARGE_CONVERSIONS["L/s"]
        expected = np.array([0.5, 2.0, 0.0])
        np.testing.assert_allclose(m3s_arr, expected)

    def test_nan_propagation(self):
        """NaN values must propagate through the conversion."""
        ls_arr = np.array([100.0, np.nan, 300.0])
        m3s_arr = ls_arr * DISCHARGE_CONVERSIONS["L/s"]
        assert np.isnan(m3s_arr[1])
        assert m3s_arr[0] == pytest.approx(0.1)
        assert m3s_arr[2] == pytest.approx(0.3)


class TestMonthlyMmConversion:
    """Verify the mm (monthly total) -> m3/s conversion logic."""

    def test_monthly_mm_to_m3s(self):
        """100 mm over 30 days in a 100 km2 catchment -> expected m3/s."""
        mm_total = 100.0
        days = 30
        area_km2 = 100.0

        mm_per_day = mm_total / days
        m3s = convert_mmd_to_m3s(mm_per_day, area_km2)

        # mm/d -> m3/s: value * area_km2 * 1000 / 86400
        expected = (100.0 / 30) * 100.0 * 1000.0 / 86400.0
        assert m3s == pytest.approx(expected)

    def test_monthly_mm_zero(self):
        """0 mm should yield 0 m3/s."""
        m3s = convert_mmd_to_m3s(0.0, 500.0)
        assert m3s == pytest.approx(0.0)


class TestReaderRegistration:
    """Verify the reader registers correctly."""

    def test_registered_name(self):
        from src.readers import READERS
        assert "camels_fr" in READERS

    def test_source_name(self):
        reader = CAMELSFRReader()
        assert reader.source_name == "camels_fr"


# ---------------------------------------------------------------------------
# Integration tests (skip when data is not available)
# ---------------------------------------------------------------------------


@skip_no_data
class TestCAMELSFRIntegrationDaily:
    """Integration tests for daily resolution against real CAMELS_FR data."""

    @pytest.fixture(scope="class")
    def datasets(self):
        reader = CAMELSFRReader()
        return reader.read_all(_make_config(max_stations=3, resolutions=["daily"]))

    def test_returns_daily_dataset(self, datasets):
        """read_all with resolutions=['daily'] must return one StationDataset."""
        assert len(datasets) == 1
        assert datasets[0].time_resolution == "daily"
        assert datasets[0].source_name == "camels_fr"

    def test_station_count(self, datasets):
        ds = datasets[0]
        assert ds.station_count > 0
        assert ds.station_count <= 3

    def test_valid_coordinates(self, datasets):
        ds = datasets[0]
        for stn in ds.stations:
            assert -90.0 <= stn.latitude <= 90.0, (
                f"lat out of range: {stn.station_id}"
            )
            assert -180.0 <= stn.longitude <= 180.0, (
                f"lon out of range: {stn.station_id}"
            )
            assert stn.upstream_area > 0, (
                f"area must be > 0: {stn.station_id}"
            )

    def test_discharge_dtype_float64(self, datasets):
        ds = datasets[0]
        for stn in ds.stations:
            assert stn.discharge.dtype == np.float64

    def test_source_unit_ls(self, datasets):
        """Metadata should record original source unit as L/s."""
        ds = datasets[0]
        for sid, meta in ds.metadata.items():
            assert meta.source_unit == "L/s", (
                f"Expected 'L/s', got '{meta.source_unit}' for {sid}"
            )

    def test_discharge_plausible(self, datasets):
        """Discharge values (m3/s) should be plausible for French rivers."""
        ds = datasets[0]
        for stn in ds.stations:
            valid = stn.discharge[~np.isnan(stn.discharge)]
            if len(valid) > 0:
                assert valid.max() < 1e6, (
                    f"Discharge suspiciously large: {stn.station_id}"
                )

    def test_timezone_hydrological_day(self, datasets):
        ds = datasets[0]
        assert ds.timezone_type == "hydrological_day"
        assert ds.timezone_utc_offset == 1.0

    def test_metadata_keys_match_stations(self, datasets):
        ds = datasets[0]
        station_ids = {s.station_id for s in ds.stations}
        meta_ids = set(ds.metadata.keys())
        assert station_ids == meta_ids


@skip_no_data
class TestCAMELSFRIntegrationMonthly:
    """Integration tests for monthly resolution against real CAMELS_FR data."""

    @pytest.fixture(scope="class")
    def datasets(self):
        reader = CAMELSFRReader()
        return reader.read_all(_make_config(max_stations=3, resolutions=["monthly"]))

    def test_returns_monthly_dataset(self, datasets):
        assert len(datasets) == 1
        assert datasets[0].time_resolution == "monthly"

    def test_station_count(self, datasets):
        ds = datasets[0]
        assert ds.station_count > 0
        assert ds.station_count <= 3

    def test_source_unit_mm(self, datasets):
        ds = datasets[0]
        for sid, meta in ds.metadata.items():
            assert meta.source_unit == "mm", (
                f"Expected 'mm', got '{meta.source_unit}' for {sid}"
            )

    def test_discharge_dtype_float64(self, datasets):
        ds = datasets[0]
        for stn in ds.stations:
            assert stn.discharge.dtype == np.float64

    def test_discharge_plausible(self, datasets):
        ds = datasets[0]
        for stn in ds.stations:
            valid = stn.discharge[~np.isnan(stn.discharge)]
            if len(valid) > 0:
                assert valid.max() < 1e6, (
                    f"Discharge suspiciously large: {stn.station_id}"
                )


@skip_no_data
class TestCAMELSFRIntegrationBoth:
    """Integration tests reading both resolutions at once."""

    @pytest.fixture(scope="class")
    def datasets(self):
        reader = CAMELSFRReader()
        return reader.read_all(_make_config(max_stations=2))

    def test_returns_two_datasets(self, datasets):
        """Default config should return both daily and monthly."""
        assert len(datasets) == 2
        resolutions = {ds.time_resolution for ds in datasets}
        assert resolutions == {"daily", "monthly"}
