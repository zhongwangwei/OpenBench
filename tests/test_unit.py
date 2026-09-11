import numpy as np

from openbench.data import unit
from openbench.data.unit import UnitProcessing


def test_heat_flux_unit_aliases_normalize_to_w_m2():
    unit._UNIT_LOOKUP_CACHE = None
    data = np.array([1.0, 2.5])

    for alias in ["W/m2", "watt/m2", "watt m-2", "W m**-2"]:
        converted, base_unit = UnitProcessing.convert_unit(data, alias)
        assert base_unit == "w m-2"
        np.testing.assert_allclose(converted, data)


def test_hydrology_unit_conversions_are_input_to_base():
    unit._UNIT_LOOKUP_CACHE = None

    cases = [
        ("l s-1", "m3 s-1", 1.0, 0.001),
        ("m3", "mcm", 1.0, 0.000001),
        ("km3", "mcm", 1.0, 1000.0),
        ("m year-1", "mm year-1", 1.0, 1000.0),
        ("cm year-1", "mm year-1", 1.0, 10.0),
    ]

    for input_unit, expected_base, value, expected in cases:
        converted, base_unit = UnitProcessing.convert_unit(value, input_unit)
        assert base_unit == expected_base
        assert converted == expected


def test_convert_nc_does_not_mutate_input_dataset():
    import xarray as xr

    from openbench.util.converttype import Convert_Type

    ds = xr.Dataset(
        {"var": ("x", np.array([1.0, 2.0], dtype=np.float64))},
        coords={"x": np.array([0.0, 1.0], dtype=np.float64)},
    )

    converted = Convert_Type.convert_nc(ds)

    assert ds["var"].dtype == np.float64
    assert converted["var"].dtype == np.float32


def test_latent_heat_flux_uses_documented_2p5e6_factor():
    unit._UNIT_LOOKUP_CACHE = None
    converted, base_unit = UnitProcessing.convert_unit(1.0, "W m-2 heat")

    assert base_unit == "mm day-1"
    assert converted == 86400.0 / 2.5e6


def test_metre_per_day_runoff_converts_to_mm_per_day():
    """ERA5-Land 'ro' is a daily runoff depth in metres (m/day); it must map to
    the mm day-1 base so it lines up with model runoff in mm s-1, not be left as
    a bare length 1000x off."""
    unit._UNIT_LOOKUP_CACHE = None
    for alias in ["m day-1", "m d-1"]:
        converted, base_unit = UnitProcessing.convert_unit(2.0, alias)
        assert base_unit == "mm day-1"
        assert converted == 2000.0


def test_cm_equivalent_water_thickness_converts_to_mm():
    """GRAiCE/GRACE TWSC in 'cm of equivalent water thickness' must reach the mm
    base (x10) to match model TWSC in mm, not stay 10x off."""
    unit._UNIT_LOOKUP_CACHE = None
    converted, base_unit = UnitProcessing.convert_unit(3.0, "cm of equivalent water thickness")
    assert base_unit == "mm"
    assert converted == 30.0


def test_bare_cm_remains_a_length_in_metres():
    """A bare 'cm' must stay a length (base metre), so adding the water-thickness
    string above does not hijack centimetre lengths into the mm depth base."""
    unit._UNIT_LOOKUP_CACHE = None
    converted, base_unit = UnitProcessing.convert_unit(100.0, "cm")
    assert base_unit == "m"
    assert converted == 1.0


def test_dimensionless_dash_is_recognized_as_unitless():
    """Albedo computed as f_sr/f_solarin is labelled '-'; it must be recognized
    as unitless (passthrough), not trigger a no-conversion warning."""
    unit._UNIT_LOOKUP_CACHE = None
    for alias in ["-", "none"]:
        converted, base_unit = UnitProcessing.convert_unit(0.15, alias)
        assert base_unit == "unitless"
        assert converted == 0.15


def test_land_model_unit_aliases_normalize():
    unit._UNIT_LOOKUP_CACHE = None

    converted, base_unit = UnitProcessing.convert_unit(2.0, "mm H2O/s")
    assert base_unit == "mm day-1"
    assert converted == 172800.0

    converted, base_unit = UnitProcessing.convert_unit(0.75, "kg kg-1")
    assert base_unit == "unitless"
    assert converted == 0.75

    converted, base_unit = UnitProcessing.convert_unit(1013.25, "hPa")
    assert base_unit == "pa"
    assert converted == 101325.0

    converted, base_unit = UnitProcessing.convert_unit(0.001, "kg C m-2 s-1")
    assert base_unit == "gc m-2 day-1"
    assert converted == 86400.0


def test_gldas_fixed_soil_layers_convert_to_volumetric_moisture(tmp_path):
    import xarray as xr

    from openbench.data.compute import execute_compute
    from openbench.data.registry.manager import RegistryManager

    profile = RegistryManager(user_dir=tmp_path).get_model("GLDAS")
    ds = xr.Dataset(
        {
            "SoilMoi0_10cm_inst": ("x", [20.0]),
            "SoilMoi10_40cm_inst": ("x", [60.0]),
            "RootMoist_inst": ("x", [200.0]),
        }
    )
    for item in ("Surface_Soil_Moisture", "Soil_Moisture_Lev2", "Root_Zone_Soil_Moisture"):
        mapping = profile.variables[item]
        assert mapping.varunit == "m3 m-3"
        result = execute_compute(ds, mapping.compute, item)
        converted, base_unit = UnitProcessing.convert_unit(result, mapping.varunit)
        np.testing.assert_allclose(converted, [0.2])
        assert base_unit == UnitProcessing.convert_unit(None, "m3 m-3")[1]


def test_month_rate_uses_preserved_noleap_calendar_days():
    import pandas as pd
    import xarray as xr

    unit._UNIT_LOOKUP_CACHE = None
    data = xr.DataArray(
        [28.0],
        dims="time",
        coords={"time": pd.DatetimeIndex(["2000-02-01"])},
    )
    data.time.attrs["original_calendar"] = "noleap"

    converted, base_unit = UnitProcessing.convert_unit(data, "mm month-1")

    assert base_unit == "mm day-1"
    np.testing.assert_allclose(converted.values, [1.0])


def test_month_rate_uses_preserved_360_day_calendar_days():
    import pandas as pd
    import xarray as xr

    unit._UNIT_LOOKUP_CACHE = None
    data = xr.DataArray(
        [30.0],
        dims="time",
        coords={"time": pd.DatetimeIndex(["2000-02-01"])},
    )
    data.time.attrs["original_calendar"] = "360_day"

    converted, base_unit = UnitProcessing.convert_unit(data, "mm month-1")

    assert base_unit == "mm day-1"
    np.testing.assert_allclose(converted.values, [1.0])


def test_day_rate_uses_preserved_noleap_calendar_year_days():
    import pandas as pd
    import xarray as xr

    data = xr.DataArray(
        [1.0],
        dims="time",
        coords={"time": pd.DatetimeIndex(["2000-01-01"])},
    )
    data.time.attrs["original_calendar"] = "noleap"

    converted = unit._per_day_to_per_year(data)

    np.testing.assert_allclose(converted.values, [365.0])


def test_day_rate_uses_preserved_360_day_calendar_year_days():
    import pandas as pd
    import xarray as xr

    data = xr.DataArray(
        [1.0],
        dims="time",
        coords={"time": pd.DatetimeIndex(["2000-01-01"])},
    )
    data.time.attrs["original_calendar"] = "360_day"

    converted = unit._per_day_to_per_year(data)

    np.testing.assert_allclose(converted.values, [360.0])


def test_day_rate_uses_gregorian_leap_year_days_without_original_calendar():
    import pandas as pd
    import xarray as xr

    data = xr.DataArray(
        [1.0, 1.0],
        dims="time",
        coords={"time": pd.DatetimeIndex(["2000-01-01", "2001-01-01"])},
    )

    converted = unit._per_day_to_per_year(data)

    np.testing.assert_allclose(converted.values, [366.0, 365.0])


def test_rate_conversions_keep_scalar_fallbacks_without_time_coordinate():
    yearly = unit._per_day_to_per_year(2.0)
    daily = unit._per_month_to_per_day(60.875)

    assert yearly == 730.5
    assert daily == 2.0


def test_declared_calendar_errors_are_not_silently_downgraded(monkeypatch):
    import pandas as pd
    import xarray as xr

    class BrokenCalendar:
        def __init__(self, *args):
            raise RuntimeError("bad declared calendar")

    data = xr.DataArray(
        [1.0],
        dims="time",
        coords={"time": pd.DatetimeIndex(["2000-01-01"])},
    )
    data.time.attrs["original_calendar"] = "360_day"
    monkeypatch.setattr(unit, "_cftime_calendar_class", lambda calendar: BrokenCalendar)

    try:
        unit._per_day_to_per_year(data)
    except RuntimeError as exc:
        assert "bad declared calendar" in str(exc)
    else:
        raise AssertionError("declared calendar failure was silently downgraded")


def test_declared_calendar_requires_cftime(monkeypatch):
    import builtins

    import pandas as pd
    import xarray as xr

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cftime":
            raise ImportError("missing cftime")
        return real_import(name, *args, **kwargs)

    data = xr.DataArray(
        [30.0],
        dims="time",
        coords={"time": pd.DatetimeIndex(["2000-02-01"])},
    )
    data.time.attrs["original_calendar"] = "360_day"
    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        unit._per_month_to_per_day(data)
    except ImportError as exc:
        assert "missing cftime" in str(exc)
    else:
        raise AssertionError("declared calendar without cftime was silently downgraded")
