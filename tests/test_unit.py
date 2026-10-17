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


def test_cama_total_runoff_compute_converts_volume_flux_to_depth_rate(tmp_path):
    """CaMa's volume flux uses the spacing of its input grid, not a fixed 0.25° grid."""
    import xarray as xr

    from openbench.data.compute import execute_compute
    from openbench.data.registry.manager import RegistryManager

    mapping = RegistryManager(user_dir=tmp_path).get_model("CaMa").variables["Total_Runoff"]
    assert mapping.varname == "runoff"
    radius_m = 6_371_000.0
    for resolution, lat, coord_names in (
        (0.25, np.array([0.125, 0.375]), ("lat", "lon")),
        (1.0, np.array([60.5, 61.5]), ("lat_cama", "lon_cama")),
    ):
        lat_name, lon_name = coord_names
        ds = xr.Dataset(
            {"runoff": (("time", lat_name, lon_name), np.ones((1, 2, 2)))},
            coords={"time": [0], lat_name: lat, lon_name: [resolution / 2.0, resolution * 1.5]},
        )

        result = execute_compute(ds, mapping.compute, "Total_Runoff")
        converted, base_unit = UnitProcessing.convert_unit(result, mapping.varunit)

        dlon_rad = np.deg2rad(resolution)
        area_m2 = (
            radius_m**2
            * dlon_rad
            * (np.sin(np.deg2rad(lat + resolution / 2.0)) - np.sin(np.deg2rad(lat - resolution / 2.0)))
        )
        expected_mm_day = 86_400_000.0 / area_m2

        np.testing.assert_allclose(converted.squeeze().values, np.repeat(expected_mm_day[:, None], 2, axis=1))
        assert base_unit == "mm day-1"


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


def test_plain_w_m2_stays_an_energy_flux():
    """'w m-2' is also listed as an input of 'mm day-1'; as a base unit it must
    still map to itself, or every energy flux turns into mm day-1."""
    unit._UNIT_LOOKUP_CACHE = None
    for spelling in ["w m-2", "W m-2"]:
        converted, base_unit = UnitProcessing.convert_unit(28.0, spelling)
        assert base_unit == "w m-2"
        assert converted == 28.0


def test_water_flux_given_in_w_m2_converts_to_mm_day():
    import xarray as xr

    from openbench.data._processing_transforms import ProcessingTransformMixin

    unit._UNIT_LOOKUP_CACHE = None
    ds = xr.Dataset({"v": ("time", np.array([2.5e6 / 86400.0]))})
    for item, base, value in [
        ("Evapotranspiration", "mm day-1", 1.0),
        ("Canopy_Transpiration", "mm day-1", 1.0),
        ("Latent_Heat", "w m-2", 2.5e6 / 86400.0),
    ]:
        proc = ProcessingTransformMixin()
        proc.item = item
        out, new_unit = proc.process_units(ds, "W m-2")
        assert new_unit == base
        np.testing.assert_allclose(out["v"].values, [value])


def test_display_unit_is_the_unit_after_conversion():
    unit._UNIT_LOOKUP_CACHE = None
    for declared, item, expected in [
        ("W m-2", "Latent_Heat", "W m-2"),
        ("K", "Surface_Air_Temperature", "K"),
        ("kg m-2 s-1", "Total_Runoff", "mm day-1"),
        ("degC", "Surface_Air_Temperature", "K"),
        ("W m-2", "Evapotranspiration", "mm day-1"),
        ("no such unit", "Latent_Heat", "no such unit"),
    ]:
        assert UnitProcessing.display_unit(declared, item) == expected


def test_plot_label_follows_the_converted_data():
    from types import SimpleNamespace

    from openbench.visualization.Fig_Basic_Plot import determine_display_unit
    from openbench.visualization.Fig_toolbox import convert_unit as label

    unit._UNIT_LOOKUP_CACHE = None
    for ref_unit, sim_unit, item, expected in [
        ("kg m-2 s-1", "kg m-2 s-1", "Total_Runoff", "mm day-1"),
        ("kg m-2 s-1", "mm s-1", "Total_Runoff", "mm day-1"),
        ("W m-2", "W m-2", "Latent_Heat", "W m-2"),
        ("degC", "K", "Surface_Air_Temperature", "K"),
    ]:
        ns = SimpleNamespace(ref_varunit=ref_unit, sim_varunit=sim_unit, item=item)
        assert determine_display_unit(ns) == label(expected)


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


def test_check_comparable_units_rejects_pairs_converted_to_different_units():
    import pytest

    unit._UNIT_LOOKUP_CACHE = None
    unit.check_comparable_units("Latent_Heat", "W/m2", "w m-2")
    unit.check_comparable_units("Evapotranspiration", "W m-2", "mm s-1")
    unit.check_comparable_units("Surface_Albedo", "-", "some_unknown_unit")

    with pytest.raises(ValueError, match="'w m-2'.*'mm day-1'"):
        unit.check_comparable_units("Latent_Heat", "W/m2", "mm day-1")


def test_file_unit_warning_only_fires_for_a_different_recognized_unit(caplog):
    import logging

    unit._UNIT_LOOKUP_CACHE = None
    unit._FILE_UNIT_WARNINGS.clear()

    with caplog.at_level(logging.WARNING):
        unit.warn_if_file_unit_differs("W/m2", "W m-2", "Latent_Heat", "PLUMBER2")
        unit.warn_if_file_unit_differs("mm.day-1", "mm day-1", "Evapotranspiration", "GLEAM")
        unit.warn_if_file_unit_differs("not a unit", "mm day-1", "Evapotranspiration", "GLEAM")
        assert not caplog.records

        unit.warn_if_file_unit_differs("mm.month-1", "mm day-1", "Evapotranspiration", "GLEAM")
        unit.warn_if_file_unit_differs("mm.month-1", "mm day-1", "Evapotranspiration", "GLEAM")

    assert len(caplog.records) == 1
    assert "GLEAM" in caplog.text and "'mm.month-1'" in caplog.text and "'mm day-1'" in caplog.text


def test_registry_unit_spellings_convert_instead_of_passing_through():
    unit._UNIT_LOOKUP_CACHE = None
    cases = [
        ("mm/s", 1.0, "mm day-1", 86400.0),
        ("degrees Celsius", 15.0, "k", 288.15),
        ("J m-2 day-1", 86400.0, "w m-2", 1.0),
        ("m of water equivalent", 0.05, "mm", 50.0),
        ("kg co2 m-2 s-1", 44.01e-3 / 86400, "gc m-2 day-1", 12.011),
        ("g co2 m-2 s-1", 44.01 / 86400, "gc m-2 day-1", 12.011),
        ("g C m-2 yr-1", 365.25, "gc m-2 day-1", 1.0),
        ("mm d-1", 2.0, "mm day-1", 2.0),
        ("m3/m3", 0.3, "unitless", 0.3),
        ("kPa", 1.0, "pa", 1000.0),
        ("Mg ha-1", 5.0, "t ha-1", 5.0),
    ]
    for declared, value, expected_base, expected in cases:
        converted, base_unit = UnitProcessing.convert_unit(value, declared)
        assert base_unit == expected_base, declared
        np.testing.assert_allclose(converted, expected, err_msg=declared)


def test_methane_fluxes_convert_to_carbon_flux():
    unit._UNIT_LOOKUP_CACHE = None
    carbon_per_ch4 = 12.011 / 16.043
    cases = [
        ("kg CH4 m-2 s-1", 1e-3 / 86400, carbon_per_ch4),
        ("g CH4 m-2 d-1", 1.0, carbon_per_ch4),
        ("mg CH4 m-2 d-1", 1000.0, carbon_per_ch4),
        ("g CH4 m-2 yr-1", 365.25, carbon_per_ch4),
        ("nmol CH4 m-2 s-1", 1e9 / 86400, 12.011),
        ("nmol m-2 s-1", 1e9 / 86400, 12.011),
    ]
    for declared, value, expected in cases:
        converted, base_unit = UnitProcessing.convert_unit(value, declared)
        assert base_unit == "gc m-2 day-1", declared
        np.testing.assert_allclose(converted, expected, err_msg=declared)


def test_sediment_units_convert_to_concentration_and_load_bases():
    unit._UNIT_LOOKUP_CACHE = None
    cases = [
        ("mg L-1", 5.0, "mg l-1", 5.0),
        ("g m-3", 5.0, "mg l-1", 5.0),
        ("kg m-3", 0.005, "mg l-1", 5.0),
        ("g/L", 0.005, "mg l-1", 5.0),
        ("t d-1", 864.0, "t day-1", 864.0),
        ("kg s-1", 10.0, "t day-1", 864.0),
        ("kg d-1", 864000.0, "t day-1", 864.0),
        ("t yr-1", 365.25, "t day-1", 1.0),
    ]
    for declared, value, base, expected in cases:
        converted, base_unit = UnitProcessing.convert_unit(value, declared)
        assert base_unit == base, declared
        np.testing.assert_allclose(converted, expected, err_msg=declared)


def test_colm2024_sediment_outputs_reach_the_sedref_units(tmp_path):
    """CoLM's per-size-class volume outputs, summed and given CoLM's grain density."""
    import xarray as xr

    from openbench.data.compute import execute_compute
    from openbench.data.registry.manager import RegistryManager

    unit._UNIT_LOOKUP_CACHE = None
    colm = RegistryManager(user_dir=tmp_path).get_model("CoLM2024")
    density = 2650.0
    ssc_mg_l, ssl_t_d = 120.0, 4300.0
    shares = (0.5, 0.3, 0.2)  # CoLM's three size classes: clay, silt, sand
    ds = xr.Dataset(
        {
            **{f"f_sedcon_{i}": ("x", [share * ssc_mg_l / 1000 / density]) for i, share in enumerate(shares, 1)},
            **{f"f_sedout_{i}": ("x", [share * ssl_t_d / 86.4 / density]) for i, share in enumerate(shares, 1)},
        }
    )
    for item, expected, base in (
        ("Suspended_Sediment_Concentration", ssc_mg_l, "mg l-1"),
        ("Suspended_Sediment_Load", ssl_t_d, "t day-1"),
    ):
        mapping = colm.variables[item]
        computed = execute_compute(ds, mapping.compute, item)
        converted, base_unit = UnitProcessing.convert_unit(computed.values, mapping.varunit)
        assert base_unit == base
        np.testing.assert_allclose(converted, [expected])
    assert colm.variables["Discharge_For_Sediment"].varname == "f_discharge"
