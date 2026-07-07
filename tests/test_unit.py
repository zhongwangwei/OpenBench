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
