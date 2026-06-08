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
