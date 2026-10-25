"""CoLM's native five-snow/ten-soil profile: thickness mean over 0–1 m."""

import numpy as np
import pytest
import xarray as xr

from openbench.data.compute import (
    ComputeError,
    MissingComputeVariable,
    compute_dependency_names,
    compute_inputs_known,
    execute_compute,
)
from openbench.data.registry.manager import RegistryManager

VARIABLE = "Root_Zone_Soil_Temperature"
WEIGHTS = np.array(
    [
        0.017512817916255204,
        0.027578969259676255,
        0.045470033242413194,
        0.07496741098620857,
        0.12360036510228051,
        0.20378255101043175,
        0.3359806264484327,
        0.17110722603430184,
    ]
)


@pytest.fixture
def mapping(tmp_path):
    return RegistryManager(user_dir=tmp_path).get_model("CoLM2024").variables[VARIABLE]


def profile(values):
    data = np.broadcast_to(np.asarray(values)[None, :, None, None], (2, len(values), 2, 3)).copy()
    return xr.Dataset(
        {"f_t_soisno": (("time", "soilsnow", "lat", "lon"), data)},
        coords={"time": [0, 1], "soilsnow": np.arange(-4, len(values) - 4), "lat": [10, 20], "lon": [0, 1, 2]},
    )


def evaluate(ds, mapping):
    return execute_compute(ds, mapping.compute, VARIABLE)


def test_constant_kelvin_and_alias(mapping, tmp_path):
    assert mapping.varname == VARIABLE
    assert mapping.varunit == "K"
    assert mapping.fallbacks is None
    np.testing.assert_allclose(evaluate(profile(np.full(15, 280.0)), mapping), 280, rtol=0, atol=1e-12)
    alias = RegistryManager(user_dir=tmp_path).get_model("CoLM")
    assert alias.variables[VARIABLE] == mapping


def test_weighted_profile_includes_partial_eighth_soil_layer(mapping):
    values = np.arange(15, dtype=float) + 270
    result = evaluate(profile(values), mapping)
    np.testing.assert_allclose(result, np.dot(values[5:13], WEIGHTS), rtol=0, atol=1e-10)
    values[12] += 10
    delta = evaluate(profile(values), mapping) - result
    np.testing.assert_allclose(delta, 10 * WEIGHTS[7], rtol=0, atol=1e-10)
    assert abs(WEIGHTS.sum() - 1) < 1e-12


def test_snow_and_deep_missing_do_not_contaminate_mean(mapping):
    values = np.full(15, 280.0)
    values[:5] = np.nan
    values[13:] = np.nan
    np.testing.assert_allclose(evaluate(profile(values), mapping), 280)


@pytest.mark.parametrize("layer", range(5, 13))
def test_any_used_soil_layer_missing_propagates_without_renormalization(mapping, layer):
    values = np.full(15, 280.0)
    values[layer] = np.nan
    assert np.isnan(evaluate(profile(values), mapping)).all()


def test_named_dimension_handles_arbitrary_axis_order(mapping):
    ds = profile(np.arange(15) + 270.0)
    expected = evaluate(ds, mapping)
    actual = evaluate(ds.transpose("lon", "soilsnow", "time", "lat"), mapping)
    xr.testing.assert_allclose(actual.transpose(*expected.dims), expected)


def test_dask_stays_lazy(mapping):
    da = pytest.importorskip("dask.array")
    result = evaluate(profile(np.full(15, 280.0)).chunk({"time": 1, "soilsnow": 3}), mapping)
    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(result.compute(), 280)


@pytest.mark.parametrize("layers", [10, 20])
def test_nonstandard_profile_length_fails_closed(mapping, layers):
    with pytest.raises(ComputeError):
        evaluate(profile(np.full(layers, 280.0)), mapping)


def test_native_dependency_and_missing_input_preflight(mapping):
    assert compute_dependency_names(mapping.compute) == ["f_t_soisno"]
    assert compute_inputs_known(mapping.compute) is True
    with pytest.raises(MissingComputeVariable) as caught:
        evaluate(xr.Dataset(), mapping)
    assert caught.value.inputs_known is True
    assert caught.value.dependencies == ("f_t_soisno",)


def test_missing_named_dimension_fails_closed(mapping):
    ds = profile(np.full(15, 280.0)).rename({"soilsnow": "unknown_layers"})
    with pytest.raises(ComputeError):
        evaluate(ds, mapping)


def test_virtual_dimension_coordinate_is_supported(mapping):
    ds = profile(np.full(15, 280.0)).drop_vars("soilsnow")
    np.testing.assert_allclose(evaluate(ds, mapping), 280)


def test_weights_match_native_grid_depth_overlaps(mapping):
    # CoLM2024.1 MOD_Vars_Global: exponential nodes, midpoint interfaces.
    nodes = 0.025 * (np.exp(0.5 * (np.arange(1, 11) - 0.5)) - 1)
    interfaces = np.concatenate(([0.0], (nodes[:-1] + nodes[1:]) / 2))
    overlaps = np.diff(np.minimum(interfaces, 1.0))[:8]
    np.testing.assert_allclose(WEIGHTS, overlaps, rtol=0, atol=1e-15)
    assert nodes[7] > 1.0  # The eighth node is deeper than 1 m, but its cell overlaps.
    for layer, thickness in enumerate(overlaps, start=5):
        values = np.zeros(15)
        values[layer] = 1.0
        np.testing.assert_allclose(evaluate(profile(values), mapping), thickness, rtol=0, atol=1e-15)
