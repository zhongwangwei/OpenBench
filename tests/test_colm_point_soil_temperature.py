"""Point-depth approximations from the fixed native CoLM2024.1 soil nodes."""

import numpy as np
import pytest
import xarray as xr

from openbench.data.compute import ComputeError, compute_dependency_names, compute_inputs_known, execute_compute
from openbench.data.registry.manager import RegistryManager

DEPTHS = (5, 10, 15, 20, 40)
PAIRS = ((6, 7), (7, 8), (8, 9), (8, 9), (10, 11))
# Pinned CoLM2024.1 MOD_Vars_Global.F90:137-153, meters below the surface.
NODES = 0.025 * (np.exp(0.5 * (np.arange(1, 11) - 0.5)) - 1)


@pytest.fixture(params=zip(DEPTHS, PAIRS))
def point(request, tmp_path):
    depth, pair = request.param
    logical = f"Soil_Temperature_{depth}cm"
    mapping = RegistryManager(user_dir=tmp_path).get_model("CoLM2024").variables[logical]
    return depth, pair, logical, mapping


def profile(values):
    data = np.broadcast_to(np.asarray(values)[None, :, None], (2, len(values), 3)).copy()
    return xr.Dataset(
        {"f_t_soisno": (("time", "soilsnow", "lat"), data)},
        coords={"time": [0, 1], "soilsnow": np.arange(-4, len(values) - 4), "lat": [0, 1, 2]},
    )


def evaluate(ds, point):
    return execute_compute(ds, point[3].compute, point[2])


def test_kelvin_constant_and_alias(point, tmp_path):
    _, _, logical, mapping = point
    assert (mapping.varname, mapping.varunit, mapping.fallbacks) == (logical, "K", None)
    np.testing.assert_allclose(evaluate(profile(np.full(15, 280.0)), point), 280, rtol=0, atol=1e-12)
    assert RegistryManager(user_dir=tmp_path).get_model("CoLM").variables[logical] == mapping


def test_linear_temperature_hits_requested_depth(point):
    depth = point[0] / 100
    values = np.r_[np.full(5, np.nan), 275 + 12 * NODES]
    np.testing.assert_allclose(evaluate(profile(values), point), 275 + 12 * depth, rtol=0, atol=1e-12)


def test_only_bracketing_nodes_contribute_with_correct_weights(point):
    depth, (left, right), _, _ = point
    z = depth / 100
    assert NODES[left - 5] < z < NODES[right - 5]
    fraction = (z - NODES[left - 5]) / (NODES[right - 5] - NODES[left - 5])
    for index in range(15):
        values = np.zeros(15)
        values[index] = 1
        expected = 1 - fraction if index == left else fraction if index == right else 0
        np.testing.assert_allclose(evaluate(profile(values), point), expected, rtol=0, atol=1e-15)


def test_unused_missing_layers_are_ignored(point):
    values = np.full(15, np.nan)
    values[list(point[1])] = 280
    np.testing.assert_allclose(evaluate(profile(values), point), 280)


@pytest.mark.parametrize("side", [0, 1])
def test_either_used_node_missing_propagates(point, side):
    values = np.full(15, 280.0)
    values[point[1][side]] = np.nan
    assert np.isnan(evaluate(profile(values), point)).all()


def test_arbitrary_dimension_order(point):
    ds = profile(np.arange(15) + 270.0)
    expected = evaluate(ds, point)
    actual = evaluate(ds.transpose("lat", "soilsnow", "time"), point)
    xr.testing.assert_allclose(actual.transpose(*expected.dims), expected)


@pytest.mark.parametrize("length", [10, 20])
def test_nonstandard_layout_fails_closed(point, length):
    with pytest.raises(ComputeError):
        evaluate(profile(np.full(length, 280.0)), point)


def test_missing_named_dimension_fails_closed(point):
    with pytest.raises(ComputeError):
        evaluate(profile(np.full(15, 280.0)).rename({"soilsnow": "layer"}), point)


def test_virtual_coordinate_supported(point):
    np.testing.assert_allclose(evaluate(profile(np.full(15, 280.0)).drop_vars("soilsnow"), point), 280)


def test_native_dependency_not_misleading_5cm_mean(point):
    mapping = point[3]
    assert compute_dependency_names(mapping.compute) == ["f_t_soisno"]
    assert compute_inputs_known(mapping.compute)
    ds = xr.Dataset({"f_t_soisno_5cm": ("time", [280.0])})
    with pytest.raises(ComputeError):
        evaluate(ds, point)


def test_dask_stays_lazy(point):
    da = pytest.importorskip("dask.array")
    result = evaluate(profile(np.full(15, 280.0)).chunk({"time": 1, "soilsnow": 3}), point)
    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(result.compute(), 280)
