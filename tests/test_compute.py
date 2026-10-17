"""Tests for compute expression executor."""

import numpy as np
import pytest
import xarray as xr

from openbench.data.compute import ComputeError, execute_compute


def _make_ds():
    """Create a test dataset."""
    return xr.Dataset(
        {
            "a": xr.DataArray(np.array([1.0, 2.0, 3.0])),
            "b": xr.DataArray(np.array([10.0, 20.0, 30.0])),
            "rain": xr.DataArray(np.array([0.5, 1.0, 0.3])),
            "snow": xr.DataArray(np.array([0.1, 0.0, 0.2])),
        }
    )


def test_simple_expression():
    ds = _make_ds()
    result = execute_compute(ds, "ds['a'] + ds['b']", "test")
    np.testing.assert_array_equal(result.values, [11.0, 22.0, 33.0])


def test_multi_step_expression():
    ds = _make_ds()
    result = execute_compute(ds, "total = ds['a'] + ds['b']; total * 2", "test")
    np.testing.assert_array_equal(result.values, [22.0, 44.0, 66.0])


def test_division():
    ds = _make_ds()
    result = execute_compute(ds, "ds['b'] / ds['a']", "test")
    np.testing.assert_array_equal(result.values, [10.0, 10.0, 10.0])


def test_precipitation_compute():
    ds = _make_ds()
    result = execute_compute(ds, "ds['rain'] + ds['snow']", "Precipitation")
    np.testing.assert_array_almost_equal(result.values, [0.6, 1.0, 0.5])


def test_numpy_available():
    ds = _make_ds()
    result = execute_compute(ds, "(ds['a']**2 + ds['b']**2)**0.5", "magnitude")
    assert result.values[0] == pytest.approx(np.sqrt(101), rel=1e-5)


def test_missing_variable_error():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="not found"):
        execute_compute(ds, "ds['nonexistent'] + ds['a']", "test")


def test_empty_expression_error():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="Empty"):
        execute_compute(ds, "", "test")


def test_fillna():
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.array([1.0, np.nan, 3.0])),
        }
    )
    result = execute_compute(ds, "ds['x'].fillna(0)", "test")
    np.testing.assert_array_equal(result.values, [1.0, 0.0, 3.0])


def test_compute_rejects_io_calls_from_allowed_roots(tmp_path):
    ds = _make_ds()
    out = tmp_path / "out.nc"
    out_expr = out.as_posix()

    with pytest.raises(ComputeError, match="xarray function 'open_dataset' is not allowed"):
        execute_compute(ds, f"xr.open_dataset('{out_expr}')", "test")

    with pytest.raises(ComputeError, match="numpy function 'fromfile' is not allowed"):
        execute_compute(ds, f"np.fromfile('{out_expr}', dtype=np.uint8)", "test")

    with pytest.raises(ComputeError, match="method 'to_netcdf' is not allowed"):
        execute_compute(ds, f"ds['a'].to_netcdf('{out_expr}')", "test")

    assert not out.exists()


def test_compute_allows_catalog_method_chain():
    ds = xr.Dataset({"resp": xr.DataArray(np.array([31536000.0]), attrs={"units": "gC year-1"})})

    result = execute_compute(
        ds,
        "ds['resp'] / 31536000.0 if 'year' in ds['resp'].attrs.get('units', '').lower() else ds['resp']",
        "Respiration",
    )

    np.testing.assert_allclose(result.values, [1.0])


def test_te_total_runoff_compute_handles_zdepth_dimension():
    from openbench.data.registry.manager import get_registry

    profile = get_registry().get_model("TE")
    expression = profile.variables["Total_Runoff"].compute
    ds = xr.Dataset(
        {
            "RUNOFF": (
                ["zdepth", "time"],
                np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
            )
        },
        coords={"zdepth": [0, 1], "time": [0, 1, 2]},
    )

    result = execute_compute(ds, expression, "Total_Runoff")

    assert result.dims == ("time",)
    np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])


def test_compute_rejects_unknown_identifier():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="identifier 'os' is not allowed"):
        execute_compute(ds, "os.system('touch /tmp/openbench_pwn')", "test")


def test_compute_rejects_invalid_assignment_target():
    ds = _make_ds()
    with pytest.raises(ComputeError, match="Invalid assignment target"):
        execute_compute(ds, "ds['a'] = ds['b']; ds['a']", "test")


def test_compute_validation_allows_explicit_extra_names():
    from openbench.data.compute import _validate_expression

    _validate_expression("value * 12.011 - f_assim", allowed_names={"value", "f_assim", "np"})
    with pytest.raises(ComputeError, match="identifier 'missing' is not allowed"):
        _validate_expression("value + missing", allowed_names={"value", "np"})


def test_compute_supports_dataset_membership_checks():
    ds = xr.Dataset(
        {
            "RUNOFF": xr.DataArray(np.array([1.0, 2.0, 3.0])),
            "fallback": xr.DataArray(np.array([10.0, 20.0, 30.0])),
        }
    )

    result = execute_compute(
        ds,
        "ds['RUNOFF'] if 'RUNOFF' in ds else ds['fallback']",
        "Runoff",
    )

    np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])


def test_compute_membership_checks_are_case_insensitive():
    ds = xr.Dataset({"runoff": xr.DataArray(np.array([4.0, 5.0]))})

    result = execute_compute(ds, "ds['RUNOFF'] if 'RUNOFF' in ds else 0", "Runoff")

    np.testing.assert_array_equal(result.values, [4.0, 5.0])


def test_sum_prefix_adds_exactly_the_numbered_parts():
    ds = xr.Dataset(
        {
            "f_sedcon_1": ("x", [1.0, 2.0]),
            "F_SEDCON_2": ("x", [0.5, np.nan]),
            "f_sedcon_total": ("x", [100.0, 100.0]),
            "f_discharge": ("x", [3.0, 4.0]),
        }
    )
    result = execute_compute(ds, "ds.sum_prefix('f_sedcon_', 2) * 2", "Suspended_Sediment_Concentration")
    np.testing.assert_array_equal(result.values, [3.0, np.nan])


def test_sum_prefix_with_a_missing_part_is_an_integrity_error():
    from openbench.data.compute import ComputeIntegrityError, MissingComputeVariable

    ds = xr.Dataset({"f_sedcon_1": ("x", [1.0]), "f_sedcon_3": ("x", [1.0])})
    with pytest.raises(ComputeIntegrityError, match="f_sedcon_2") as caught:
        execute_compute(ds, "ds.sum_prefix('f_sedcon_', 3)", "Suspended_Sediment_Concentration")
    assert not isinstance(caught.value, MissingComputeVariable)


def test_sum_prefix_refuses_parts_beyond_the_count():
    from openbench.data.compute import ComputeIntegrityError

    ds = xr.Dataset({f"f_sedcon_{i}": ("x", [1.0]) for i in (1, 2, 3, 4)})
    with pytest.raises(ComputeIntegrityError, match="f_sedcon_4 found beyond the 3 parts"):
        execute_compute(ds, "ds.sum_prefix('f_sedcon_', 3)", "Suspended_Sediment_Concentration")


@pytest.mark.parametrize("parts", ["0", "True", "'3'", "2.0"])
def test_sum_prefix_needs_a_positive_whole_part_count(parts):
    ds = xr.Dataset({"f_sedcon_1": ("x", [1.0])})
    from openbench.data.compute import ComputeIntegrityError

    with pytest.raises(ComputeIntegrityError, match="positive whole number"):
        execute_compute(ds, f"ds.sum_prefix('f_sedcon_', {parts})", "Suspended_Sediment_Concentration")


def test_compute_dependency_names_expand_summed_parts():
    from openbench.data.compute import compute_dependency_names

    assert compute_dependency_names("ds.sum_prefix('f_sedout_', 3) * 2650 + ds['f_x']") == [
        "f_x",
        "f_sedout_1",
        "f_sedout_2",
        "f_sedout_3",
    ]


def test_sum_prefix_without_any_parts_is_missing_data():
    from openbench.data.compute import MissingComputeVariable

    with pytest.raises(MissingComputeVariable, match="No numbered variables"):
        execute_compute(xr.Dataset(), "ds.sum_prefix('f_sedcon_', 3)", "Sediment")


def test_compute_context_preserves_exception_with_custom_constructor(monkeypatch):
    from openbench.data import compute

    class DetailedMissing(compute.MissingComputeVariable):
        def __init__(self, variable, reason):
            self.variable = variable
            super().__init__(f"{variable}: {reason}")

    original = DetailedMissing("rain", "unavailable")

    def missing(self, key):
        raise original

    monkeypatch.setattr(compute._CaseInsensitiveDatasetProxy, "__getitem__", missing)
    with pytest.raises(DetailedMissing, match="Computing Runoff") as caught:
        execute_compute(xr.Dataset(), "ds [ 'rain' ] + ds['snow']", "Runoff")
    assert caught.value is original
    assert caught.value.variable == "rain"
    assert caught.value.dependencies == ("rain", "snow")


def test_compute_dependencies_parse_dataset_access_without_method_names():
    from openbench.data.compute import compute_dependency_names

    names = compute_dependency_names(
        "a = ds.runoff + ds.get('rain'); a + ds [ 'snow' ] + ds.sum_prefix(prefix='part', parts=2).mean()"
    )
    assert set(names) == {"runoff", "rain", "snow", "part1", "part2"}
    assert compute_dependency_names("ds['rain'") == []


@pytest.mark.parametrize(
    "expression",
    [" ds['rain'] + ds['snow']", "t = ds['rain'];\n    t + ds['snow']", "\tds.rain + ds.get('snow')"],
)
def test_compute_dependencies_survive_leading_whitespace_and_indented_steps(expression):
    from openbench.data.compute import compute_dependency_names

    assert set(compute_dependency_names(expression)) == {"rain", "snow"}


def test_compute_dependencies_keep_parsable_steps_when_one_is_malformed():
    from openbench.data.compute import compute_dependency_names

    assert compute_dependency_names("ds['rain']; ds['snow'] +") == ["rain"]


@pytest.mark.parametrize("expression", ["ds.rain + ds.snow", "ds.get('rain') + ds['snow']", "ds.RAIN * 1"])
def test_missing_attribute_or_get_read_is_a_missing_input(expression):
    from openbench.data.compute import MissingComputeVariable

    ds = xr.Dataset({"Rain": ("t", [1.0])})
    if expression == "ds.RAIN * 1":
        np.testing.assert_allclose(execute_compute(ds, expression, "X").values, [1.0])
        return
    with pytest.raises(MissingComputeVariable) as caught:
        execute_compute(ds, expression, "X")
    assert set(caught.value.dependencies) == {"rain", "snow"}


def test_dataset_proxy_get_default_and_hasattr_semantics():
    from openbench.data.compute import _CaseInsensitiveDatasetProxy

    proxy = _CaseInsensitiveDatasetProxy(xr.Dataset({"Rain": ("t", [1.0])}))
    assert float(proxy.get("rain").values[0]) == 1.0
    assert proxy.get("snow", 0) == 0
    assert not hasattr(proxy, "snow")
    assert hasattr(proxy, "dims")


def test_sum_prefix_refuses_two_names_for_one_part():
    from openbench.data.compute import ComputeIntegrityError

    ds = xr.Dataset({name: ("t", [1.0]) for name in ("f_sedcon_1", "f_sedcon_01", "f_sedcon_2")})
    with pytest.raises(ComputeIntegrityError, match="both part 1"):
        execute_compute(ds, "ds.sum_prefix('f_sedcon_', 2)", "X")


@pytest.mark.parametrize(
    "expression",
    [
        "ds['a'] + ds['b']",
        "ds.a * 2",
        "ds.get('a', 0)",
        "ds.get('a', default=0)",
        "ds.sum_prefix('x_', 2)",
        "ds['a'] if 'a' in ds else ds['b']",
        "ds['SMOIS'].isel(z=0) if 'z' in ds.dims else ds['SMOIS']",
        "total = ds['a'];\n    total * 2",
    ],
)
def test_listed_reads_make_the_inputs_known(expression):
    from openbench.data.compute import compute_inputs_known

    assert compute_inputs_known(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "key = 'runoff'; ds[key] + ds['missing']",
        "ds[['runoff']]['runoff'] + ds['missing']",
        "ds.data_vars['runoff'] + ds['missing']",
        "ds.variables['runoff'] + ds['missing']",
        "ds['run' + 'off'] + ds['missing']",
        "d = ds; d['runoff'] + ds['missing']",
        "ds['runoff'] +",
    ],
)
def test_unlisted_reads_leave_the_inputs_unknown(expression):
    from openbench.data.compute import MissingComputeVariable, compute_inputs_known

    assert not compute_inputs_known(expression)
    if expression.endswith("+"):
        return
    ds = xr.Dataset({"runoff": ("t", [2.0, 3.0])})
    with pytest.raises(MissingComputeVariable) as caught:
        execute_compute(ds, expression, "Runoff")
    assert caught.value.may_read("runoff")


def test_bundled_compute_expressions_list_all_their_inputs():
    from pathlib import Path

    import yaml

    from openbench.data.compute import compute_inputs_known

    registry = Path(__file__).resolve().parents[1] / "src" / "openbench" / "data" / "registry"
    unknown = []
    for filename in ("model_catalog.yaml", "reference_catalog.yaml"):
        for name, entry in (yaml.safe_load((registry / filename).read_text(encoding="utf-8")) or {}).items():
            for variable, mapping in ((entry or {}).get("variables") or {}).items():
                expression = mapping.get("compute") if isinstance(mapping, dict) else None
                if expression and not compute_inputs_known(expression):
                    unknown.append(f"{name}.{variable}")
    assert not unknown


def test_get_accepts_default_as_a_keyword():
    ds = xr.Dataset({"runoff": ("t", [2.0, 3.0])})

    result = execute_compute(ds, "ds.get('absent', default=ds['runoff'])", "Runoff")

    np.testing.assert_allclose(result.values, [2.0, 3.0])


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("ds['rain'] + ds['snow']", ["rain", "snow"]),
        ("ds.sum_prefix('f_sedcon_', 2)", ["f_sedcon_1", "f_sedcon_2"]),
        ("ds.get('rain') + ds.get(key='snow')", ["rain", "snow"]),
        ("ds['rain'] if 0 > 1 > ds['snow'] else ds['rain']", []),
        ("ds['rain'] if 'rain' in ds else ds['snow']", []),
        ("ds.get('snow', default=ds['rain'])", ["rain"]),
        ("'snow' in ds and ds['snow']", []),
        ("value = ds.rain; value + ds['snow']", ["rain", "snow"]),
    ],
)
def test_required_compute_inputs_exclude_optional_reads(expression, expected):
    from openbench.data.compute import compute_required_inputs

    assert compute_required_inputs(expression)[0] == expected
