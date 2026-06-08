from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner


def test_km_per_hour_converts_to_meters_per_second():
    import openbench.data.unit as unit_mod
    from openbench.data.unit import UnitProcessing

    unit_mod._UNIT_LOOKUP_CACHE = None
    converted, base_unit = UnitProcessing.convert_unit(np.array([3.6]), "km h-1")

    assert base_unit == "m s-1"
    assert converted == pytest.approx(np.array([1.0]))


def test_process_units_preserves_time_for_calendar_aware_month_conversion():
    from openbench.data._processing_transforms import ProcessingTransformMixin

    class Processor(ProcessingTransformMixin):
        pass

    ds = xr.Dataset(
        {"precip": ("time", [29.0])},
        coords={"time": pd.DatetimeIndex(["2020-02-15"])},
    )

    converted, new_unit = Processor().process_units(ds, "mm month-1")

    assert new_unit == "mm day-1"
    assert float(converted["precip"].isel(time=0)) == pytest.approx(1.0)
    assert "time" in converted["precip"].coords


def test_model_catalog_resolution_time_offset_is_applied():
    from openbench.data._processing_time_adjustments import TimeAdjustmentMixin

    class Processor(TimeAdjustmentMixin):
        sim_source = "BCC_AVIM"
        sim_prefix = ""
        sim_dir = ""
        item = "Streamflow"

    ds = xr.Dataset(
        {"q": ("time", [1.0])},
        coords={"time": pd.DatetimeIndex(["2001-01-02T00:00:00"])},
    )

    adjusted = Processor().apply_model_specific_time_adjustment(ds, "sim", 2001, 2001, "Day")

    assert pd.Timestamp(adjusted.time.values[0]) == pd.Timestamp("2001-01-01T00:30:00")


def test_reference_builder_preserves_compute_and_prefix_fallback():
    from openbench.data.registry.manager import _build_reference

    ref = _build_reference(
        {
            "name": "ExampleRef",
            "description": "example",
            "data_type": "grid",
            "tim_res": "Month",
            "variables": {
                "Runoff": {
                    "varname": "runoff",
                    "varunit": "mm day-1",
                    "compute": "ds['surface'] + ds['subsurface']",
                    "prefix_fallback": ["alt_"],
                }
            },
        }
    )

    mapping = ref.variables["Runoff"]
    assert mapping.compute == "ds['surface'] + ds['subsurface']"
    assert mapping.prefix_fallback == ["alt_"]


def test_cache_status_prints_resolved_regrid_cache_directory(tmp_path):
    from openbench.cli.cache import cache

    result = CliRunner().invoke(cache, ["status", "--regrid", "--dir", str(tmp_path)])

    assert result.exit_code == 0
    assert f"Regrid weight cache: {tmp_path}" in result.output
    assert "OPENBENCH_REGRID_WEIGHT_CACHE_DIR" not in result.output


def test_cp_uses_same_pairwise_nan_mask_for_numerator_and_denominator():
    from openbench.core.metrics import metrics

    obs = xr.DataArray([np.nan, 2.0, 3.0], dims="time", coords={"time": pd.date_range("2000-01-01", periods=3)})
    sim = xr.DataArray([0.0, 10.0, 3.0], dims="time", coords={"time": pd.date_range("2000-01-01", periods=3)})

    value = metrics().cp(sim, obs)

    assert float(value) == pytest.approx(1.0)


def test_station_csv_and_pair_ref_regressions_are_source_guarded():
    config_source = Path("src/openbench/data/_processing_config.py").read_text()
    evaluation_source = Path("src/openbench/core/evaluation.py").read_text()
    preprocessing_source = Path("src/openbench/runner/preprocessing.py").read_text()

    assert "write_file_atomic(" in config_source
    assert "_write_file_atomic(" in evaluation_source
    assert "if not os.path.exists(pair_ref):" not in preprocessing_source


def test_water_depth_units_do_not_collapse_to_annual_rate():
    import openbench.data.unit as unit_mod
    from openbench.data.unit import UnitProcessing

    unit_mod._UNIT_LOOKUP_CACHE = None
    converted, base_unit = UnitProcessing.convert_unit(np.array([2.0]), "kg m-2")
    assert base_unit == "mm"
    assert converted == pytest.approx(np.array([2.0]))

    converted_rate, rate_unit = UnitProcessing.convert_unit(np.array([1.0]), "kg/m2/s")
    assert rate_unit == "mm day-1"
    assert converted_rate == pytest.approx(np.array([86400.0]))


def test_taylor_summary_uses_geometrically_consistent_crmsd_for_diagram():
    from openbench.core._comparison_taylor import _taylor_summary_statistics

    summary = _taylor_summary_statistics(
        std_sim=4.0,
        cor_sim=0.5,
        mean_crmsd=10.0,
        std_ref=2.0,
    )

    expected_diagram_crmsd = np.sqrt(4.0**2 + 2.0**2 - 2.0 * 4.0 * 2.0 * 0.5)
    assert summary.diagram_crmsd == pytest.approx(expected_diagram_crmsd)
    assert summary.mean_crmsd == pytest.approx(10.0)


def test_glob_nc_matches_uppercase_extensions(tmp_path):
    from openbench.data.coordinates import glob_nc, nc_exists

    upper = tmp_path / "CASE.NC4"
    upper.write_text("not really netcdf")

    assert glob_nc(tmp_path) == [upper]
    assert nc_exists(str(tmp_path / "CASE.nc")) == str(upper)


def test_ubkge_uses_two_component_unbiased_form():
    from openbench.core.metrics import metrics

    time = pd.date_range("2000-01-01", periods=3)
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": time})
    sim = xr.DataArray([2.0, 4.0, 6.0], dims="time", coords={"time": time})

    assert float(metrics().ubKGE(sim, obs)) == pytest.approx(0.0)


def test_accumulated_precipitation_resamples_with_sum():
    from openbench.data._processing_time_core import TimeCoreMixin

    class Processor(TimeCoreMixin):
        item = "Precipitation"
        compare_tim_res = "2D"

    data = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=4, freq="D")},
        attrs={"units": "mm"},
    )

    result = Processor()._resample_to_compare_resolution(data, "test precip")

    np.testing.assert_allclose(result.values, [3.0, 7.0])


def test_unified_mask_non_strict_uses_overlapping_times(tmp_path):
    from openbench.runner.masking import apply_unified_mask
    from openbench.util.netcdf import write_netcdf_atomic

    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    ref_path = data_dir / "Runoff_ref_TestRef_runoff_ref.nc"
    sim_path = data_dir / "Runoff_sim_SimA_runoff_sim.nc"

    xr.Dataset(
        {"runoff_ref": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={"time": pd.date_range("2001-01-01", periods=3, freq="D"), "lat": [0.0], "lon": [10.0]},
    ).to_netcdf(ref_path)
    xr.Dataset(
        {"runoff_sim": (("time", "lat", "lon"), np.array([[[np.nan]], [[1.0]]]))},
        coords={"time": pd.date_range("2001-01-02", periods=2, freq="D"), "lat": [0.0], "lon": [10.0]},
    ).to_netcdf(sim_path)

    apply_unified_mask(
        {
            "casedir": str(case_dir),
            "ref_varname": "runoff_ref",
            "sim_varname": "runoff_sim",
            "time_alignment": "intersection",
        },
        "Runoff",
        "TestRef",
        "SimA",
        write_netcdf_atomic_fn=write_netcdf_atomic,
    )

    with xr.open_dataset(ref_path) as ds:
        values = ds["runoff_ref"].values[:, 0, 0]
    assert values[0] == pytest.approx(1.0)
    assert np.isnan(values[1])
    assert values[2] == pytest.approx(1.0)


def test_unified_mask_non_strict_write_failure_is_not_success(tmp_path):
    from openbench.runner.masking import apply_unified_mask

    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    ref_path = data_dir / "Runoff_ref_TestRef_runoff_ref.nc"
    sim_path = data_dir / "Runoff_sim_SimA_runoff_sim.nc"

    xr.Dataset(
        {"runoff_ref": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": pd.date_range("2001-01-01", periods=2, freq="D"), "lat": [0.0], "lon": [10.0]},
    ).to_netcdf(ref_path)
    xr.Dataset(
        {"runoff_sim": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"time": pd.date_range("2001-01-01", periods=2, freq="D"), "lat": [0.0], "lon": [10.0]},
    ).to_netcdf(sim_path)

    def broken_writer(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    with pytest.raises(RuntimeError, match="simulated write failure"):
        apply_unified_mask(
            {
                "casedir": str(case_dir),
                "ref_varname": "runoff_ref",
                "sim_varname": "runoff_sim",
                "time_alignment": "intersection",
            },
            "Runoff",
            "TestRef",
            "SimA",
            write_netcdf_atomic_fn=broken_writer,
        )


def test_input_file_signature_falls_back_to_recursive_netcdf_scan(tmp_path):
    from openbench.runner.hashing import input_file_signature

    nested = tmp_path / "2001" / "region"
    nested.mkdir(parents=True)
    nc_path = nested / "unmatched_name.NC"
    nc_path.write_bytes(b"netcdf-ish")

    signature = input_file_signature(
        {"Case_dir": str(tmp_path), "Case_prefix": "prefix", "Case_varname": "missing", "Case_suffix": "suffix"},
        "Case",
    )

    assert [Path(item["path"]).name for item in signature["files"]] == ["unmatched_name.NC"]


def test_regrid_hash_ignores_unselected_backend_environment(tmp_path):
    import openbench.config.adapter as adapter
    from openbench.config.schema import (
        EvaluationConfig,
        OpenBenchConfig,
        ProjectConfig,
        ReferenceConfig,
        SimulationEntry,
    )
    from openbench.runner.cache import EvaluationCache
    from openbench.runner.hashing import task_hash_payload

    cfg = OpenBenchConfig(
        project=ProjectConfig(name="case", output_dir=str(tmp_path), years=[2000, 2001]),
        reference=ReferenceConfig(data_root=str(tmp_path), sources={"Runoff": "TestRef"}),
        simulation={"SimA": SimulationEntry(root_dir=str(tmp_path), model="ModelA")},
        evaluation=EvaluationConfig(variables={"Runoff": {"metrics": ["bias"], "scores": ["Overall_Score"]}}),
    )
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={
            "syear": 2000,
            "eyear": 2001,
            "min_lat": -90,
            "max_lat": 90,
            "min_lon": -180,
            "max_lon": 180,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "compare_tzone": 0,
            "weight": "area",
            "unified_mask": False,
            "time_alignment": "intersection",
            "only_drawing": False,
            "regrid_backend": "openbench_conservative",
        },
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={"Runoff": {"SimA_varname": "runoff_sim"}},
            ),
        },
    )()

    def digest(signature):
        return EvaluationCache.hash_config(
            task_hash_payload(
                cfg=cfg,
                bindings=bindings,
                var_name="Runoff",
                sim_source="SimA",
                ref_source="TestRef",
                metric_vars=["bias"],
                score_vars=["Overall_Score"],
                comparison_vars=[],
                statistic_vars=[],
                regrid_backend_signature_fn=lambda: signature,
            )
        )

    first = digest({"openbench_conservative": True, "cdo": {"path": "/a/cdo"}, "scipy": {"version": "1"}})
    second = digest({"openbench_conservative": True, "cdo": {"path": "/b/cdo"}, "scipy": {"version": "2"}})
    assert first == second


def test_task_hash_payload_includes_algorithm_source_fingerprint(tmp_path):
    import openbench.config.adapter as adapter
    from openbench.config.schema import (
        EvaluationConfig,
        OpenBenchConfig,
        ProjectConfig,
        ReferenceConfig,
        SimulationEntry,
    )
    from openbench.runner.hashing import algorithm_source_fingerprint, task_hash_payload

    cfg = OpenBenchConfig(
        project=ProjectConfig(name="case", output_dir=str(tmp_path), years=[2000, 2001]),
        reference=ReferenceConfig(data_root=str(tmp_path), sources={"Runoff": "TestRef"}),
        simulation={"SimA": SimulationEntry(root_dir=str(tmp_path), model="ModelA")},
        evaluation=EvaluationConfig(variables={"Runoff": {"metrics": ["bias"], "scores": ["Overall_Score"]}}),
    )
    runner_cfg = adapter.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Runoff": True},
        metrics=["bias"],
        scores=["Overall_Score"],
        comparisons=[],
        statistics=[],
        general={"syear": 2000, "eyear": 2001, "regrid_backend": "openbench_conservative"},
    )
    bindings = type(
        "Bindings",
        (),
        {
            "runner_cfg": runner_cfg,
            "namelists": adapter.LegacyNamelists(
                main={"general": runner_cfg.general},
                reference={"Runoff": {"TestRef_varname": "runoff_ref"}},
                simulation={"Runoff": {"SimA_varname": "runoff_sim"}},
            ),
        },
    )()

    payload = task_hash_payload(
        cfg=cfg,
        bindings=bindings,
        var_name="Runoff",
        sim_source="SimA",
        ref_source="TestRef",
        metric_vars=["bias"],
        score_vars=["Overall_Score"],
        comparison_vars=[],
        statistic_vars=[],
    )

    assert payload["openbench"]["source_fingerprint"] == algorithm_source_fingerprint()


def test_taylor_diagram_uses_population_std_consistent_with_crmsd():
    from openbench.core._comparison_taylor import _taylor_standard_deviation

    values = xr.DataArray([1.0, 2.0], dims="time", coords={"time": pd.date_range("2001-01-01", periods=2)})

    assert float(_taylor_standard_deviation(values)) == pytest.approx(0.5)


def test_comparison_smpi_masks_zero_variance_without_clipping_large_values():
    from openbench.core._comparison_smpi import _smpi_normalized_diff

    times = pd.date_range("2001-01-01", periods=3)
    obs = xr.DataArray(
        np.array([[1.0, 0.0], [1.0, 10.0], [1.0, 20.0]]),
        dims=("time", "site"),
        coords={"time": times, "site": ["constant", "variable"]},
    )
    sim = xr.DataArray(
        np.array([[2.0, 500.0], [2.0, 510.0], [2.0, 520.0]]),
        dims=("time", "site"),
        coords={"time": times, "site": ["constant", "variable"]},
    )

    normalized = _smpi_normalized_diff(sim, obs)

    assert np.isnan(float(normalized.sel(site="constant")))
    assert float(normalized.sel(site="variable")) > 100.0


def test_climatology_weighting_honors_explicit_daily_resolution_over_delta_inference():
    from openbench.data.climatology import ClimatologyProcessor

    processor = ClimatologyProcessor()
    times = pd.to_datetime(["2001-01-15", "2001-02-15"])
    ds = xr.Dataset({"v": ("time", [31.0, 0.0])}, coords={"time": times})

    inferred = processor.prepare_reference_climatology(ds, processor.ANNUAL_CLIMATOLOGY, 2001)
    explicit_daily = processor.prepare_reference_climatology(
        ds,
        processor.ANNUAL_CLIMATOLOGY,
        2001,
        source_tim_res="Day",
    )

    assert float(inferred["v"].isel(time=0)) != pytest.approx(15.5)
    assert float(explicit_daily["v"].isel(time=0)) == pytest.approx(15.5)


def test_vertical_coordinate_aliases_are_centralized():
    from openbench.data.coordinates import COORDINATE_MAP_WITH_VERTICAL, VERTICAL_COORDINATE_MAP

    assert COORDINATE_MAP_WITH_VERTICAL["elevation"] == "elev"
    assert COORDINATE_MAP_WITH_VERTICAL["height"] == "elev"
    assert VERTICAL_COORDINATE_MAP["altitude"] == "elev"

    for path in [
        "src/openbench/core/comparison.py",
        "src/openbench/data/_processing_base.py",
        "src/openbench/visualization/only_drawing.py",
        "src/openbench/core/statistics/Mod_Statistics.py",
    ]:
        source = Path(path).read_text()
        assert "COORDINATE_MAP_WITH_VERTICAL" in source
        assert '"elevation": "elev"' not in source
