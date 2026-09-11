import numpy as np
import pytest
import xarray as xr

from openbench.config import adapter as adapter_module
from openbench.core._comparison_helpers import _apply_pairwise_valid_mask
from openbench.core.statistics.Mod_Statistics import BasicProcessing
from openbench.core.statistics.stat_partial_least_squares_regression import stat_partial_least_squares_regression


def _runner_bindings(*, simulations=("SimA", "SimB")):
    general = {
        "basename": "case",
        "basedir": ".",
        "compare_tim_res": "Month",
        "compare_grid_res": 0.5,
        "compare_tzone": 0,
        "num_cores": 1,
        "syear": 2000,
        "eyear": 2000,
        "time_alignment": "intersection",
        "unified_mask": True,
    }
    sim_section = {
        "general": {"Tair_sim_source": list(simulations)},
        "Tair": {},
    }
    for sim in simulations:
        sim_section["Tair"].update(
            {
                f"{sim}_data_type": "grid",
                f"{sim}_varname": "tas_sim",
                f"{sim}_varunit": "K",
            }
        )
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"Tair": True},
        metrics=[],
        scores=[],
        comparisons=[],
        statistics=[],
        general=general,
    )
    return adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": general},
            reference={
                "general": {"Tair_ref_source": "RefA"},
                "Tair": {
                    "RefA_data_type": "grid",
                    "RefA_varname": "tas_ref",
                    "RefA_varunit": "K",
                },
            },
            simulation=sim_section,
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )


def test_statistics_loader_uses_preprocessed_dataset_units_not_original_config_unit():
    proc = BasicProcessing.__new__(BasicProcessing)
    proc.check_coordinate = lambda ds: ds
    proc.check_dataset_time_integrity = lambda ds, *_args: ds
    proc.select_timerange = lambda ds, *_args: ds
    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), [[[273.15]]], {"units": "K"})},
        coords={"time": [0], "lat": [0.0], "lon": [0.0]},
        attrs={"units": "K"},
    )

    out = proc.load_and_process_dataset(ds, 2000, 2000, "Month", "c")

    assert float(out["tas"].values[0, 0, 0]) == pytest.approx(273.15)
    assert out["tas"].attrs["units"].lower() == "k"


def test_advanced_statistics_context_matches_runtime_source_contracts():
    ctx = _runner_bindings().build_statistics_context(
        ["ANOVA", "Partial_Least_Squares_Regression", "Three_Cornered_Hat"], ["Tair"]
    )

    anova = ctx.stats_nml["ANOVA"]
    assert ctx.stats_nml["general"]["ANOVA_data_source"] == "Tair_SimA,Tair_SimB"
    assert anova["Tair_SimA_Y_prefix"] == "Tair_ref_RefA_tas_ref"
    assert anova["Tair_SimA_X_prefix"] == "Tair_sim_SimA_tas_sim"

    plsr = ctx.stats_nml["Partial_Least_Squares_Regression"]
    assert ctx.stats_nml["general"]["Partial_Least_Squares_Regression_data_source"] == "Tair_SimA,Tair_SimB"
    assert plsr["Tair_SimB_nX"] == 1
    assert plsr["Tair_SimB_Y_prefix"] == "Tair_ref_RefA_tas_ref"
    assert plsr["Tair_SimB_X1_prefix"] == "Tair_sim_SimB_tas_sim"

    tch = ctx.stats_nml["Three_Cornered_Hat"]
    assert ctx.stats_nml["general"]["Three_Cornered_Hat_data_source"] == "Tair_RefA"
    assert tch["Tair_RefA_nX"] == 3
    assert tch["Tair_RefA1_prefix"] == "Tair_sim_SimA_tas_sim"
    assert tch["Tair_RefA2_prefix"] == "Tair_sim_SimB_tas_sim"
    assert tch["Tair_RefA3_prefix"] == "Tair_ref_RefA_tas_ref"


def test_three_cornered_hat_context_with_too_few_sources_still_sets_nx_for_runtime_error():
    ctx = _runner_bindings(simulations=("SimA",)).build_statistics_context(["Three_Cornered_Hat"], ["Tair"])

    assert ctx.stats_nml["general"]["Three_Cornered_Hat_data_source"] == "Tair_RefA"
    assert ctx.stats_nml["Three_Cornered_Hat"]["Tair_RefA_nX"] == 2


def test_three_cornered_hat_per_pair_context_uses_masked_pair_refs():
    bindings = _runner_bindings()
    general = dict(bindings.runner_cfg.general, time_alignment="per_pair")
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=".",
        evaluation_items={"Tair": True},
        metrics=[],
        scores=[],
        comparisons=[],
        statistics=[],
        general=general,
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": general},
            reference=bindings.namelists.reference,
            simulation=bindings.namelists.simulation,
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )

    ctx = bindings.build_statistics_context(["Three_Cornered_Hat"], ["Tair"])
    tch = ctx.stats_nml["Three_Cornered_Hat"]

    assert tch["Tair_RefA_nX"] == 3
    assert tch["Tair_RefA3_prefix"] == "Tair_ref_RefA_SimA_tas_ref"


def test_comparison_pairwise_valid_mask_rejects_inf_pairs():
    sim = xr.DataArray([1.0, np.inf, 3.0], dims=["time"])
    ref = xr.DataArray([1.0, 2.0, np.inf], dims=["time"])

    masked_sim, masked_ref = _apply_pairwise_valid_mask(sim, ref)

    assert np.isfinite(masked_sim.values[0])
    assert np.isnan(masked_sim.values[1])
    assert np.isnan(masked_ref.values[2])


class _PLSRSelf:
    stats_nml = {"Partial_Least_Squares_Regression": {"max_components": 1, "n_splits": 2, "n_jobs": 1}}


def _cube(values, times):
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(len(values), 1, 1),
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )


def test_public_plsr_aligns_inputs_by_time_before_using_values():
    series = np.array([0.0, 2.0, -1.0, 3.0, 1.0, -2.0, 4.0, -3.0, 5.0])
    y = _cube(series[:-1], np.arange(8))
    x = _cube(series[1:], np.arange(1, 9))

    result = stat_partial_least_squares_regression(_PLSRSelf(), y, x)

    assert float(result["r_squared"].values[0, 0]) == pytest.approx(1.0)


def _runtime_bindings(tmp_path, *, simulations=("SimA", "SimB"), refs=("RefA",)):
    bindings = _runner_bindings(simulations=simulations)
    general = dict(
        bindings.runner_cfg.general,
        basedir=str(tmp_path),
        min_lon=-0.5,
        max_lon=0.5,
        min_lat=-0.5,
        max_lat=0.5,
    )
    reference = {"general": {"Tair_ref_source": list(refs) if len(refs) > 1 else refs[0]}, "Tair": {}}
    for ref in refs:
        reference["Tair"].update(
            {
                f"{ref}_data_type": "grid",
                f"{ref}_varname": f"tas_{ref.lower()}",
                f"{ref}_varunit": "K",
            }
        )
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Tair": True},
        metrics=[],
        scores=[],
        comparisons=[],
        statistics=[],
        general=general,
    )
    return adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": general},
            reference=reference,
            simulation=bindings.namelists.simulation,
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )


def _write_cube(path, varname, values):
    times = np.array([f"2000-{month:02d}-29" for month in range(1, 13)], dtype="datetime64[D]")
    data = np.asarray(values, dtype=float).reshape(12, 1, 1)
    ds = xr.Dataset(
        {varname: (("time", "lat", "lon"), data, {"units": "K"})},
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
        attrs={"units": "K"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _statistics_processor(ctx):
    from openbench.core.statistics.Mod_Statistics import StatisticsProcessing

    proc = StatisticsProcessing(ctx.namelists.main, ctx.stats_nml, ctx.stats_dir, num_cores=1)
    proc.remap_data = lambda data_list: data_list
    return proc


def test_advanced_statistics_context_runs_anova_plsr_and_tch(tmp_path):
    bindings = _runtime_bindings(tmp_path, simulations=("SimA", "SimB"), refs=("RefA",))
    ctx = bindings.build_statistics_context(
        ["ANOVA", "Partial_Least_Squares_Regression", "Three_Cornered_Hat"], ["Tair"]
    )
    data_dir = tmp_path / "case" / "data"
    y = np.array([0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6], dtype=float) + 273.15
    x1 = np.arange(12, dtype=float) + 273.15
    x2 = y + np.array([0.05, -0.05, 0.04, -0.04, 0.03, -0.03, 0.02, -0.02, 0.01, -0.01, 0.0, 0.0])
    _write_cube(data_dir / "Tair_ref_RefA_tas_refa.nc", "tas_refa", y)
    _write_cube(data_dir / "Tair_sim_SimA_tas_sim.nc", "tas_sim", x1)
    _write_cube(data_dir / "Tair_sim_SimB_tas_sim.nc", "tas_sim", x2)

    proc = _statistics_processor(ctx)

    anova_out = proc.run_analysis("Tair_SimA", ["Tair_SimA_Y", "Tair_SimA_X"], "ANOVA")
    with xr.open_dataset(anova_out) as ds:
        assert np.isfinite(ds["F_statistic"].values).any()
        assert np.isfinite(ds["p_value"].values).any()

    plsr_out = proc.run_analysis("Tair_SimB", ["Tair_SimB_Y", "Tair_SimB_X1"], "Partial_Least_Squares_Regression")
    with xr.open_dataset(plsr_out) as ds:
        assert np.isfinite(ds["r_squared"].values).any()
        assert float(ds["r_squared"].values[0, 0]) > 0.9

    e1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    e2 = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    e3 = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    _write_cube(data_dir / "Tair_ref_RefA_tas_refa.nc", "tas_refa", np.full(12, 280.0) + e3)
    _write_cube(data_dir / "Tair_sim_SimA_tas_sim.nc", "tas_sim", np.full(12, 280.0) + e1)
    _write_cube(data_dir / "Tair_sim_SimB_tas_sim.nc", "tas_sim", np.full(12, 280.0) + e2)

    tch_out = proc.run_analysis("Tair_RefA", ["Tair_RefA1", "Tair_RefA2", "Tair_RefA3"], "Three_Cornered_Hat")
    with xr.open_dataset(tch_out) as ds:
        assert ds.attrs["n_datasets"] == 3
        assert np.isfinite(ds["uncertainty"].values).any()


def test_advanced_tch_multiref_context_runs_each_ref_group(tmp_path):
    bindings = _runtime_bindings(tmp_path, simulations=("SimA", "SimB"), refs=("RefA", "RefB"))
    ctx = bindings.build_statistics_context(["Three_Cornered_Hat"], ["Tair"])
    data_dir = tmp_path / "case" / "data"
    base = np.full(12, 280.0)
    e1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    e2 = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    e3 = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    _write_cube(data_dir / "Tair_sim_SimA_tas_sim.nc", "tas_sim", base + e1)
    _write_cube(data_dir / "Tair_sim_SimB_tas_sim.nc", "tas_sim", base + e2)
    _write_cube(data_dir / "Tair_ref_RefA_tas_refa.nc", "tas_refa", base + e3)
    _write_cube(data_dir / "Tair_ref_RefB_tas_refb.nc", "tas_refb", base - e3)

    proc = _statistics_processor(ctx)

    assert ctx.stats_nml["general"]["Three_Cornered_Hat_data_source"] == "Tair_RefA,Tair_RefB"
    for source in ("Tair_RefA", "Tair_RefB"):
        n_x = ctx.stats_nml["Three_Cornered_Hat"][f"{source}_nX"]
        out = proc.run_analysis(source, [f"{source}{i}" for i in range(1, n_x + 1)], "Three_Cornered_Hat")
        with xr.open_dataset(out) as ds:
            assert ds.attrs["n_datasets"] == 3
            assert np.isfinite(ds["uncertainty"].values).any()


def test_tch_runtime_error_is_clear_when_adapter_provides_too_few_sources(tmp_path):
    bindings = _runtime_bindings(tmp_path, simulations=("SimA",), refs=("RefA",))
    ctx = bindings.build_statistics_context(["Three_Cornered_Hat"], ["Tair"])
    data_dir = tmp_path / "case" / "data"
    _write_cube(data_dir / "Tair_sim_SimA_tas_sim.nc", "tas_sim", np.arange(12) + 280.0)
    _write_cube(data_dir / "Tair_ref_RefA_tas_refa.nc", "tas_refa", np.arange(12) + 281.0)
    proc = _statistics_processor(ctx)

    with pytest.raises(ValueError, match="requires at least 3 datasets; got nX=2"):
        proc.scenarios_Three_Cornered_Hat_analysis("Three_Cornered_Hat", ctx.stats_nml["Three_Cornered_Hat"], {})


def test_tch_per_pair_runtime_uses_one_ref_source_and_joint_valid_samples(tmp_path):
    bindings = _runtime_bindings(tmp_path, simulations=("SimA", "SimB"), refs=("RefA",))
    general = dict(bindings.runner_cfg.general, time_alignment="per_pair")
    runner_cfg = adapter_module.RunnerConfig(
        basename="case",
        basedir=str(tmp_path),
        evaluation_items={"Tair": True},
        metrics=[],
        scores=[],
        comparisons=[],
        statistics=[],
        general=general,
    )
    bindings = adapter_module.RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=adapter_module.LegacyNamelists(
            main={"general": general},
            reference=bindings.namelists.reference,
            simulation=bindings.namelists.simulation,
        ),
        figures=adapter_module.LegacyFigureConfig(raw={}),
    )
    ctx = bindings.build_statistics_context(["Three_Cornered_Hat"], ["Tair"])
    data_dir = tmp_path / "case" / "data"
    base = np.full(12, 280.0)
    e1 = np.array([1, -1, 1, -1, 1, -1, np.nan, -1, 1, -1, 1, -1], dtype=float)
    e2 = np.array([1, 1, -1, -1, 1, 1, -1, -1, np.nan, 1, -1, -1], dtype=float)
    e3 = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, np.nan], dtype=float)
    _write_cube(data_dir / "Tair_sim_SimA_tas_sim.nc", "tas_sim", base + e1)
    _write_cube(data_dir / "Tair_sim_SimB_tas_sim.nc", "tas_sim", base + e2)
    _write_cube(data_dir / "Tair_ref_RefA_SimA_tas_refa.nc", "tas_refa", base + e3)
    _write_cube(data_dir / "Tair_ref_RefA_SimB_tas_refa.nc", "tas_refa", base + 100.0)
    proc = _statistics_processor(ctx)

    assert ctx.stats_nml["Three_Cornered_Hat"]["Tair_RefA_nX"] == 3
    out = proc.run_analysis("Tair_RefA", ["Tair_RefA1", "Tair_RefA2", "Tair_RefA3"], "Three_Cornered_Hat")

    from openbench.core.statistics.stat_three_cornered_hat import _tch_uncertainty_from_samples

    expected, _ = _tch_uncertainty_from_samples(np.stack([base + e1, base + e2, base + e3], axis=1))
    with xr.open_dataset(out) as ds:
        np.testing.assert_allclose(ds["uncertainty"].values[:, 0, 0], expected)


class _PLSRCVSelf:
    stats_nml = {"Partial_Least_Squares_Regression": {"max_components": 3, "n_splits": 5, "n_jobs": 1}}


def test_plsr_component_candidates_respect_smallest_cv_train_fold(recwarn):
    times = np.arange(12)
    y = _cube(np.linspace(0.0, 11.0, 12), times)
    x1 = _cube(np.linspace(0.0, 11.0, 12), times)
    x2 = _cube(np.linspace(1.0, 12.0, 12), times)
    x3 = _cube(np.linspace(2.0, 13.0, 12), times)

    result = stat_partial_least_squares_regression(_PLSRCVSelf(), y, x1, x2, x3)

    assert int(result["best_n_components"].values[0, 0]) <= 2
    assert not [w for w in recwarn if w.category.__name__ == "FitFailedWarning"]
