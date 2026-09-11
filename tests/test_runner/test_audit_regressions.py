"""Real NetCDF regressions for source reuse and per-pair cache artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.config import adapter
from openbench.config.schema import EvaluationConfig, OpenBenchConfig, ProjectConfig, ReferenceConfig, SimulationEntry
from openbench.runner import local


def _case(tmp_path, monkeypatch, references, *, alignment="per_pair", masked=False):
    monkeypatch.chdir(tmp_path)
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            lat_range=[0, 2],
            lon_range=[0, 2],
            tim_res="Month",
            grid_res=1,
            num_cores=1,
            time_alignment=alignment,
            unified_mask=masked,
            generate_report=False,
        ),
        evaluation=EvaluationConfig(["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": list(references)}),
        simulation={"Sim": SimulationEntry(model="custom", root_dir=str(tmp_path / "Sim"))},
        metrics=["bias"],
        scores=[],
    )
    specs = {**references, "Sim": (2000, 2001)}
    sections = {}
    for name, (start, end) in specs.items():
        directory = tmp_path / name
        directory.mkdir()
        for year in range(start, end + 1):
            values = np.ones((12, 2, 2)) * (3 if name == "Long" and year == 2001 else 1)
            if masked:
                values[:] = np.arange(1.0, 13.0)[:, None, None]
                if name == "Sim":
                    values[6:] = np.nan
            xr.Dataset(
                {"v": (("time", "lat", "lon"), values, {"units": "mm day-1"})},
                coords={
                    "time": pd.date_range(f"{year}-01-01", periods=12, freq="MS"),
                    "lat": [0.5, 1.5],
                    "lon": [0.5, 1.5],
                },
            ).to_netcdf(directory / f"{year}.nc")
        fields = dict(
            data_type="grid",
            varname="v",
            varunit="mm day-1",
            data_groupby="Year",
            dir=str(directory),
            tim_res="Month",
            grid_res=1,
            syear=start,
            eyear=end,
            prefix="",
            suffix="",
            timezone=0,
            model="custom",
        )
        sections.update({f"{name}_{key}": value for key, value in fields.items()})
    runner_cfg = adapter.build_runner_config(cfg)
    bindings = adapter.RunnerBindings(
        runner_cfg,
        adapter.LegacyNamelists(
            {"general": runner_cfg.general, "evaluation_items": {"Runoff": True}},
            {"general": {"Runoff_ref_source": list(references)}, "Runoff": sections},
            {"general": {"Runoff_sim_source": ["Sim"]}, "Runoff": sections},
        ),
        adapter.LegacyFigureConfig({}),
    )
    monkeypatch.setattr(adapter, "build_runner_bindings", lambda _cfg: bindings)
    monkeypatch.setattr("openbench.runner.config_preflight.build_runner_bindings", lambda _cfg: bindings)
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_grid", lambda _evaluator: None)
    return cfg, bindings, tmp_path / "case"


@pytest.mark.parametrize("alignment", ["intersection", "per_pair"])
@pytest.mark.parametrize("short_first", [True, False])
@pytest.mark.parametrize("unified_mask", [False, True])
def test_multireference_years_do_not_depend_on_reference_order(
    tmp_path, monkeypatch, alignment, short_first, unified_mask
):
    references = {"Short": (2000, 2000), "Long": (2000, 2001)}
    if not short_first:
        references = dict(reversed(list(references.items())))
    cfg, bindings, case = _case(tmp_path, monkeypatch, references, alignment=alignment)
    cfg.project.unified_mask = unified_mask
    bindings.runner_cfg.general["unified_mask"] = unified_mask
    result = local._run_evaluation_impl(cfg, dask_distributed_active=False)
    assert result["status"] == "success", result["errors"]
    with xr.open_dataset(case / "data/Runoff_sim_Sim_v.nc") as sim:
        assert sim.sizes["time"] == 24
    with xr.open_dataset(case / "metrics/Runoff_ref_Long_sim_Sim_bias.nc") as result:
        np.testing.assert_allclose(result.bias, -1)


def test_per_pair_outputs_survive_cache_reuse_and_comparison_only(tmp_path, monkeypatch):
    cfg, bindings, case = _case(tmp_path, monkeypatch, {"Ref": (2000, 2001)}, masked=True)
    cfg.comparison.enabled = True
    cfg.comparison.items = ["Mean"]
    bindings.runner_cfg.comparisons[:] = ["Mean"]
    monkeypatch.setattr("openbench.core.comparison.make_geo_plot_index", lambda *args, **kwargs: None)
    pair = case / "data/Runoff_ref_Ref_Sim_v.nc"
    mean_path = case / "comparisons/Mean/Runoff_ref_Ref_sim_Sim_v_Mean.nc"
    for run_index, comparison_only in enumerate([False, False, True]):
        result = local._run_evaluation_impl(cfg, comparison_only=comparison_only, dask_distributed_active=False)
        assert result["status"] == "success", result["errors"]
        assert pair.is_file()
        if run_index == 1:
            assert all(task["skipped"] for task in result["evaluated"])
        with xr.open_dataset(mean_path) as means:
            np.testing.assert_allclose(means.Mean, 3.5)


def test_missing_pair_ref_invalidates_cache_and_output_only_preflight(tmp_path, monkeypatch):
    cfg, bindings, case = _case(tmp_path, monkeypatch, {"Ref": (2000, 2001)}, masked=True)
    result = local._run_evaluation_impl(cfg, dask_distributed_active=False)
    assert result["status"] == "success", result["errors"]
    pair = case / "data/Runoff_ref_Ref_Sim_v.nc"
    pair.unlink(missing_ok=True)
    assert local.existing_output_preflight_errors(cfg)
    result = local._run_evaluation_impl(cfg, dask_distributed_active=False)
    assert result["status"] == "success", result["errors"]
    assert not any(task["skipped"] for task in result["evaluated"])
    assert pair.is_file()
    tasks = local._build_evaluation_tasks(
        cfg=cfg,
        bindings=bindings,
        output_dir=case,
        metric_vars=["bias"],
        score_vars=[],
        comparison_vars=[],
        statistic_vars=[],
        use_cache=False,
        only_drawing=True,
    )
    assert Path(local._build_bridge_runtime_info(tasks[0])["ref_file_override"]) == pair


def test_masked_comparison_does_not_fall_back_to_unmasked_reference(tmp_path):
    from openbench.core._comparison_common import CommonComparisonMixin

    comparison = CommonComparisonMixin()
    comparison.time_alignment = "per_pair"
    comparison.unified_mask = True
    with pytest.raises(FileNotFoundError, match="[Pp]air"):
        comparison._ref_data_path(str(tmp_path), "Runoff", "Ref", "v", "Sim")
    comparison.unified_mask = False
    assert comparison._ref_data_path(str(tmp_path), "Runoff", "Ref", "v", "Sim").endswith("Runoff_ref_Ref_v.nc")


@pytest.mark.parametrize("comparison_only,force", [(False, False), (True, False), (False, True)])
def test_failed_post_phase_keeps_successful_pair_ref(tmp_path, monkeypatch, comparison_only, force):
    cfg, _bindings, case = _case(tmp_path, monkeypatch, {"Ref": (2000, 2001)}, masked=True)
    cfg.project.IGBP_groupby = True
    monkeypatch.setattr(local, "_run_groupby", lambda *args, **kwargs: None)
    assert local._run_evaluation_impl(cfg, dask_distributed_active=False)["status"] == "success"
    pair = case / "data/Runoff_ref_Ref_Sim_v.nc"
    original = pair.read_bytes()

    def failed_groupby(*args, **kwargs):
        raise RuntimeError("plot failure")

    monkeypatch.setattr(local, "_run_groupby", failed_groupby)
    result = local._run_evaluation_impl(
        cfg, comparison_only=comparison_only, force=force, dask_distributed_active=False
    )
    assert result["errors"]
    assert pair.read_bytes() == original
