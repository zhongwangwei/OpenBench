from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_isolated_rc_closes_new_figures_on_failure():
    """Renderer failures should not leak figures into later plotting calls."""
    import matplotlib.pyplot as plt

    from openbench.visualization._rc_isolation import with_isolated_rc

    plt.close("all")
    existing = plt.figure()

    @with_isolated_rc
    def fail_after_opening_figure():
        plt.figure()
        raise RuntimeError("figure boom")

    with pytest.raises(RuntimeError, match="figure boom"):
        fail_after_opening_figure()

    assert plt.get_fignums() == [existing.number]
    plt.close(existing)


def _parallel_coordinates_option():
    return {
        "font": "DejaVu Sans",
        "axes_linewidth": 1,
        "xticksize": 8,
        "yticksize": 8,
        "x_wise": 4,
        "y_wise": 3,
        "set_legend": False,
        "bbox_to_anchor_x": 0.5,
        "bbox_to_anchor_y": -0.15,
        "models_to_highlight_by_line": True,
        "models_to_highlight_markers_size": 22,
        "cmap": "viridis",
        "legend_off": False,
        "legend_ncol": 2,
        "legend_loc": "upper center",
        "fontsize": 8,
        "yticklabel": "",
        "xticklabel": "",
        "title": "",
        "title_size": 10,
        "saving_format": "png",
        "dpi": 80,
    }


def _kde_option():
    return {
        "font": "DejaVu Sans",
        "axes_linewidth": 1,
        "fontsize": 8,
        "xtick": 8,
        "ytick": 8,
        "line_width": 1,
        "linestyle": "",
        "linewidth": None,
        "alpha": None,
        "x_wise": 4,
        "y_wise": 3,
        "ncol": 1,
        "set_legend": False,
        "loc": "best",
        "bbox_to_anchor_x": 0.5,
        "bbox_to_anchor_y": -0.15,
        "grid": False,
        "grid_style": "--",
        "grid_linewidth": 0.5,
        "title": "",
        "xticklabel": "",
        "yticklabel": "",
        "title_fontsize": 10,
        "saving_format": "png",
        "dpi": 80,
    }


def _portrait_option():
    return {
        "font": "DejaVu Sans",
        "axes_linewidth": 1,
        "xtick": 8,
        "ytick": 8,
        "x_wise": 4,
        "y_wise": 3,
        "colorbar_label": "",
        "vmin_max_on": False,
        "vmin": 0,
        "vmax": 1,
        "colorbar_off": False,
        "extend": "neither",
        "colorbar_position": "vertical",
        "colorbar_position_set": False,
        "colorbar_left": 0.9,
        "colorbar_bottom": 0.1,
        "colorbar_width": 0.02,
        "colorbar_height": 0.8,
        "cmap": "viridis",
        "colorbar_labelsize": 8,
        "fontsize": 8,
        "legend_box_x": 0,
        "legend_box_y": 0,
        "legend_box_size": 1,
        "legend_lw": 1,
        "legend_fontsize": 8,
        "x_rotation": 0,
        "x_ha": "center",
        "y_rotation": 0,
        "y_ha": "right",
        "ylabel": "",
        "xlabel": "",
        "title": "",
        "title_size": 10,
        "saving_format": "png",
        "dpi": 80,
    }


def _radar_option():
    return {
        "cmap": "viridis",
        "x_wise": 4,
        "y_wise": 3,
        "dpi": 80,
        "saving_format": "png",
        "font_family": "DejaVu Sans",
        "font_size": 8,
        "titlesize": 10,
        "linewidth": 1,
        "patch_linewidth": 1,
        "lines_linestyle": "-",
        "xtick_labelsize": 8,
        "legend_fontsize": 8,
        "legend_title": "Simulation",
        "vmin_max_on": False,
    }


def test_radar_map_passes_score_name_to_color_scale(tmp_path, monkeypatch):
    """Radar maps should use the requested score name, not a stale undefined varname."""
    import matplotlib.pyplot as plt

    import openbench.visualization.Fig_radarmap as fig_radar

    source_csv = tmp_path / "radar.csv"
    pd.DataFrame(
        [
            {"Item": "Runoff", "Reference": "RefA", "SimA": 0.8, "SimB": 0.6},
            {"Item": "ET", "Reference": "RefB", "SimA": 0.7, "SimB": 0.5},
        ]
    ).to_csv(source_csv, index=False)

    seen = {}

    def fake_get_index(vmin, vmax, cmap_name, varname):
        seen["varname"] = varname
        return plt.get_cmap(cmap_name), np.array([0.0, 0.5, 1.0]), None, None, "neither"

    monkeypatch.setattr(fig_radar, "get_index", fake_get_index)
    monkeypatch.setattr(fig_radar, "save_figure", lambda *args, **kwargs: None)

    fig_radar.make_scenarios_comparison_radar_map(str(source_csv), "nBiasScore", _radar_option())

    assert seen["varname"] == "nBiasScore"


def test_parallel_coordinates_inner_plot_failures_propagate(tmp_path, monkeypatch):
    """Parallel Coordinates must not swallow renderer failures and claim success."""
    import openbench.visualization.Fig_parallel_coordinates as fig_parallel

    comparison_dir = tmp_path / "comparisons" / "Parallel_Coordinates"
    comparison_dir.mkdir(parents=True)
    source_csv = comparison_dir / "Parallel_Coordinates_evaluations.csv"
    pd.DataFrame(
        [
            {"Item": "Runoff", "Reference": "RefA", "Simulation": "SimA", "Overall_Score": 0.8, "bias": 1.0},
            {"Item": "Runoff", "Reference": "RefA", "Simulation": "SimB", "Overall_Score": 0.7, "bias": 2.0},
        ]
    ).to_csv(source_csv, sep="\t", index=False)

    def fail_plot(*args, **kwargs):
        raise RuntimeError("parallel plot boom")

    monkeypatch.setattr(fig_parallel, "parallel_coordinate_plot", fail_plot)

    option = _parallel_coordinates_option()
    with pytest.raises(RuntimeError, match="parallel plot boom"):
        fig_parallel.make_scenarios_comparison_parallel_coordinates(
            str(source_csv),
            str(tmp_path),
            ["Runoff"],
            ["Overall_Score"],
            ["bias"],
            option,
        )

    assert "situation" not in option


def test_diff_plot_option_mutations_do_not_leak(monkeypatch):
    """Diff Plot should derive per-plot title/cmap/label from a copy of the caller's option dict."""
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    calls = []

    def fake_plot_grid_map(*args):
        calls.append(args)

    monkeypatch.setattr(fig_diff, "plot_grid_map", fake_plot_grid_map)

    option = {"cmap": "", "colorbar_label": "original"}
    fig_diff.plot_diff_results(
        "/tmp",
        "anomaly",
        "bias",
        "Runoff",
        "RefA",
        "SimA",
        {},
        {"Runoff": {"SimA_varunit": "mm/day"}},
        "grid",
        option,
    )

    assert option == {"cmap": "", "colorbar_label": "original"}
    assert calls[0][-1] is not option
    assert calls[0][-1]["cmap"] == "RdBu_r"


def test_relative_score_option_mutations_do_not_leak(tmp_path, monkeypatch):
    """Relative Score should not leave computed vmin/vmax/extend/cmap in the shared option dict."""
    import openbench.visualization.Fig_Relative_Score as fig_relative

    def fail_after_option_setup(*args, **kwargs):
        raise RuntimeError("stop before cartopy")

    monkeypatch.setattr(fig_relative.plt, "figure", fail_after_option_setup)

    option = {
        "cmap": "",
        "vmin_max_on": False,
        "font": "DejaVu Sans",
        "labelsize": 8,
        "xtick": 8,
        "ytick": 8,
        "x_wise": 4,
        "y_wise": 3,
        "markersize": 10,
        "marker": "o",
        "line_width": 1,
        "set_lat_lon": False,
        "max_lon": 360,
        "min_lon": 0,
        "max_lat": 90,
        "min_lat": -90,
        "xticklabel": "",
        "yticklabel": "",
        "title": "",
        "title_size": 10,
        "colorbar_position_set": False,
        "colorbar_position": "vertical",
        "saving_format": "png",
        "dpi": 80,
    }

    with pytest.raises(RuntimeError, match="stop before cartopy"):
        fig_relative.make_stn_plot_index(
            str(tmp_path / "relative.csv"),
            "Overall_Score",
            np.array([0.8, 0.9]),
            np.array([30.0, 31.0]),
            np.array([100.0, 101.0]),
            {"min_lon": 0, "max_lon": 360, "min_lat": -90, "max_lat": 90},
            option,
        )

    assert option["cmap"] == ""
    assert "vmin" not in option
    assert "vmax" not in option
    assert "extend" not in option


def test_fig_toolbox_get_index_uses_current_core_score_names():
    from openbench.visualization.Fig_toolbox import get_index

    _cmap, mticks, norm, _bnd, extend = get_index(0.2, 0.8, "viridis", "nBiasScore")

    assert min(mticks) >= 0
    assert max(mticks) <= 1
    assert norm.vmin >= 0
    assert norm.vmax <= 1
    assert extend in {"neither", "min", "max", "both"}


def test_basic_station_map_skips_all_nan_metric(tmp_path, monkeypatch):
    """A metric with no finite station values should not fail the whole run."""
    import openbench.visualization.Fig_Basic_Plot as fig_basic

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    pd.DataFrame([{"sim_lon": 10.0, "sim_lat": 40.0, "correlation": np.nan}]).to_csv(
        scores_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("all-NaN metric should be skipped before plotting")

    monkeypatch.setattr(fig_basic, "plot_stn_map", fail_if_called)

    fig_basic.make_plot_index_stn(
        SimpleNamespace(
            casedir=str(tmp_path),
            item="Runoff",
            ref_source="RefA",
            sim_source="SimA",
            metrics=["correlation"],
            scores=[],
            fig_nml={
                "make_stn_plot_index": {"vmin_max_on": False, "cmap": "viridis"},
                "make_geo_plot_index": {"extend": "both"},
            },
        )
    )


def test_portrait_plot_inner_failures_propagate(tmp_path, monkeypatch):
    """Portrait seasonal should not hide sub-plot renderer failures."""
    import openbench.visualization.Fig_portrait_plot_seasonal as fig_portrait

    comparison_dir = tmp_path / "comparisons" / "Portrait_Plot_seasonal"
    comparison_dir.mkdir(parents=True)
    source_csv = comparison_dir / "Portrait_Plot_seasonal.csv"
    row = {"Item": "Runoff", "Reference": "RefA", "Simulation": "SimA"}
    for season in ("DJF", "MAM", "JJA", "SON"):
        row[f"Overall_Score_{season}"] = 0.8
        row[f"bias_{season}"] = 1.0
    pd.DataFrame([row]).to_csv(source_csv, sep="\t", index=False)

    def fail_plot(*args, **kwargs):
        raise RuntimeError("portrait plot boom")

    monkeypatch.setattr(fig_portrait, "portrait_plot", fail_plot)

    with pytest.raises(RuntimeError, match="portrait plot boom"):
        fig_portrait.make_scenarios_comparison_Portrait_Plot_seasonal(
            str(source_csv),
            str(tmp_path),
            ["Runoff"],
            ["Overall_Score"],
            ["bias"],
            _portrait_option(),
        )


def test_portrait_plot_single_metric_axes_are_indexable(tmp_path, monkeypatch):
    """Portrait seasonal should handle the single-metric shared-axes branch."""
    import matplotlib.pyplot as plt

    import openbench.visualization.Fig_portrait_plot_seasonal as fig_portrait

    comparison_dir = tmp_path / "comparisons" / "Portrait_Plot_seasonal"
    comparison_dir.mkdir(parents=True)
    source_csv = comparison_dir / "Portrait_Plot_seasonal.csv"
    row = {"Item": "Runoff", "Reference": "RefA", "Simulation": "SimA"}
    for season in ("DJF", "MAM", "JJA", "SON"):
        row[f"Overall_Score_{season}"] = 0.8
        row[f"bias_{season}"] = 1.0
    pd.DataFrame([row]).to_csv(source_csv, sep="\t", index=False)

    def fake_plot(*args, **kwargs):
        fig = kwargs.get("fig")
        ax = kwargs.get("ax")
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        return fig, ax, None

    monkeypatch.setattr(fig_portrait, "portrait_plot", fake_plot)

    fig_portrait.make_scenarios_comparison_Portrait_Plot_seasonal(
        str(source_csv),
        str(tmp_path),
        ["Runoff"],
        ["Overall_Score"],
        ["bias"],
        _portrait_option(),
    )


def test_kernel_density_inner_kde_failures_propagate(tmp_path, monkeypatch):
    """KDE should fail if a required simulation density cannot be computed."""
    import openbench.visualization.Fig_kernel_density_estimate as fig_kde

    def fail_kde(*args, **kwargs):
        raise RuntimeError("kde boom")

    monkeypatch.setattr(fig_kde, "gaussian_kde", fail_kde)

    with pytest.raises(RuntimeError, match="kde boom"):
        fig_kde.make_scenarios_comparison_Kernel_Density_Estimate(
            str(tmp_path),
            "Runoff",
            "RefA",
            ["SimA"],
            "Overall_Score",
            [np.array([0.7, 0.8, 0.9])],
            _kde_option(),
        )


def test_relative_score_station_inner_plot_failures_propagate(tmp_path, monkeypatch):
    """Relative Score station plotting must not swallow per-score renderer failures."""
    import openbench.visualization.Fig_Relative_Score as fig_relative

    relative_csv = tmp_path / "Runoff_stn_RefA_SimA_relative_scores.csv"
    pd.DataFrame(
        [
            {
                "ref_lon": 10.0,
                "ref_lat": 45.0,
                "relative_Overall_Score_SimA": 0.8,
            }
        ]
    ).to_csv(relative_csv, index=False)

    def fail_plot(*args, **kwargs):
        raise RuntimeError("relative station plot boom")

    monkeypatch.setattr(fig_relative, "make_stn_plot_index", fail_plot)

    with pytest.raises(RuntimeError, match="relative station plot boom"):
        fig_relative.make_scenarios_comparison_Relative_Score(
            str(tmp_path),
            "Runoff",
            "RefA",
            "SimA",
            ["Overall_Score"],
            "stn",
            {},
            {},
        )


def test_relative_score_station_all_nan_raises_before_plot(tmp_path, monkeypatch):
    """Relative Score station renderer should fail on all-NaN data instead of silently skipping."""
    import openbench.visualization.Fig_Relative_Score as fig_relative

    plot_calls = []
    monkeypatch.setattr(fig_relative.plt, "figure", lambda *args, **kwargs: plot_calls.append(args))

    with pytest.raises(ValueError, match="no finite data"):
        fig_relative.make_stn_plot_index(
            str(tmp_path / "relative.csv"),
            "Overall_Score",
            np.array([np.nan]),
            np.array([30.0]),
            np.array([100.0]),
            {"min_lon": 0, "max_lon": 360, "min_lat": -90, "max_lat": 90},
            {
                "cmap": "viridis",
                "vmin_max_on": False,
                "font": "DejaVu Sans",
                "labelsize": 8,
                "xtick": 8,
                "ytick": 8,
                "x_wise": 4,
                "y_wise": 3,
                "markersize": 10,
                "marker": "o",
                "line_width": 1,
                "set_lat_lon": False,
                "max_lon": 360,
                "min_lon": 0,
                "max_lat": 90,
                "min_lat": -90,
                "xticklabel": "",
                "yticklabel": "",
                "title": "",
                "title_size": 10,
                "colorbar_position_set": False,
                "colorbar_position": "vertical",
                "saving_format": "png",
                "dpi": 80,
            },
        )

    assert plot_calls == []


def test_diff_plot_grid_all_nan_raises_before_map(tmp_path, monkeypatch):
    """Diff Plot grid renderer should reject all-NaN NetCDF inputs before map creation."""
    import xarray as xr

    import openbench.visualization.Fig_Diff_Plot as fig_diff

    xr.Dataset(
        {"bias_anomaly": (("lat", "lon"), np.array([[np.nan]]))},
        coords={"lat": [30.0], "lon": [100.0]},
    ).to_netcdf(tmp_path / "all_nan.nc")
    plot_calls = []
    monkeypatch.setattr(fig_diff.plt, "figure", lambda *args, **kwargs: plot_calls.append(args))

    with pytest.raises(ValueError, match="no finite data"):
        fig_diff.plot_grid_map(
            str(tmp_path),
            "all_nan.nc",
            {"min_lon": 0, "max_lon": 360, "min_lat": -90, "max_lat": 90},
            "bias",
            "bias_anomaly",
            {
                "font": "DejaVu Sans",
                "labelsize": 8,
                "xtick": 8,
                "ytick": 8,
                "vmin_max_on": False,
                "cmap": "viridis",
                "x_wise": 4,
                "y_wise": 3,
                "show_method": "interpolate",
                "line_width": 1,
                "set_lat_lon": False,
                "max_lon": 360,
                "min_lon": 0,
                "max_lat": 90,
                "min_lat": -90,
                "xticklabel": "",
                "yticklabel": "",
                "title": "",
                "title_size": 10,
                "colorbar_position_set": False,
                "colorbar_position": "vertical",
                "saving_format": "png",
                "dpi": 80,
            },
        )

    assert plot_calls == []


def test_parallel_coordinates_all_nan_axis_raises_before_plot():
    """Parallel Coordinates should reject axes that contain no finite values."""
    import openbench.visualization.Fig_parallel_coordinates as fig_parallel

    with pytest.raises(ValueError, match="Parallel Coordinates/bias: no finite data"):
        fig_parallel.parallel_coordinate_plot(
            np.array([[0.8, np.nan], [0.7, np.nan]]),
            ["Overall_Score", "bias"],
            ["SimA", "SimB"],
            option={"situation": 1},
        )


def test_portrait_plot_all_nan_raises_before_figure():
    """Portrait plotting core should reject all-NaN matrices before figure creation."""
    import matplotlib.pyplot as plt

    import openbench.visualization.Fig_portrait_plot_seasonal as fig_portrait

    plt.close("all")
    with pytest.raises(ValueError, match="no finite data"):
        fig_portrait.portrait_plot(
            np.array([[np.nan]]),
            ["SimA"],
            ["bias"],
            fig=None,
            ax=None,
        )
    assert plt.get_fignums() == []


def test_diff_plot_score_plot_failures_propagate(tmp_path, monkeypatch):
    """Diff Plot should fail when a score anomaly renderer fails."""
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    def fail_plot(*args, **kwargs):
        raise RuntimeError("diff plot boom")

    monkeypatch.setattr(fig_diff, "plot_grid_map", fail_plot)

    with pytest.raises(RuntimeError, match="diff plot boom"):
        fig_diff.make_scenarios_comparison_Diff_Plot(
            str(tmp_path),
            metrics=[],
            scores=["bias"],
            evaluation_item="Runoff",
            ref_source="RefA",
            sim_sources=["SimA"],
            main_nml={},
            sim_nml={
                "Runoff": {
                    "SimA_varunit": "mm/day",
                }
            },
            ref_data_type="grid",
            option={"cmap": "RdBu_r"},
        )


def test_public_diagram_helpers_do_not_open_auto_figures_before_input_validation():
    """Invalid 3-argument public diagram calls should not leave hidden figures open."""
    import matplotlib.pyplot as plt

    from openbench.visualization.Fig_target_diagram import target_diagram
    from openbench.visualization.Fig_taylor_diagram import taylor_diagram

    plt.close("all")

    with pytest.raises(ValueError):
        taylor_diagram([1.0], np.array([0.0]), np.array([1.0]))
    assert plt.get_fignums() == []

    with pytest.raises(ValueError):
        target_diagram([0.0], np.array([0.0]), np.array([1.0]))
    assert plt.get_fignums() == []


def test_basic_metric_plot_preparation_failures_propagate(tmp_path, monkeypatch):
    """Basic metric maps should fail the run when metric input/prep fails."""
    import openbench.visualization.Fig_Basic_Plot as fig_basic

    def fail_open_dataset(*args, **kwargs):
        raise RuntimeError("metric nc boom")

    monkeypatch.setattr(fig_basic.xr, "open_dataset", fail_open_dataset)

    renderer = SimpleNamespace(
        ref_varname="Runoff",
        metrics=["bias"],
        scores=[],
        fig_nml={"make_geo_plot_index": {"vmin_max_on": False, "cmap": "viridis"}},
        casedir=str(tmp_path),
        item="Runoff",
        ref_source="RefA",
        sim_source="SimA",
    )

    with pytest.raises(RuntimeError, match="metric nc boom"):
        fig_basic.make_plot_index_grid(renderer)


def test_basic_station_map_renderer_skips_empty_filtered_data(tmp_path, monkeypatch):
    """All-NaN station metric maps are optional figures and should be skipped."""
    import openbench.visualization.Fig_Basic_Plot as fig_basic

    case_dir = tmp_path / "case"
    scores_dir = case_dir / "scores"
    scores_dir.mkdir(parents=True)
    pd.DataFrame([{"ID": "S1", "ref_lon": 100.0, "ref_lat": 30.0, "bias": np.nan}]).to_csv(
        scores_dir / "Runoff_stn_RefA_SimA_evaluations.csv",
        index=False,
    )

    renderer = SimpleNamespace(
        casedir=str(case_dir),
        item="Runoff",
        ref_source="RefA",
        sim_source="SimA",
        ref_varunit="mm",
        sim_varunit="mm",
        metrics=["bias"],
        scores=[],
        fig_nml={
            "make_geo_plot_index": {"extend": "both"},
            "make_stn_plot_index": {
                "cmap": "viridis",
                "vmin_max_on": False,
                "colorbar_label": "",
            },
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("all-NaN metric should be skipped before plotting")

    monkeypatch.setattr(fig_basic, "plot_stn_map", fail_if_called)

    fig_basic.make_plot_index_stn(renderer)


def test_basic_grid_renderer_rejects_all_nan_netcdf(tmp_path):
    """Basic grid renderer should reject all-NaN NC inputs before creating a misleading map."""
    import xarray as xr

    import openbench.visualization.Fig_Basic_Plot as fig_basic

    source_nc = tmp_path / "Mean.nc"
    xr.Dataset(
        {"Mean": (("lat", "lon"), np.array([[np.nan]]))},
        coords={"lat": [30.0], "lon": [100.0]},
    ).to_netcdf(source_nc)

    with pytest.raises(ValueError, match="Mean basic map: no finite data to plot"):
        fig_basic.make_Basic(str(source_nc), "Mean", "RefA", {}, {"cmap": "viridis"})


def test_stn_plot_index_renderer_rejects_all_nan_inputs(tmp_path):
    """Standalone station plot renderer should fail fast on stale all-NaN CSV data."""
    import openbench.visualization.Fig_stn_plot_index as fig_stn

    source_csv = tmp_path / "basic.csv"
    pd.DataFrame([{"ID": "S1", "ref_lon": 100.0, "ref_lat": 30.0, "ref_value": np.nan, "sim_value": np.nan}]).to_csv(
        source_csv, index=False
    )

    with pytest.raises(ValueError, match="Mean station map/ref_value: no finite data to plot"):
        fig_stn.make_stn_plot_index(
            str(source_csv),
            "Mean",
            {"min_lon": 0, "max_lon": 180, "min_lat": -90, "max_lat": 90},
            ("RefA", "SimA"),
            {"cmap": "viridis"},
        )
