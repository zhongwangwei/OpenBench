import numpy as np
import pandas as pd
import pytest
import xarray as xr


class _ProcessorMixin:
    main_nml = {"general": {"min_lon": 0, "max_lon": 360, "min_lat": -90, "max_lat": 90}}


def _processor():
    from openbench.core._comparison_relative import RelativeScoreComparisonMixin

    class Processor(RelativeScoreComparisonMixin, _ProcessorMixin):
        pass

    return Processor()


def _station_scores(root, source, values, *, nse=None, status=None, reason=None):
    status = status or ["ok"] * len(values)
    reason = reason or [""] * len(values)
    scores = root / "scores"
    scores.mkdir(exist_ok=True)
    data = {
        "ID": [f"S{i + 1}" for i in range(len(values))],
        "ref_lon": [100.0 + i for i in range(len(values))],
        "ref_lat": [30.0 + i for i in range(len(values))],
        "Overall_Score": values,
    }
    if nse is not None:
        data["NSE"] = nse
    pd.DataFrame(data).to_csv(scores / f"Runoff_stn_RefA_{source}_evaluations.csv", index=False)
    status_dir = root / "data" / f"stn_RefA_{source}"
    status_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": [f"S{i + 1}" for i in range(len(values))], "status": status, "reason": reason}).to_csv(
        status_dir / "Runoff_evaluation_status.csv", index=False
    )


def test_station_relative_uses_configured_sources_and_keeps_unavailable_rows(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module
    from openbench.util.filenames import relative_station_scores_filename

    _station_scores(tmp_path, "SimA", [0.8, 0.5], nse=[1.0, 3.0])
    _station_scores(
        tmp_path, "SimB", [0.8, 9.9], nse=[2.0, 5.0], status=["ok", "unavailable"], reason=["", "missing nc"]
    )
    _station_scores(tmp_path, "StaleSim", [0.1, 0.1], nse=[100.0, 100.0])
    calls = []
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Relative_Score", lambda *args: calls.append(args))

    _processor().scenarios_Relative_Score_comparison(
        str(tmp_path),
        {
            "general": {"Runoff_sim_source": ["SimA", "SimB"]},
            "Runoff": {
                "SimA_data_type": "stn",
                "SimA_varname": "flow",
                "SimB_data_type": "stn",
                "SimB_varname": "flow",
            },
        },
        {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "stn"}},
        ["Runoff"],
        ["Overall_Score", "NSE"],
        [],
        {},
    )

    out = pd.read_csv(
        tmp_path / "comparisons/Relative_Score" / relative_station_scores_filename("Runoff", "RefA", "SimA")
    )
    value = "relative_Overall_Score_SimA"
    assert out["ID"].tolist() == ["S1", "S2"]
    assert out[value].isna().tolist() == [True, True]
    assert out[f"status_{value}"].tolist() == ["unavailable", "unavailable"]
    assert out[f"reason_{value}"].tolist() == ["zero across-model variance", "fewer than two finite model evaluations"]
    assert out["status"].tolist() == ["partial", "unavailable"]
    assert out["reason"].fillna("").tolist() == [
        "",
        "fewer than two finite model evaluations; fewer than two finite model evaluations",
    ]
    assert out["status_relative_NSE_SimA"].tolist() == ["ok", "unavailable"]
    assert all(call[3] in {"SimA", "SimB"} for call in calls)


def test_station_relative_missing_configured_source_is_fatal(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module

    _station_scores(tmp_path, "SimA", [0.8])
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Relative_Score", lambda *args: None)

    with pytest.raises(FileNotFoundError):
        _processor().scenarios_Relative_Score_comparison(
            str(tmp_path),
            {
                "general": {"Runoff_sim_source": ["SimA", "MissingSim"]},
                "Runoff": {
                    "SimA_data_type": "stn",
                    "SimA_varname": "flow",
                    "MissingSim_data_type": "stn",
                    "MissingSim_varname": "flow",
                },
            },
            {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "stn"}},
            ["Runoff"],
            ["Overall_Score"],
            [],
            {},
        )


def test_grid_relative_uses_configured_sources_not_stale_glob(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module
    from openbench.util.filenames import relative_grid_score_filename

    (tmp_path / "scores").mkdir()
    for source, value in {"SimA": 2.0, "SimB": 2.0, "StaleSim": 9.0}.items():
        xr.Dataset(
            {"Overall_Score": (("lat", "lon"), np.array([[value]], dtype=float))},
            coords={"lat": [30.0], "lon": [100.0]},
        ).to_netcdf(tmp_path / "scores" / f"Runoff_ref_RefA_sim_{source}_Overall_Score.nc")
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Relative_Score", lambda *args: None)

    _processor().scenarios_Relative_Score_comparison(
        str(tmp_path),
        {
            "general": {"Runoff_sim_source": ["SimA", "SimB"]},
            "Runoff": {
                "SimA_data_type": "grid",
                "SimA_varname": "flow",
                "SimB_data_type": "grid",
                "SimB_varname": "flow",
            },
        },
        {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}},
        ["Runoff"],
        ["Overall_Score"],
        [],
        {},
    )

    with xr.open_dataset(
        tmp_path
        / "comparisons/Relative_Score"
        / relative_grid_score_filename("Runoff", "RefA", "SimA", "Overall_Score")
    ) as ds:
        assert np.isnan(ds["relative_Overall_Score"].values).all()


def _relative_option():
    return {
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
        "colorbar_label": "",
        "saving_format": "png",
        "dpi": 80,
        "show_method": "imshow",
    }


def test_relative_station_renderer_uses_signed_z_score_range(tmp_path, monkeypatch):
    import openbench.visualization.Fig_Relative_Score as fig_relative

    clims = []

    def capture(fig, *args, **kwargs):
        clims.append(fig.axes[0].collections[0].get_clim())

    monkeypatch.setattr(fig_relative, "save_figure", capture)
    main = {"min_lon": 0, "max_lon": 360, "min_lat": -90, "max_lat": 90}
    fig_relative.make_stn_plot_index(
        str(tmp_path / "positive.csv"),
        "Overall_Score",
        np.array([2.0]),
        np.array([30.0]),
        np.array([100.0]),
        main,
        _relative_option(),
    )
    fig_relative.make_stn_plot_index(
        str(tmp_path / "negative.csv"),
        "Overall_Score",
        np.array([-2.0]),
        np.array([30.0]),
        np.array([100.0]),
        main,
        _relative_option(),
    )

    assert clims == [(-2.0, 2.0), (-2.0, 2.0)]


def test_relative_grid_renderer_uses_signed_z_score_range(tmp_path, monkeypatch):
    import openbench.visualization.Fig_Relative_Score as fig_relative

    clims = []

    def capture(fig, *args, **kwargs):
        clims.append(fig.axes[0].images[0].get_clim())

    monkeypatch.setattr(fig_relative, "save_figure", capture)
    main = {"min_lon": 0, "max_lon": 360, "min_lat": -90, "max_lat": 90}
    for value in (2.0, -2.0):
        fig_relative.make_geo_plot_index(
            str(tmp_path / f"{value}.nc"),
            xr.DataArray([[value]], dims=("lat", "lon"), coords={"lat": [30.0], "lon": [100.0]}),
            np.array([30.0]),
            np.array([100.0]),
            main,
            _relative_option(),
        )

    assert clims == [(-2.0, 2.0), (-2.0, 2.0)]


def test_relative_station_renderer_honors_asymmetric_manual_bounds(tmp_path, monkeypatch):
    import openbench.visualization.Fig_Relative_Score as fig_relative

    clims = []

    def capture(fig, *args, **kwargs):
        clims.append(fig.axes[0].collections[0].norm.vmin)
        clims.append(fig.axes[0].collections[0].norm.vmax)

    option = _relative_option()
    option.update({"vmin_max_on": True, "vmin": -0.25, "vmax": 0.75})
    monkeypatch.setattr(fig_relative, "save_figure", capture)
    fig_relative.make_stn_plot_index(
        str(tmp_path / "manual.csv"),
        "Overall_Score",
        np.array([0.5]),
        np.array([30.0]),
        np.array([100.0]),
        {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
        option,
    )

    assert clims == [-0.25, 0.75]


def test_relative_grid_renderer_honors_asymmetric_manual_bounds(tmp_path, monkeypatch):
    import openbench.visualization.Fig_Relative_Score as fig_relative

    clims = []

    def capture(fig, *args, **kwargs):
        clims.append(fig.axes[0].images[0].norm.vmin)
        clims.append(fig.axes[0].images[0].norm.vmax)

    option = _relative_option()
    option.update({"vmin_max_on": True, "vmin": -0.25, "vmax": 0.75})
    monkeypatch.setattr(fig_relative, "save_figure", capture)
    fig_relative.make_geo_plot_index(
        str(tmp_path / "manual.nc"),
        xr.DataArray([[0.5]], dims=("lat", "lon"), coords={"lat": [30.0], "lon": [100.0]}),
        np.array([30.0]),
        np.array([100.0]),
        {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
        option,
    )

    assert clims == [-0.25, 0.75]


def test_relative_renderer_rejects_invalid_manual_bounds(tmp_path):
    import openbench.visualization.Fig_Relative_Score as fig_relative

    option = _relative_option()
    option.update({"vmin_max_on": True, "vmin": 1.0, "vmax": 1.0})
    with pytest.raises(ValueError, match="finite increasing bounds"):
        fig_relative.make_stn_plot_index(
            str(tmp_path / "bad.csv"),
            "Overall_Score",
            np.array([0.5]),
            np.array([30.0]),
            np.array([100.0]),
            {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
            option,
        )
