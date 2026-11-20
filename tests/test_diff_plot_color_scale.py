import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr


class _StopBeforePlot(RuntimeError):
    pass


def _capture_color_limits(monkeypatch, fig_diff):
    captured = {}

    def fake_get_index(vmin, vmax, cmap):
        captured["limits"] = (vmin, vmax)
        ticks = np.array([vmin, vmax], dtype=float)
        return plt.get_cmap(cmap), ticks, mcolors.Normalize(vmin=vmin, vmax=vmax), ticks, "neither"

    monkeypatch.setattr(fig_diff, "get_index", fake_get_index)
    monkeypatch.setattr(fig_diff.plt, "figure", lambda *args, **kwargs: (_ for _ in ()).throw(_StopBeforePlot))
    return captured


def _scale_option(**overrides):
    option = {
        "font": "DejaVu Sans",
        "labelsize": 8,
        "xtick": 8,
        "ytick": 8,
        "vmin_max_on": False,
        "cmap": "viridis",
        "x_wise": 4,
        "y_wise": 3,
    }
    option.update(overrides)
    return option


@pytest.mark.parametrize("metric", ["Q95_Bias", "Q99_Bias"])
def test_diff_station_quantile_bias_uses_robust_uncapped_limits(monkeypatch, tmp_path, metric):
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    values = np.append(np.arange(100.0, 200.0), 110_000.0)
    captured = _capture_color_limits(monkeypatch, fig_diff)

    with pytest.raises(_StopBeforePlot):
        fig_diff.plot_stn_map(
            str(tmp_path),
            "quantile_bias.csv",
            np.arange(values.size),
            np.zeros(values.size),
            values,
            {},
            metric,
            f"{metric}_diff",
            _scale_option(),
        )

    low, high = np.percentile(values, [5, 95])
    assert captured["limits"] == (math.floor(low), math.ceil(high))
    assert captured["limits"][1] > 100


def test_diff_grid_q99_bias_uses_robust_limits(monkeypatch, tmp_path):
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    values = np.append(np.arange(100.0, 200.0), 110_000.0)
    xr.Dataset(
        {"Q99_Bias_diff": (("lat", "lon"), values.reshape(1, -1))},
        coords={"lat": [0.0], "lon": np.arange(values.size)},
    ).to_netcdf(tmp_path / "quantile_bias.nc")
    captured = _capture_color_limits(monkeypatch, fig_diff)

    with pytest.raises(_StopBeforePlot):
        fig_diff.plot_grid_map(
            str(tmp_path),
            "quantile_bias.nc",
            {},
            "Q99_Bias",
            "Q99_Bias_diff",
            _scale_option(),
        )

    low, high = np.percentile(values, [5, 95])
    assert captured["limits"] == (math.floor(low), math.ceil(high))


def test_diff_other_metric_keeps_full_data_limits(monkeypatch, tmp_path):
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    values = np.array([1.0, 2.0, 110_000.0])
    captured = _capture_color_limits(monkeypatch, fig_diff)

    with pytest.raises(_StopBeforePlot):
        fig_diff.plot_stn_map(
            str(tmp_path),
            "custom.csv",
            np.arange(values.size),
            np.zeros(values.size),
            values,
            {},
            "custom_metric",
            "custom_metric_diff",
            _scale_option(),
        )

    assert captured["limits"] == (1.0, 110_000.0)


def test_diff_manual_color_limits_are_not_overridden(monkeypatch, tmp_path):
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    captured = _capture_color_limits(monkeypatch, fig_diff)

    with pytest.raises(_StopBeforePlot):
        fig_diff.plot_stn_map(
            str(tmp_path),
            "quantile_bias.csv",
            np.arange(3),
            np.zeros(3),
            np.array([-1000.0, 0.0, 1000.0]),
            {},
            "Q99_Bias",
            "Q99_Bias_diff",
            _scale_option(vmin_max_on=True, vmin=-25.0, vmax=250.0),
        )

    assert captured["limits"] == (-25.0, 250.0)


def test_diff_grid_manual_color_limits_are_not_overridden(monkeypatch, tmp_path):
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    xr.Dataset(
        {"Q95_Bias_diff": (("lat", "lon"), [[-1000.0, 0.0, 1000.0]])},
        coords={"lat": [0.0], "lon": [0.0, 1.0, 2.0]},
    ).to_netcdf(tmp_path / "quantile_bias.nc")
    captured = _capture_color_limits(monkeypatch, fig_diff)

    with pytest.raises(_StopBeforePlot):
        fig_diff.plot_grid_map(
            str(tmp_path),
            "quantile_bias.nc",
            {},
            "Q95_Bias",
            "Q95_Bias_diff",
            _scale_option(vmin_max_on=True, vmin=-50.0, vmax=300.0),
        )

    assert captured["limits"] == (-50.0, 300.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([-1000.0, 0.0, 1000.0], "both"), ([0.0, 1000.0], "max"), ([-1000.0, 0.0], "min")],
)
def test_quantile_bias_outliers_set_colorbar_extension(values, expected):
    from openbench.visualization.Fig_Diff_Plot import _data_colorbar_extend

    assert _data_colorbar_extend(np.asarray(values), -100.0, 100.0) == expected


def test_diff_quantile_bias_difference_uses_percentage_points_label(monkeypatch):
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    plotted = []
    monkeypatch.setattr(fig_diff, "_diff_input_filename", lambda *args: "unused.nc")
    monkeypatch.setattr(fig_diff, "plot_grid_map", lambda *args: plotted.append(args[-1]))

    fig_diff.plot_diff_results(
        "/tmp",
        "difference",
        "Q99_Bias",
        "Streamflow",
        "RefA",
        ("CTL", "D"),
        {},
        {"Streamflow": {"CTL_varunit": "m3 s-1", "D_varunit": "m3 s-1"}},
        "grid",
        {"cmap": "", "colorbar_label": ""},
    )

    assert plotted[0]["colorbar_label"] == "pp"
