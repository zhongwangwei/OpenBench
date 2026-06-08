import numpy as np
import pytest
import xarray as xr


def test_downsample_for_plot_leaves_small_data_unchanged():
    from openbench.visualization._downsample import downsample_for_plot

    da = xr.DataArray(
        np.ones((4, 5)),
        dims=("lat", "lon"),
        coords={"lat": np.arange(4), "lon": np.arange(5)},
        name="bias",
    )

    out = downsample_for_plot(da, {"render_resolution": "auto", "max_pixels": 100})

    assert out.sizes == da.sizes


def test_downsample_for_plot_coarsens_large_data_to_max_pixels():
    from openbench.visualization._downsample import downsample_for_plot

    da = xr.DataArray(
        np.arange(100, dtype=float).reshape(10, 10),
        dims=("lat", "lon"),
        coords={"lat": np.arange(10), "lon": np.arange(10)},
        name="bias",
    )

    out = downsample_for_plot(da, {"render_resolution": "auto", "max_pixels": 25})

    assert out.sizes["lat"] == 5
    assert out.sizes["lon"] == 5
    assert out.name == "bias"
    assert float(out.isel(lat=0, lon=0)) == np.mean([[0, 1], [10, 11]])


def test_downsample_for_plot_native_never_downsamples():
    from openbench.visualization._downsample import downsample_for_plot

    da = xr.DataArray(np.ones((10, 10)), dims=("lat", "lon"), coords={"lat": np.arange(10), "lon": np.arange(10)})

    out = downsample_for_plot(da, {"render_resolution": "native", "max_pixels": 1})

    assert out.sizes == da.sizes


def test_downsample_for_plot_uses_quality_presets():
    from openbench.visualization._downsample import downsample_for_plot

    da = xr.DataArray(np.ones((1000, 1000)), dims=("lat", "lon"))

    out = downsample_for_plot(da, {"render_resolution": "low"})

    assert out.sizes["lat"] * out.sizes["lon"] <= 300_000


def _map_option():
    return {
        "font": "DejaVu Sans",
        "labelsize": 8,
        "xtick": 8,
        "ytick": 8,
        "x_wise": 4,
        "y_wise": 3,
        "cmap": "viridis",
        "vmin_max_on": False,
        "show_method": "imshow",
        "line_width": 1,
        "max_lat": 90,
        "min_lat": -90,
        "max_lon": 180,
        "min_lon": -180,
        "set_lat_lon": False,
        "colorbar_position_set": False,
        "colorbar_position": "vertical",
        "xticklabel": "",
        "yticklabel": "",
        "title": "",
        "title_size": 10,
        "saving_format": "png",
        "dpi": 80,
        "render_resolution": "low",
    }


def _write_grid(path, name):
    xr.Dataset(
        {name: (("lat", "lon"), np.arange(12, dtype=float).reshape(3, 4))},
        coords={"lat": [-1.0, 0.0, 1.0], "lon": [10.0, 20.0, 30.0, 40.0]},
    ).to_netcdf(path)


def test_diff_plot_grid_map_invokes_plot_downsample(tmp_path, monkeypatch):
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    _write_grid(tmp_path / "bias_anomaly.nc", "bias_anomaly")

    def fail_downsample(data, option):
        assert data.name == "bias_anomaly"
        assert option["render_resolution"] == "low"
        raise RuntimeError("downsample called")

    monkeypatch.setattr(fig_diff, "downsample_for_plot", fail_downsample)

    with pytest.raises(RuntimeError, match="downsample called"):
        fig_diff.plot_grid_map(
            str(tmp_path),
            "bias_anomaly.nc",
            {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
            "bias",
            "bias_anomaly",
            _map_option(),
        )


def test_relative_score_grid_map_invokes_plot_downsample(monkeypatch):
    import openbench.visualization.Fig_Relative_Score as fig_relative

    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [-1.0, 0.0, 1.0], "lon": [10.0, 20.0, 30.0, 40.0]},
    )

    def fail_downsample(candidate, option):
        assert candidate.identical(data)
        assert option["render_resolution"] == "low"
        raise RuntimeError("downsample called")

    monkeypatch.setattr(fig_relative, "downsample_for_plot", fail_downsample)

    with pytest.raises(RuntimeError, match="downsample called"):
        fig_relative.make_geo_plot_index(
            "unused.nc",
            data,
            data.lat.values,
            data.lon.values,
            {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
            _map_option(),
        )


def test_basic_grid_map_invokes_plot_downsample(tmp_path, monkeypatch):
    import openbench.visualization.Fig_Basic_Plot as fig_basic

    _write_grid(tmp_path / "Mean.nc", "Mean")

    def fail_downsample(data, option):
        assert data.name == "Mean"
        assert option["render_resolution"] == "low"
        raise RuntimeError("downsample called")

    monkeypatch.setattr(fig_basic, "downsample_for_plot", fail_downsample)

    with pytest.raises(RuntimeError, match="downsample called"):
        fig_basic.make_Basic(
            str(tmp_path / "Mean.nc"),
            "Mean",
            ["SimA"],
            {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
            _map_option(),
        )


def test_default_fig_options_expose_plot_downsample_controls():
    from openbench.config.adapter import build_fig_nml

    fig_nml = build_fig_nml()

    for section in (
        fig_nml["make_geo_plot_index"],
        fig_nml["Comparison"]["Diff_Plot"],
        fig_nml["Comparison"]["Relative_Score"],
        fig_nml["Statistic"]["Basic"],
    ):
        assert section["render_resolution"] == "auto"
        assert section["max_pixels"] == 1_000_000
        assert section["downsample_method"] == "coarsen_mean"
