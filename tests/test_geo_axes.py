from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from openbench.visualization._geo_axes import adaptive_degree_ticks, configure_geo_axis

_VISUALIZATION_DIR = Path(__file__).resolve().parents[1] / "src" / "openbench" / "visualization"


def test_adaptive_degree_ticks_for_amazon_extent():
    np.testing.assert_array_equal(adaptive_degree_ticks(-82, -35), [-80, -70, -60, -50, -40])
    np.testing.assert_array_equal(adaptive_degree_ticks(-22, 11), [-20, -10, 0, 10])


def test_adaptive_degree_ticks_preserve_global_spacing():
    np.testing.assert_array_equal(adaptive_degree_ticks(-180, 180), [-120, -60, 0, 60, 120])
    np.testing.assert_array_equal(adaptive_degree_ticks(-90, 90), [-60, -30, 0, 30, 60])


def test_adaptive_degree_ticks_support_small_and_positive_extents():
    for bounds in ((100, 101), (-1, 1), (0.001, 0.009)):
        ticks = adaptive_degree_ticks(*bounds)
        assert 4 <= ticks.size <= 7
        assert np.all(ticks > bounds[0])
        assert np.all(ticks < bounds[1])


def test_configure_geo_axis_uses_selected_extent_and_matching_gridlines():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    option = {
        "set_lat_lon": False,
        "min_lon": -180,
        "max_lon": 180,
        "min_lat": -90,
        "max_lat": 90,
    }

    extent, lon_ticks, lat_ticks = configure_geo_axis(
        ax,
        option,
        (-82, -35, -22, 11),
        gridline_kwargs={"linestyle": ":", "linewidth": 0.5},
    )

    assert extent == (-82.0, -35.0, -22.0, 11.0)
    np.testing.assert_array_equal(lon_ticks, [-80, -70, -60, -50, -40])
    np.testing.assert_array_equal(lat_ticks, [-20, -10, 0, 10])
    np.testing.assert_array_equal(ax.get_xticks(), lon_ticks)
    np.testing.assert_array_equal(ax.get_yticks(), lat_ticks)
    assert not ax.xaxis.get_ticks_position() == "top"
    assert not ax.yaxis.get_ticks_position() == "right"
    plt.close(fig)


def test_configure_geo_axis_honors_plot_extent_override():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    option = {
        "set_lat_lon": True,
        "min_lon": 100,
        "max_lon": 120,
        "min_lat": 20,
        "max_lat": 30,
    }

    extent, lon_ticks, lat_ticks = configure_geo_axis(ax, option, (-180, 180, -90, 90))

    assert extent == (100.0, 120.0, 20.0, 30.0)
    assert lon_ticks.size >= 4
    assert lat_ticks.size >= 4
    plt.close(fig)


def test_map_renderers_do_not_reintroduce_hard_coded_geographic_ticks():
    forbidden = (
        'np.arange(option["max_lon"], option["min_lon"], -60)',
        'np.arange(option["max_lat"], option["min_lat"], -30)',
        'np.arange(main_nml["max_lon"], main_nml["min_lon"], -60)',
        'np.arange(main_nml["max_lat"], main_nml["min_lat"], -30)',
    )
    for path in _VISUALIZATION_DIR.glob("Fig_*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(pattern in source for pattern in forbidden), path.name


def test_configure_geo_axis_keeps_rendering_degenerate_or_dateline_extents():
    for extent in ((100, 100, 20, 40), (100, 120, 30, 30), (170, -170, -10, 10)):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        _, lon_ticks, lat_ticks = configure_geo_axis(
            ax, {"set_lat_lon": False}, extent, gridline_kwargs={"linestyle": ":"}
        )
        fig.canvas.draw()

        assert lon_ticks.size == 0 or lat_ticks.size == 0
        plt.close(fig)
