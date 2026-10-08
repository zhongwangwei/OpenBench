from pathlib import Path

import numpy as np
import pytest

from openbench.visualization import Fig_target_diagram, Fig_taylor_diagram

_VIZ_DIR = Path(__file__).resolve().parents[1] / "src" / "openbench" / "visualization"


def test_taylor_option_file_parses_tuple_literals(tmp_path):
    options_file = tmp_path / "taylor_options.csv"
    options_file.write_text('key,value\ncolcor,"(0.1, 0.2, 0.3)"\ncolrms,"[0, 0.6, 0]"\n', encoding="utf-8")

    options = Fig_taylor_diagram._read_options(
        Fig_taylor_diagram._default_options(np.array([0.5])), taylor_options_file=str(options_file)
    )

    assert options["colcor"] == (0.1, 0.2, 0.3)
    assert options["colrms"] == (0, 0.6, 0)


def test_taylor_option_file_does_not_eval_tuple_values(tmp_path):
    marker = tmp_path / "pwned"
    payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('owned')"
    options_file = tmp_path / "taylor_options.csv"
    options_file.write_text(f"key,value\ncolcor,{payload}\n", encoding="utf-8")

    with pytest.raises(Exception, match="Invalid colcor"):
        Fig_taylor_diagram._read_options(
            Fig_taylor_diagram._default_options(np.array([0.5])), taylor_options_file=str(options_file)
        )

    assert not marker.exists()


def test_visualization_option_parsers_do_not_contain_raw_eval():
    for source in (Path(Fig_taylor_diagram.__file__), Path(Fig_target_diagram.__file__)):
        assert "eval(" not in source.read_text(encoding="utf-8")


def test_portrait_plot_defaults_do_not_require_colorbar_options():
    import matplotlib

    matplotlib.use("Agg", force=True)

    from openbench.visualization.Fig_portrait_plot_seasonal import portrait_plot

    fig, ax, cbar = portrait_plot(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        ["a", "b"],
        ["c", "d"],
    )

    assert fig is not None
    assert ax is not None
    assert cbar is not None


def test_parallel_coordinate_plot_accepts_fig_only_and_marker_highlight_defaults():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from openbench.visualization.Fig_parallel_coordinates import parallel_coordinate_plot

    data = np.array([[1.0, 2.0], [2.0, 3.0]])
    fig = plt.figure()
    out_fig, ax = parallel_coordinate_plot(data, ["m1", "m2"], ["a", "b"], fig=fig)
    assert out_fig is fig
    assert ax is not None

    out_fig, ax = parallel_coordinate_plot(
        data,
        ["m1", "m2"],
        ["a", "b"],
        models_to_highlight=["a"],
        models_to_highlight_by_line=False,
    )
    assert out_fig is not None
    assert ax is not None


def test_taylor_legend_dict_uses_values_when_marker_legend_is_on():
    options = Fig_taylor_diagram._get_options(
        Fig_taylor_diagram._default_options(np.array([0.5])),
        markerlegend="on",
        legend={"set_legend": True, "bbox_to_anchor_x": 1.2, "bbox_to_anchor_y": 0.8},
    )

    assert options["legend"] == {"set_legend": True, "bbox_to_anchor_x": 1.2, "bbox_to_anchor_y": 0.8}


def test_fig_toolbox_get_index_honors_requested_builtin_colormap():
    from openbench.visualization.Fig_toolbox import get_index

    cmap, *_ = get_index(-1.0, 1.0, colormap="viridis")

    assert cmap.name == "viridis"


def test_fig_toolbox_get_index_keeps_negative_apfb_ticks():
    from openbench.visualization.Fig_toolbox import get_index

    _cmap, mticks, *_ = get_index(-1.0, 1.0, colormap="viridis", varname="APFB")

    assert min(mticks) < 0 < max(mticks)


def test_fig_toolbox_get_index_caps_mfm_components_at_one():
    from openbench.visualization.Fig_toolbox import get_index

    _cmap, mticks, *_ = get_index(0.0, 1.4, colormap="viridis", varname="MFM_eta")

    assert max(mticks) <= 1


@pytest.mark.parametrize(
    ("data_range", "expected"),
    [
        ((-1.0, 2.0), "max"),
        ((-2.0, 1.0), "min"),
        ((-2.0, 2.0), "both"),
        ((-1.0, 1.0), "neither"),
        ((-0.5, 0.5), "neither"),
    ],
)
def test_basic_plot_colorbar_extend_checks_each_end(data_range, expected):
    from openbench.visualization.Fig_Basic_Plot import colorbar_extend

    assert colorbar_extend(*data_range, -1.0, 1.0) == expected


def test_special_stat_figures_do_not_use_removed_cm_get_cmap_api():
    files = [
        "Fig_ANOVA.py",
        "Fig_Hellinger_Distance.py",
        "Fig_Three_Cornered_Hat.py",
        "Fig_Partial_Least_Squares_Regression.py",
    ]
    viz_dir = Path(__file__).resolve().parents[1] / "src" / "openbench" / "visualization"

    for filename in files:
        assert "cm.get_cmap" not in (viz_dir / filename).read_text(encoding="utf-8")


def test_heatmap_honors_configured_colormap_instead_of_hardcoding_rd_bu():
    source = (_VIZ_DIR / "Fig_heatmap.py").read_text(encoding="utf-8")

    assert "cmaps.MPL_RdBu_r" not in source
    assert 'get_colormap(option.get("cmap", "coolwarm"))' in source


def test_radarmap_uses_configurable_colormap_and_dynamic_series_colors():
    source = (_VIZ_DIR / "Fig_radarmap.py").read_text(encoding="utf-8")
    defaults = (
        Path(__file__).resolve().parents[1] / "src" / "openbench" / "data" / "fignml" / "RadarMap.yaml"
    ).read_text(encoding="utf-8")

    assert "cmap:" in defaults
    assert 'get_index(min_value, max_value, option.get("cmap", "Spectral"), score)' in source
    assert 'colors = ["#2887c5"' not in source
    assert "np.linspace(0.1, 0.9, max(series_count, 1))" in source


def test_all_nan_sensitive_map_renderers_use_finite_validation_helpers():
    expected = {
        "Fig_Standard_Deviation.py": ['finite_min_max(data, label="Standard Deviation")'],
        "Fig_ANOVA.py": ['finite_min_max(F_statistic, label="ANOVA F_statistic")'],
        "Fig_Three_Cornered_Hat.py": ["finite_min_max(", "relative_uncertainty", "uncertainty"],
        "Fig_geo_plot_index.py": ['finite_min_max(data, label=f"{method_name} geo plot index")'],
    }

    for filename, snippets in expected.items():
        source = (_VIZ_DIR / filename).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in source, f"{filename} missing {snippet!r}"

    assert "math.ceil(data.max().values)" not in (_VIZ_DIR / "Fig_Standard_Deviation.py").read_text(encoding="utf-8")
    assert "np.nanmin(data), np.nanmax(data)" not in (_VIZ_DIR / "Fig_geo_plot_index.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-2.3, 5.1, (-3, 6)),
        (-0.0031, 0.012, (-0.01, 0.02)),
        (0.0, 0.034, (0.0, 0.04)),
        (0.0, 0.0, (0, 0)),
    ],
)
def test_basic_plot_rounded_limits_keep_sub_unit_ranges(low, high, expected):
    from openbench.visualization.Fig_Basic_Plot import rounded_limits

    assert rounded_limits(low, high) == pytest.approx(expected)


def test_get_index_ticks_resolve_ranges_below_one_hundredth():
    from openbench.visualization.Fig_toolbox import get_index

    _cmap, mticks, _norm, _bnd, _extend = get_index(0.0, 0.004, "viridis", "RMSE")

    assert list(mticks) == pytest.approx([0.0, 0.001, 0.002, 0.003, 0.004])


@pytest.mark.parametrize(("vmin", "vmax"), [(-0.004, 0.001), (-4, 1), (-1, 4)])
def test_get_index_centres_on_zero_when_range_spans_it(vmin, vmax):
    from openbench.visualization.Fig_toolbox import get_index

    _cmap, mticks, _norm, _bnd, _extend = get_index(vmin, vmax, "viridis", "bias")

    assert mticks[0] == pytest.approx(-mticks[-1])
    assert 0 in list(mticks)


@pytest.mark.parametrize(
    ("low", "high", "metric"),
    [
        (0.0, 4e-5, "RMSE"),
        (-4e-5, 1e-5, "bias"),
        (-3e-5, 3e-5, "bias"),
        (1.2e-9, 1.8e-9, "RMSE"),
        (0.0, 4e-24, "RMSE"),
        (0.0, 0.0, "RMSE"),
    ],
)
def test_tiny_automatic_colorbar_ranges_render_contours(low, high, metric):
    import matplotlib.pyplot as plt

    from openbench.visualization.Fig_Basic_Plot import rounded_limits
    from openbench.visualization.Fig_toolbox import get_index

    cmap, ticks, norm, levels, extend = get_index(*rounded_limits(low, high), "viridis", metric)

    assert len(ticks) >= 2
    assert np.all(np.diff(levels) > 0)
    assert norm.vmin < norm.vmax
    if low < high:
        assert norm(low) < norm(high)
    fig, ax = plt.subplots()
    try:
        ax.contourf(np.linspace(low, high, 4).reshape(2, 2), levels=levels, cmap=cmap, norm=norm, extend=extend)
    finally:
        plt.close(fig)
