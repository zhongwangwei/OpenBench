"""Shared coverage for only-drawing entry points, geometry, units and color scales."""

from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import xarray as xr
import yaml
from PIL import Image


# Representative cases replace the full group/count/angle/statistic product.
# Each group still exercises single/multiple metrics and single/multiple scores.
_LAYOUT_CASES = [
    ("IGBP", "metric", 1, 18, 45, 17),
    ("IGBP", "metric", 2, 18, 45, 17),
    ("IGBP", "score", 1, 18, 90, 22),
    ("IGBP", "score", 3, 18, 45, 17),
    ("PFT", "metric", 1, 16, 0, 22),
    ("PFT", "metric", 2, 18, 45, 17),
    ("PFT", "score", 1, 18, 45, 17),
    ("PFT", "score", 3, 16, 0, 22),
    ("CZ", "metric", 1, 16, 45, 17),
    ("CZ", "metric", 2, 31, 45, 17),
    ("CZ", "score", 1, 18, 90, 22),
    ("CZ", "score", 3, 31, 45, 17),
]


@pytest.mark.parametrize("group,kind,row_count,column_count,rotation,font", _LAYOUT_CASES)
def test_only_drawing_layout_and_colorbars(tmp_path, monkeypatch, group, kind, row_count,
                                          column_count, rotation, font):
    import openbench.visualization.Fig_LC_based_heat_map as plot
    import openbench.visualization.only_drawing as drawing
    from openbench.util.filenames import groupby_class_netcdf_stem

    statistics = (["bias", "RMSE"] if kind == "metric" else
                  ["Overall_Score", "nBiasScore", "nRMSEScore"])[:row_count]
    columns = [f"Class_{i:02d}" for i in range(column_count - 1)] + ["Overall"]
    rows = [f"# statistic_type: {kind}", "# ref_unit: mm day-1", "# sim_unit: mm day-1",
            "# weight: none", "# aggregation: test", "\t".join([kind, *columns])]
    values = np.linspace(0.1, 0.9, column_count)
    for statistic in statistics:
        rows.append("\t".join([statistic, *[f"{value:.2f}" for value in values]]))
    rows.append("\t".join(["n_valid", *[str(1000 + i) for i in range(column_count)]]))
    pair_dir = tmp_path / "comparisons" / f"{group}_groupby" / "Sim__Ref"
    pair_dir.mkdir(parents=True)
    source = pair_dir / f"Evapotranspiration__Sim__Ref__{kind}s.csv"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Multiple metrics exercise automatic ranges from the real class bundles;
    # single-row cases use custom ranges, which must not need NetCDF inputs.
    if kind == "metric" and row_count > 1:
        for statistic in statistics:
            stem = groupby_class_netcdf_stem("Evapotranspiration", "Ref", "Sim", statistic, group)
            xr.Dataset({statistic: (("class", "lat", "lon"), values.reshape(-1, 1, 1))}).to_netcdf(
                pair_dir / f"{stem}__classes.nc"
            )
    options = yaml.safe_load(
        Path(f"src/openbench/data/fignml/{group}_groupby_source.yaml").read_text(encoding="utf-8")
    )["general"]
    options.update(x_wise=6, y_wise=2, dpi=72, saving_format="png", xtick=font,
                   x_rotation=rotation, vmin_max_on=not (kind == "metric" and row_count > 1),
                   vmin=0, vmax=1, colorbar_position="vertical", extend="both")
    toolbox_scales = []
    original_get_index = plot.get_index

    def get_index(*args):
        scale = original_get_index(*args)
        toolbox_scales.append(scale)
        return scale

    monkeypatch.setattr(plot, "get_index", get_index)
    saved = []

    def save(fig, path, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        data_axes = [ax for ax in fig.axes if ax.images]
        expected_columns = [16, column_count - 16] if group == "CZ" and column_count > 16 else [column_count]
        metric_rows = row_count if kind == "metric" else 1
        assert [ax.images[0].get_array().shape[1] for ax in data_axes] == [
            count for count in expected_columns for _ in range(metric_rows)
        ]
        cell_widths = [ax.get_window_extent(renderer).width / ax.images[0].get_array().shape[1]
                       for ax in data_axes]
        cell_heights = [ax.get_window_extent(renderer).height / ax.images[0].get_array().shape[0]
                        for ax in data_axes]
        np.testing.assert_allclose(cell_heights, cell_widths)
        np.testing.assert_allclose(cell_widths, cell_widths[0])
        tick_boxes = []
        for ax in data_axes:
            assert ax.images[0].get_array().shape[0] == (1 if kind == "metric" else row_count)
            if not ax.xaxis.get_visible():
                continue
            for label in ax.get_xticklabels():
                assert "\nn=" in label.get_text()
                box = label.get_window_extent(renderer)
                tick_x = ax.get_xaxis_transform().transform((label.get_position()[0], 0))[0]
                assert (box.x0 + box.x1) / 2 == pytest.approx(tick_x, abs=1)
                tick_boxes.append(box)
        assert len(tick_boxes) == column_count
        for first, second in combinations(tick_boxes, 2):
            assert not first.overlaps(second)
        for box in tick_boxes:
            assert all(not box.overlaps(ax.get_window_extent(renderer)) for ax in data_axes)
        canvas = fig.bbox
        for ax in fig.axes:
            bounds = ax.get_tightbbox(renderer)
            assert bounds.x0 >= canvas.x0 - 1 and bounds.y0 >= canvas.y0 - 1
            assert bounds.x1 <= canvas.x1 + 1 and bounds.y1 <= canvas.y1 + 1
        colorbars = [ax.images[0].colorbar for ax in data_axes if ax.images[0].colorbar is not None]
        assert colorbars
        if kind == "metric" or row_count == 1:
            assert len(colorbars) == len(data_axes)
            for ax, colorbar in zip(data_axes, colorbars):
                assert colorbar.orientation == "horizontal"
                assert colorbar.ax.get_position().x0 > ax.get_position().x1
        else:
            assert colorbars[0].orientation == "vertical"
        y_labels = [label.get_text() for ax in data_axes for label in ax.get_yticklabels()]
        assert not any("n valid" in label for label in y_labels)
        if kind == "score":
            assert all(bar.extend == "both" for bar in colorbars)
            assert not any("mm" in label for label in y_labels)
        else:
            assert any("mm" in label for label in y_labels)
            assert len(toolbox_scales) == row_count
            for ax in data_axes:
                image = ax.images[0]
                scale = next(scale for scale in toolbox_scales if image.norm is scale[2])
                assert image.cmap is scale[0]
                np.testing.assert_array_equal(image.colorbar.get_ticks(), scale[1])
                assert image.colorbar.extend == scale[4]
        # One end-to-end PNG smoke check; other cases only render in memory.
        if group == "IGBP" and kind == "metric" and row_count == 1:
            with matplotlib.rc_context({"savefig.bbox": None}):
                fig.savefig(path, **kwargs)
            with Image.open(path) as image:
                assert image.size == (int(canvas.width), int(canvas.height))
        saved.append(path)

    monkeypatch.setattr(plot, "save_figure", save)
    main = {"general": dict(basedir=str(tmp_path.parent), basename=tmp_path.name,
                            compare_grid_res=1, compare_tim_res="month")}
    ref = {"general": {"Evapotranspiration_ref_source": "Ref"},
           "Evapotranspiration": {"Ref_data_type": "grid"}}
    sim = {"general": {"Evapotranspiration_sim_source": "Sim"},
           "Evapotranspiration": {"Sim_data_type": "grid"}}
    metrics, scores = (statistics, []) if kind == "metric" else ([], statistics)
    handler = (drawing.CZ_groupby_only_drawing if group == "CZ" else drawing.LC_groupby_only_drawing)(
        main, scores, metrics
    )
    getattr(handler, f"scenarios_{group}_groupby_comparison")(
        str(tmp_path), sim, ref, ["Evapotranspiration"], scores, metrics, options
    )
    assert saved == [str(source.with_name(source.stem + "_heatmap.png"))]


@pytest.mark.parametrize("metric,expected", [
    ("bias", (-2, 4)), ("RMSE", (0, 4)), ("correlation", (-1, 1)),
    ("KGE", (-1, 1)), ("MFM", (0, 1)),
])
def test_metric_scale_uses_basic_ranges_and_toolbox(monkeypatch, metric, expected):
    import openbench.visualization.Fig_LC_based_heat_map as plot

    data = xr.Dataset({metric: (("lat", "lon"), [[-2.2, 3.8]])})
    monkeypatch.setattr(plot, "_open_groupby_class_distribution", lambda *args: data)
    calls = []
    sentinel = object()

    def get_index(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(plot, "get_index", get_index)
    assert plot._metric_color_scale({"vmin_max_on": False, "cmap": "viridis"}, metric) is sentinel
    assert calls == [(*expected, "viridis", metric)]


def test_metric_scale_custom_range_uses_toolbox_without_netcdf(monkeypatch):
    import openbench.visualization.Fig_LC_based_heat_map as plot
    from openbench.visualization.Fig_toolbox import get_index

    def unexpected(*args):
        pytest.fail("A custom range should not need class NetCDF distributions")

    monkeypatch.setattr(plot, "_open_groupby_class_distribution", unexpected)
    actual = plot._metric_color_scale({"vmin_max_on": True, "vmin": -.7, "vmax": .7, "cmap": "viridis"}, "bias")
    expected = get_index(-.7, .7, "viridis", "bias")
    assert actual[0].name == expected[0].name
    np.testing.assert_array_equal(actual[1], expected[1])
    assert (actual[2].vmin, actual[2].vmax, actual[4]) == (expected[2].vmin, expected[2].vmax, expected[4])
