import numpy as np
import pytest


def test_sample_series_for_plot_strides_large_arrays():
    from openbench.visualization._sampling import sample_series_for_plot

    data = np.arange(100)

    out = sample_series_for_plot(data, {"max_samples_per_series": 10})

    assert len(out) == 10
    assert np.array_equal(out, data[::10])


def test_sample_series_for_plot_uses_kde_limit():
    from openbench.visualization._sampling import sample_series_for_plot

    data = np.arange(100)

    out = sample_series_for_plot(data, {"max_samples_per_series": 80, "kde_max_samples": 5}, purpose="kde")

    assert len(out) == 5
    assert np.array_equal(out, data[::20])


def test_sample_series_for_plot_native_keeps_full_series():
    from openbench.visualization._sampling import sample_series_for_plot

    data = np.arange(100)

    out = sample_series_for_plot(data, {"plotting_mode": "full", "max_samples_per_series": 5})

    assert np.array_equal(out, data)


def test_sample_distribution_series_applies_to_each_series():
    from openbench.visualization._sampling import sample_distribution_series

    series = [np.arange(100), np.arange(50)]

    out = sample_distribution_series(series, {"max_samples_per_series": 10})

    assert [len(item) for item in out] == [10, 10]


def test_kde_plot_samples_each_series_before_kde(tmp_path, monkeypatch):
    import openbench.visualization.Fig_kernel_density_estimate as fig_kde

    observed_lengths = []

    class FakeKDE:
        def __init__(self, data):
            observed_lengths.append(len(data))
            self.covariance = np.array([[1.0]])

        def __call__(self, values):
            return np.ones_like(values, dtype=float)

    monkeypatch.setattr(fig_kde, "gaussian_kde", FakeKDE)

    fig_kde.make_scenarios_comparison_Kernel_Density_Estimate(
        str(tmp_path),
        "Runoff",
        "RefA",
        ["SimA"],
        "bias",
        [np.arange(100, dtype=float)],
        {
            "font": "DejaVu Sans",
            "axes_linewidth": 1,
            "fontsize": 8,
            "xtick": 8,
            "ytick": 8,
            "line_width": 1,
            "x_wise": 4,
            "y_wise": 3,
            "linestyle": "solid",
            "linewidth": 1,
            "alpha": 0.2,
            "ncol": 1,
            "set_legend": False,
            "loc": "best",
            "bbox_to_anchor_x": 1,
            "bbox_to_anchor_y": 1,
            "grid": False,
            "grid_style": "dotted",
            "grid_linewidth": 0.7,
            "title": "",
            "xticklabel": "",
            "yticklabel": "",
            "title_fontsize": 10,
            "saving_format": "png",
            "dpi": 80,
            "kde_max_samples": 10,
        },
    )

    assert observed_lengths == [10]


def test_kde_plot_rejects_all_nan_input_before_min_or_kde(tmp_path, monkeypatch):
    import openbench.visualization.Fig_kernel_density_estimate as fig_kde

    def fail_kde(data):
        raise AssertionError("gaussian_kde should not be called for all-NaN input")

    monkeypatch.setattr(fig_kde, "gaussian_kde", fail_kde)

    with pytest.raises(ValueError, match="no finite data to plot"):
        fig_kde.make_scenarios_comparison_Kernel_Density_Estimate(
            str(tmp_path),
            "Runoff",
            "RefA",
            ["SimA"],
            "bias",
            [np.array([np.nan, np.nan])],
            {
                "font": "DejaVu Sans",
                "axes_linewidth": 1,
                "fontsize": 8,
                "xtick": 8,
                "ytick": 8,
                "line_width": 1,
                "x_wise": 4,
                "y_wise": 3,
                "linestyle": "solid",
                "linewidth": 1,
                "alpha": 0.2,
                "ncol": 1,
                "set_legend": False,
                "loc": "best",
                "bbox_to_anchor_x": 1,
                "bbox_to_anchor_y": 1,
                "grid": False,
                "grid_style": "dotted",
                "grid_linewidth": 0.7,
                "title": "",
                "xticklabel": "",
                "yticklabel": "",
                "title_fontsize": 10,
                "saving_format": "png",
                "dpi": 80,
                "kde_max_samples": 10,
            },
        )


def test_whisker_plot_samples_before_boxplot(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    import openbench.visualization.Fig_Whisker_Plot as fig_whisker

    observed_lengths = []

    def fake_boxplot(self, data, *args, **kwargs):
        observed_lengths.extend(len(item) for item in data)
        return {"boxes": []}

    monkeypatch.setattr(Axes, "boxplot", fake_boxplot)
    monkeypatch.setattr(fig_whisker, "save_figure", lambda *args, **kwargs: None)

    fig_whisker.make_scenarios_comparison_Whisker_Plot(
        str(tmp_path),
        "Runoff",
        "RefA",
        ["SimA"],
        "bias",
        [np.arange(100, dtype=float)],
        {
            "font": "DejaVu Sans",
            "axes_linewidth": 1,
            "xtick": 8,
            "ytick": 8,
            "line_width": 1,
            "x_wise": 4,
            "y_wise": 3,
            "boxpropslinewidth": 1,
            "patch_artist": False,
            "boxpropsedgecolor": "k",
            "vert": True,
            "showfliers": True,
            "flierpropsmarker": "o",
            "flierpropsmarkerfacecolor": "FFFFFF",
            "flierpropsmarkersize": 5,
            "flierpropsmarkeredgecolor": "k",
            "flierpropsmarkeredgewidth": 1,
            "box_widths": 0.2,
            "box_showmeans": True,
            "meanline": True,
            "meanpropslinestyle": "--",
            "meanpropslinewidth": 1,
            "meanpropscolor": "k",
            "medianpropslinestyle": "-",
            "medianpropslinewidth": 1,
            "medianpropscolor": "k",
            "whiskerpropslinestyle": "--",
            "whiskerpropslinewidth": 1,
            "whiskerpropscolor": "k",
            "cappropslinestyle": "-",
            "cappropslinewidth": 1,
            "cappropscolor": "k",
            "x_rotation": 0,
            "y_rotation": 0,
            "ha": "center",
            "xticklabel": "",
            "yticklabel": "",
            "grid": False,
            "grid_style": "dashed",
            "grid_linewidth": 1,
            "limit_on": False,
            "value_min": -1,
            "value_max": 1,
            "title": "",
            "title_fontsize": 10,
            "saving_format": "png",
            "dpi": 80,
            "max_samples_per_series": 10,
        },
    )

    assert observed_lengths == [10]


def test_default_distribution_fig_options_expose_sampling_controls():
    from openbench.config.adapter import build_fig_nml

    fig_nml = build_fig_nml()

    for section in (
        fig_nml["Comparison"]["Kernel_Density_Estimate"],
        fig_nml["Comparison"]["Ridgeline_Plot"],
        fig_nml["Comparison"]["Whisker_Plot"],
    ):
        assert section["plotting_mode"] == "balanced"
        assert section["max_samples_per_series"] == 50_000
        assert section["kde_max_samples"] == 10_000
        assert section["sample_method"] == "stride"
