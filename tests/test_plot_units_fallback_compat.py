from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.data import unit
from openbench.data.registry.manager import RegistryManager
from openbench.data.unit import UnitProcessing


class _StationProcessor:
    from openbench.data._processing_station_core import StationProcessingCoreMixin

    class Processor(StationProcessingCoreMixin):
        item = "Latent_Heat"
        ref_source = "FLUXNET_PLUMBER2"
        ref_varname = "Qle_cor"
        ref_varunit = "W m-2"
        compare_tim_res = "Hour"

        def __init__(self):
            mapping = RegistryManager().get_reference("FLUXNET_PLUMBER2").variables["Latent_Heat"]
            self.FLUXNET_PLUMBER2_fallbacks = [fb.to_dict() for fb in mapping.fallbacks]
            self.units_seen = []
            self.compute_calls = 0

        def _is_climatology_mode(self):
            return True

        def _try_compute_from_profile(self, *args, **kwargs):
            # FLUXNET_PLUMBER2 has no catalog compute for Latent_Heat.
            self.compute_calls += 1
            return None

        def check_coordinate(self, ds):
            return ds

        def check_dataset_time_integrity(self, ds, *args, **kwargs):
            return ds

        def process_units(self, ds, unit_name):
            self.units_seen.append(unit_name)
            return ds, unit_name

        def select_timerange(self, ds, *args, **kwargs):
            return ds


def _station_ds(**variables):
    time = pd.date_range("2000-01-01", periods=2, freq="h")
    return xr.Dataset({name: ("time", values) for name, values in variables.items()}, coords={"time": time})


def test_fluxnet_plumber2_catalog_has_raw_heat_fallbacks():
    ref = RegistryManager().get_reference("FLUXNET_PLUMBER2")

    latent = ref.variables["Latent_Heat"]
    sensible = ref.variables["Sensible_Heat"]

    assert latent.varname == "Qle_cor"
    assert [fb.varname for fb in latent.fallbacks] == ["Qle"]
    assert sensible.varname == "Qh_cor"
    assert [fb.varname for fb in sensible.fallbacks] == ["Qh"]


def test_fluxnet_plumber2_primary_wins_over_raw_fallback():
    proc = _StationProcessor.Processor()
    ds = _station_ds(Qle=[1.0, 1.0], Qle_cor=[2.0, 3.0])

    out = proc.process_single_station_data(ds, 2000, 2000, "ref")

    np.testing.assert_allclose(out.values, [2.0, 3.0])
    assert proc.ref_varname == "Qle_cor"
    assert proc.units_seen == ["W m-2"]


def test_fluxnet_plumber2_raw_fallback_is_used_when_corrected_missing():
    proc = _StationProcessor.Processor()
    ds = _station_ds(Qle=[4.0, 5.0])

    out = proc.process_single_station_data(ds, 2000, 2000, "ref")

    np.testing.assert_allclose(out.values, [4.0, 5.0])
    assert proc.ref_varname == "Qle_cor"
    assert proc.units_seen == ["W m-2"]
    assert proc.compute_calls == 0  # catalog fallback runs before compute


def test_unit_aliases_convert_exactly():
    unit._UNIT_LOOKUP_CACHE = None

    converted, base_unit = UnitProcessing.convert_unit(7.0, "gc m-2 d-1")
    assert base_unit == "gc m-2 day-1"
    assert converted == 7.0

    converted, base_unit = UnitProcessing.convert_unit(25.0, "degrees c")
    assert base_unit == "k"
    assert converted == 298.15


class _Axis:
    def __init__(self):
        self.kwargs = []
        self.titles = []

    def set_major_formatter(self, *args, **kwargs):
        pass

    def set_ticks_position(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass


class _Axes:
    def __init__(self):
        self.spines = {"left": SimpleNamespace(set_linewidth=lambda *args: None)}
        self.xaxis = _Axis()
        self.yaxis = _Axis()
        self.scatter_kwargs = []
        self.titles = []

    def scatter(self, *args, **kwargs):
        self.scatter_kwargs.append(kwargs)
        assert not ("norm" in kwargs and ("vmin" in kwargs or "vmax" in kwargs))
        return SimpleNamespace()

    def add_feature(self, *args, **kwargs):
        pass

    def gridlines(self, *args, **kwargs):
        pass

    def set_extent(self, *args, **kwargs):
        pass

    def set_xticks(self, *args, **kwargs):
        pass

    def set_yticks(self, *args, **kwargs):
        pass

    def tick_params(self, *args, **kwargs):
        pass

    def set_adjustable(self, *args, **kwargs):
        pass

    def set_aspect(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def set_title(self, title, *args, **kwargs):
        self.titles.append(title)

    def get_position(self):
        return SimpleNamespace(x0=0.1, x1=0.8, y0=0.1, width=0.7, height=0.7)


class _Figure:
    def __init__(self, ax):
        self.ax = ax
        self.colorbar_calls = []

    def add_subplot(self, *args, **kwargs):
        return self.ax

    def add_axes(self, *args, **kwargs):
        return SimpleNamespace()

    def colorbar(self, *args, **kwargs):
        self.colorbar_calls.append(kwargs)
        return SimpleNamespace(solids=SimpleNamespace(set_edgecolor=lambda *args: None))


def _diff_option():
    return {
        "font": "DejaVu Sans",
        "labelsize": 8,
        "xtick": 8,
        "ytick": 8,
        "vmin_max_on": False,
        "cmap": "viridis",
        "x_wise": 4,
        "y_wise": 3,
        "markersize": 20,
        "marker": "o",
        "line_width": 1,
        "max_lat": 90,
        "min_lat": -90,
        "max_lon": 180,
        "min_lon": -180,
        "set_lat_lon": False,
        "xticklabel": "",
        "yticklabel": "",
        "title": "Station diff",
        "title_size": 10,
        "colorbar_position_set": False,
        "colorbar_position": "vertical",
        "saving_format": "png",
        "dpi": 80,
    }


def test_diff_station_scatter_uses_norm_without_vmin_vmax(monkeypatch, tmp_path):
    pytest.importorskip("cartopy")
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    ax = _Axes()
    fig = _Figure(ax)
    monkeypatch.setattr(fig_diff.plt, "figure", lambda *args, **kwargs: fig)
    monkeypatch.setattr(fig_diff.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(fig_diff, "save_figure", lambda *args, **kwargs: None)

    fig_diff.plot_stn_map(
        str(tmp_path),
        "bias_anomaly.csv",
        np.array([10.0, 20.0]),
        np.array([0.0, 1.0]),
        np.array([np.nan, 2.0]),
        {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
        "bias",
        "bias_anomaly",
        _diff_option(),
    )

    assert "norm" in ax.scatter_kwargs[0]
    assert "vmin" not in ax.scatter_kwargs[0]
    assert "vmax" not in ax.scatter_kwargs[0]
    assert fig.colorbar_calls[0]["extend"] in {"neither", "min", "max", "both"}


def test_diff_station_all_nan_renders_neutral_without_colorbar(monkeypatch, tmp_path):
    pytest.importorskip("cartopy")
    import openbench.visualization.Fig_Diff_Plot as fig_diff

    ax = _Axes()
    fig = _Figure(ax)
    monkeypatch.setattr(fig_diff.plt, "figure", lambda *args, **kwargs: fig)
    monkeypatch.setattr(fig_diff.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(fig_diff, "save_figure", lambda *args, **kwargs: None)

    fig_diff.plot_stn_map(
        str(tmp_path),
        "bias_anomaly.csv",
        np.array([10.0, 20.0]),
        np.array([0.0, 1.0]),
        np.array([np.nan, np.nan]),
        {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
        "bias",
        "bias_anomaly",
        _diff_option(),
    )

    assert ax.scatter_kwargs[0]["color"] == "#d9d9d9"
    assert "No valid paired data" in ax.titles[-1]
    assert fig.colorbar_calls == []


def test_whisker_labels_are_applied_without_removed_boxplot_keyword(monkeypatch, tmp_path):
    from matplotlib.axes import Axes

    import openbench.visualization.Fig_Whisker_Plot as fig_whisker

    seen = {}

    def fake_boxplot(self, data, *args, **kwargs):
        seen.update(kwargs)
        return {"boxes": []}

    monkeypatch.setattr(Axes, "boxplot", fake_boxplot)
    monkeypatch.setattr(fig_whisker, "save_figure", lambda *args, **kwargs: None)

    option = {
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
        "whiskerpropslinestyle": "-",
        "whiskerpropslinewidth": 1,
        "whiskerpropscolor": "k",
        "cappropslinestyle": "-",
        "cappropslinewidth": 1,
        "cappropscolor": "k",
        "x_rotation": 0,
        "y_rotation": 0,
        "ha": "center",
        "grid": False,
        "grid_style": "dotted",
        "grid_linewidth": 0.7,
        "limit_on": False,
        "value_min": None,
        "value_max": None,
        "xticklabel": "",
        "yticklabel": "",
        "title": "",
        "title_fontsize": 10,
        "saving_format": "png",
        "dpi": 80,
    }
    fig_whisker.make_scenarios_comparison_Whisker_Plot(
        str(tmp_path), "Runoff", "RefA", ["SimA"], "bias", [np.arange(5, dtype=float)], option
    )

    assert "labels" not in seen
    assert "tick_labels" not in seen
