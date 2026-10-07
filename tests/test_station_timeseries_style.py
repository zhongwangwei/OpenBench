"""Station time-series lines keep one width whatever the record length."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


def _plot_stn_option():
    from importlib.resources import files

    text = (files("openbench.data.fignml") / "plot_stn.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)["general"]


def _drawn_figure(monkeypatch, tmp_path, n_points, freq, station_id="A", lat_lon=(10.0, 20.0)):
    import openbench.visualization.Fig_Basic_Plot as fig_basic

    figures = []
    monkeypatch.setattr(fig_basic, "save_figure", lambda fig, *args, **kwargs: figures.append(fig))
    times = pd.date_range("2000-01-01", periods=n_points, freq=freq)
    obs = xr.DataArray(np.linspace(1.0, 2.0, n_points), coords={"time": times}, dims="time")
    sim = obs * 1.1
    caller = SimpleNamespace(
        fig_nml={"plot_stn": _plot_stn_option()},
        casedir=str(tmp_path),
        ref_source="Ref",
        sim_source="Sim",
        item="Streamflow",
        ref_varunit="m3 s-1",
        sim_varunit="m3 s-1",
    )

    fig_basic.plot_stn(caller, sim, obs, station_id, ["Streamflow"], 0.1, 0.5, 0.9, list(lat_lon))
    return figures[0]


def _drawn_lines(monkeypatch, tmp_path, n_points, freq):
    obs_line, sim_line = _drawn_figure(monkeypatch, tmp_path, n_points, freq).axes[0].get_lines()[:2]
    return obs_line, sim_line


@pytest.mark.parametrize(("n_points", "freq"), [(12, "MS"), (144, "MS"), (3650, "D")])
def test_station_timeseries_line_width_does_not_depend_on_record_length(monkeypatch, tmp_path, n_points, freq):
    option = _plot_stn_option()
    obs_line, sim_line = _drawn_lines(monkeypatch, tmp_path, n_points, freq)

    assert obs_line.get_linewidth() == pytest.approx(option["obs_lineswidth"])
    assert sim_line.get_linewidth() == pytest.approx(option["sim_lineswidth"])


def test_station_timeseries_drops_markers_on_dense_records(monkeypatch, tmp_path):
    option = _plot_stn_option()

    obs_short, _ = _drawn_lines(monkeypatch, tmp_path, 36, "MS")
    obs_dense, _ = _drawn_lines(monkeypatch, tmp_path, option["marker_max_points"] + 1, "D")

    assert obs_short.get_marker() == option["obs_marker"]
    assert obs_short.get_markersize() == pytest.approx(option["obs_markersize"])
    assert obs_dense.get_marker() in (None, "None", "")


def test_station_timeseries_reads_legacy_total_widths_at_their_calibrated_length():
    from openbench.visualization.Fig_Basic_Plot import _stn_line_style

    legacy = {"obs_lineswidth": 144, "obs_markersize": 432, "obs_marker": "^"}

    assert _stn_line_style(legacy, "obs", 12) == (1.0, "^", 3.0)
    assert _stn_line_style(legacy, "obs", 3650) == (1.0, None, 3.0)


def test_station_timeseries_title_keeps_station_id_and_clears_the_metrics(monkeypatch, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text

    station_id = "464114097260900_USGS_LONGID"  # 27 characters, the longest ids in the data
    fig = _drawn_figure(monkeypatch, tmp_path, 36, "MS", station_id=station_id, lat_lon=(-46.7, -169.72))
    ax = fig.axes[0]
    # plot_stn closes the figure, and newer Matplotlib then detaches its canvas.
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    texts = [t for t in ax.get_children() if isinstance(t, Text) and t.get_text()]
    title = next(t for t in texts if t.get_text().startswith("ID: "))
    metrics = next(t for t in texts if t.get_text().startswith("RMSE: "))

    assert station_id in title.get_text()
    title_box, metrics_box = title.get_window_extent(renderer), metrics.get_window_extent(renderer)
    assert not title_box.overlaps(metrics_box)
    assert metrics_box.y0 >= ax.get_window_extent(renderer).y1
