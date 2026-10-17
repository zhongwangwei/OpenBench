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


def _drawn_figure(
    monkeypatch,
    tmp_path,
    n_points,
    freq,
    station_id="A",
    lat_lon=(10.0, 20.0),
    obs_values=None,
    sim_values=None,
    scalar_coords=None,
):
    import openbench.visualization.Fig_Basic_Plot as fig_basic

    figures = []
    monkeypatch.setattr(fig_basic, "save_figure", lambda fig, *args, **kwargs: figures.append(fig))
    times = pd.date_range("2000-01-01", periods=n_points, freq=freq)
    values = np.linspace(1.0, 2.0, n_points) if obs_values is None else obs_values
    obs = xr.DataArray(values, coords={"time": times}, dims="time").assign_coords(scalar_coords or {})
    sim = obs * 1.1 if sim_values is None else xr.DataArray(sim_values, coords={"time": times}, dims="time")
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


def test_station_timeseries_title_replaces_the_xarray_coordinate_title(monkeypatch, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Station series carry scalar coordinates; xarray turns them into a centred title.
    coords = {"lat": 35.62, "lon": -89.38, "variable": "f_discharge"}
    fig = _drawn_figure(
        monkeypatch, tmp_path, 120, "MS", station_id="US_0005359", lat_lon=(35.62, -89.38), scalar_coords=coords
    )
    ax = fig.axes[0]
    FigureCanvasAgg(fig).draw()

    assert ax.get_title(loc="center") == ""
    assert ax.get_title(loc="right") == ""
    assert ax.get_title(loc="left").startswith("ID: US_0005359")


def _every_other_day(n_points=365):
    values = np.linspace(1.0, 2.0, n_points)
    values[1::2] = np.nan
    return values


def test_station_timeseries_marks_isolated_values_of_dense_records(monkeypatch, tmp_path):
    option = _plot_stn_option()
    values = _every_other_day()
    fig = _drawn_figure(monkeypatch, tmp_path, len(values), "D", obs_values=values)
    obs_line = fig.axes[0].get_lines()[0]

    assert len(values) > option["marker_max_points"]
    assert obs_line.get_marker() == option["obs_marker"]
    assert list(obs_line.get_markevery()) == list(range(0, len(values), 2))
    assert obs_line.get_linewidth() == pytest.approx(option["obs_lineswidth"])


def test_station_timeseries_draws_records_without_adjacent_values(monkeypatch, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def obs_pixels(obs_values):
        missing = np.full(obs_values.size, np.nan)
        fig = _drawn_figure(monkeypatch, tmp_path, obs_values.size, "D", obs_values=obs_values, sim_values=missing)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3].astype(int)
        # the obs colour F96969 at alpha 0.8 over white
        return int(np.count_nonzero((rgb[..., 0] > 200) & (rgb[..., 1] < 170) & (rgb[..., 2] < 170)))

    legend_only = obs_pixels(np.full(365, np.nan))

    assert obs_pixels(_every_other_day()) > legend_only + 183


def test_station_timeseries_leaves_dense_records_without_gaps_unmarked(monkeypatch, tmp_path):
    fig = _drawn_figure(monkeypatch, tmp_path, 3650, "D")
    obs_line = fig.axes[0].get_lines()[0]

    assert obs_line.get_marker() in (None, "None", "")
    assert obs_line.get_markevery() is None
