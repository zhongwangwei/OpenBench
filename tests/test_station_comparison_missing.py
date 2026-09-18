"""Station comparison retains data gaps without dropping valid sources."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core._comparison_helpers import _station_frames_aligned_by_id, _station_pairwise_difference_by_id


def test_station_difference_retains_union_and_missing_coordinates():
    left = pd.DataFrame({"ID": ["A", "B"], "bias": [3.0, np.nan], "ref_lon": [1.0, 2.0]})
    right = pd.DataFrame({"ID": ["C", "A"], "bias": [4.0, 1.0], "ref_lon": [3.0, 1.0]})
    aligned = _station_frames_aligned_by_id({"left": left, "right": right}, "bias")
    assert aligned["left"].ID.tolist() == ["A", "B", "C"]
    assert aligned["left"].ref_lon.tolist() == [1.0, 2.0, 3.0]
    frame, diff = _station_pairwise_difference_by_id(left, right, "bias", left_label="left", right_label="right")
    assert frame.ID.tolist() == ["A", "B", "C"]
    assert diff.iloc[0] == 2 and diff.iloc[1:].isna().all()


def test_diff_dispatch_keeps_mixed_sources(tmp_path, monkeypatch):
    import openbench.core._comparison_diff_plot as module
    import openbench.core.comparison as comparison

    calls = []
    monkeypatch.setattr(
        module, "process_grid_diff_plot", lambda **kw: calls.append(("grid", kw["ref_source"], kw["sim_sources"]))
    )
    monkeypatch.setattr(
        module, "process_station_diff_plot", lambda **kw: calls.append(("stn", kw["ref_source"], kw["sim_sources"]))
    )
    monkeypatch.setattr(comparison, "make_scenarios_comparison_Diff_Plot", lambda *args: None)
    sim = {
        "general": {"Runoff_sim_source": ["GridSim", "StationSim"]},
        "Runoff": {"GridSim_data_type": "grid", "StationSim_data_type": "stn"},
    }
    ref = {
        "general": {"Runoff_ref_source": ["GridRef", "StationRef"]},
        "Runoff": {"GridRef_data_type": "grid", "StationRef_data_type": "stn"},
    }
    module.DiffPlotScenarioMixin.scenarios_Diff_Plot_comparison(
        SimpleNamespace(general_config={}), str(tmp_path), sim, ref, ["Runoff"], [], ["bias"], {}
    )
    assert ("grid", "GridRef", ["GridSim"]) in calls
    assert ("stn", "GridRef", ["StationSim"]) in calls
    assert ("stn", "StationRef", ["GridSim", "StationSim"]) in calls


def test_station_comparison_loads_recorded_gaps_and_single_time(tmp_path):
    from openbench.core._comparison_helpers import _load_station_pair, _station_evaluation_frame
    from openbench.data.station_missing import StationDataUnavailable

    directory = tmp_path / "data/stn_Ref_Sim"
    directory.mkdir(parents=True)
    (tmp_path / "scores").mkdir()
    status = pd.DataFrame(
        {
            "ID": ["A", "B"],
            "use_syear": [2000, 2000],
            "use_eyear": [2000, 2000],
            "status": ["ok", "unavailable"],
            "reason": ["", "missing flow"],
        }
    )
    status.to_csv(directory / "Runoff_evaluation_status.csv", index=False)
    pd.DataFrame({"ID": ["A"], "bias": [2.0]}).to_csv(
        tmp_path / "scores/Runoff_stn_Ref_Sim_evaluations.csv", index=False
    )
    frame = _station_evaluation_frame(str(tmp_path), "Runoff", "Ref", "Sim")
    assert frame.ID.tolist() == ["A", "B"] and np.isnan(frame.bias.iloc[1])
    handler = SimpleNamespace(compare_tim_res="Day")
    with pytest.raises(StationDataUnavailable, match="missing flow"):
        _load_station_pair(handler, str(tmp_path), "Runoff", "Ref", "Sim", frame.iloc[1], "flow", "flow")
    for role, hour in [("sim", "00"), ("ref", "12")]:
        xr.Dataset({"flow": ("time", [1.0])}, coords={"time": [np.datetime64(f"2000-01-01T{hour}:00")]}).to_netcdf(
            directory / f"Runoff_{role}_A_2000_2000.nc"
        )
    s, o = _load_station_pair(handler, str(tmp_path), "Runoff", "Ref", "Sim", frame.iloc[0], "flow", "flow")
    assert s.sizes["time"] == o.sizes["time"] == 1
    xr.testing.assert_equal(s.time, o.time)


@pytest.mark.parametrize("ref_type,sim_type", [("stn", "grid"), ("grid", "stn"), ("stn", "stn")])
def test_basic_station_pair_keeps_missing_rows(tmp_path, monkeypatch, ref_type, sim_type):
    import openbench.core.comparison as comparison
    from openbench.core._comparison_basic import BasicComparisonMixin

    (tmp_path / "scores").mkdir()
    directory = tmp_path / "data/stn_Ref_Sim"
    directory.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "ID": ["A", "B"],
            "sim_lat": [1.0, 2.0],
            "sim_lon": [3.0, 4.0],
            "use_syear": [2000, 2000],
            "use_eyear": [2000, 2000],
            "status": ["ok", "unavailable"],
            "reason": ["", "missing flow"],
        }
    )
    rows.to_csv(directory / "Runoff_evaluation_status.csv", index=False)
    rows.iloc[:1].assign(bias=1.0).drop(columns=["status", "reason"]).to_csv(
        tmp_path / "scores/Runoff_stn_Ref_Sim_evaluations.csv", index=False
    )
    for role, values in [("sim", [2.0, 4.0]), ("ref", [1.0, 3.0])]:
        xr.Dataset({"flow": ("time", values)}, coords={"time": pd.date_range("2000-01-01", periods=2)}).to_netcdf(
            directory / f"Runoff_{role}_A_2000_2000.nc"
        )
    handler = SimpleNamespace(
        compare_tim_res="Day",
        main_nml={"general": {}},
        stat_mean=lambda data: data.mean("time"),
        _run_parallel_or_serial=lambda tasks: [fn(*args, **kwargs) for fn, args, kwargs in tasks],
    )
    monkeypatch.setattr(comparison, "make_stn_plot_index", lambda *args: None)
    sim = {"general": {"Runoff_sim_source": "Sim"}, "Runoff": {"Sim_data_type": sim_type, "Sim_varname": "flow"}}
    ref = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": ref_type, "Ref_varname": "flow"}}
    BasicComparisonMixin.scenarios_Basic_comparison(
        handler, str(tmp_path), sim, ref, ["Runoff"], [], [], {"key": "Mean"}
    )
    result = pd.read_csv(tmp_path / "comparisons/Mean/Runoff_stn_Ref_Sim_Mean.csv")
    assert result.ID.tolist() == ["A", "B"]
    assert result.ref_value.iloc[0] == 2 and result.sim_value.iloc[0] == 3
    assert result[["ref_value", "sim_value"]].iloc[1].isna().all()
    assert result.reason.iloc[1] == "missing flow"


def test_station_diff_gap_rows_have_reason_not_fake_zero(tmp_path):
    from openbench.core._comparison_diff_station import process_station_diff_plot

    (tmp_path / "metrics").mkdir()
    output = tmp_path / "comparisons/Diff_Plot"
    output.mkdir(parents=True)
    for source, ids, values in [("A", ["S1", "S2"], [3.0, np.nan]), ("B", ["S1", "S3"], [1.0, 5.0])]:
        pd.DataFrame({"ID": ids, "bias": values, "ref_lat": [1.0, 2.0], "ref_lon": [3.0, 4.0]}).to_csv(
            tmp_path / f"metrics/Runoff_stn_Ref_{source}_evaluations.csv", index=False
        )
    process_station_diff_plot(
        basedir=str(tmp_path),
        dir_path=str(output),
        evaluation_item="Runoff",
        ref_source="Ref",
        sim_sources=["A", "B"],
        metrics=["bias"],
        scores=[],
        sim_nml={"Runoff": {"A_varname": "flow", "B_varname": "flow"}},
    )
    difference = pd.read_csv(next(output.glob("*diff*.csv")))
    assert difference.ID.tolist() == ["S1", "S2", "S3"]
    assert difference.bias_diff.iloc[0] == 2
    assert difference.bias_diff.iloc[1:].isna().all()
    assert difference.reason.iloc[1:].notna().all()
    for path in output.glob("*anomaly*.csv"):
        anomaly = pd.read_csv(path)
        assert anomaly.bias_anomaly.iloc[1:].isna().all()


def test_only_drawing_retains_mixed_station_source_and_recorded_nan(tmp_path, monkeypatch):
    import openbench.visualization.only_drawing as drawing
    from openbench.util.filenames import diff_grid_anomaly_filename, diff_station_anomaly_filename

    output = tmp_path / "comparisons/Diff_Plot"
    output.mkdir(parents=True)
    pd.DataFrame(
        {
            "ID": ["A"],
            "lat": [1.0],
            "lon": [2.0],
            "bias_anomaly": [np.nan],
            "status": ["unavailable"],
            "reason": ["missing variable"],
        }
    ).to_csv(output / diff_station_anomaly_filename("Runoff", "Ref", "Station", "bias"), index=False)
    xr.Dataset({"bias_anomaly": (("lat", "lon"), [[1.0]])}, coords={"lat": [1.0], "lon": [2.0]}).to_netcdf(
        output / diff_grid_anomaly_filename("Runoff", "Ref", "Grid", "bias")
    )
    sim = {
        "general": {"Runoff_sim_source": ["Grid", "Station"]},
        "Runoff": {"Grid_data_type": "grid", "Station_data_type": "stn"},
    }
    ref = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": "grid"}}
    calls = []
    monkeypatch.setattr(drawing, "make_scenarios_comparison_Diff_Plot", lambda *args: calls.append((args[5], args[8])))
    drawing.ComparisonProcessing_only_drawing.scenarios_Diff_Plot_comparison(
        SimpleNamespace(general_config={}), str(tmp_path), sim, ref, ["Runoff"], [], ["bias"], {}
    )
    assert calls == [(["Station"], "stn"), (["Grid"], "grid")]


def test_portrait_explicit_missing_cells_have_no_numeric_scale():
    import matplotlib.pyplot as plt

    from openbench.visualization.Fig_portrait_plot_seasonal import portrait_plot

    fig, ax, _ = portrait_plot(np.full((4, 1, 1), np.nan), ["Sim"], ["bias"], allow_empty=True)
    try:
        assert len(fig.axes) == 1
        assert "N/A" in [text.get_text() for text in ax.texts]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("name", ["Kernel_Density_Estimate", "Whisker_Plot", "Ridgeline_Plot"])
def test_only_drawing_distributions_cannot_revive_stale_station_values(tmp_path, monkeypatch, name):
    import openbench.visualization.only_drawing as drawing

    (tmp_path / "scores").mkdir()
    directory = tmp_path / "data/stn_Ref_Sim"
    directory.mkdir(parents=True)
    pd.DataFrame({"ID": ["A", "B", "C"], "Overall_Score": [0.8, 0.1, 0.6]}).to_csv(
        tmp_path / "scores/Runoff_stn_Ref_Sim_evaluations.csv", index=False
    )
    pd.DataFrame(
        {"ID": ["A", "B", "C"], "status": ["ok", "unavailable", "ok"], "reason": ["", "missing variable", ""]}
    ).to_csv(directory / "Runoff_evaluation_status.csv", index=False)
    sim = {"general": {"Runoff_sim_source": ["Sim"]}, "Runoff": {"Sim_data_type": "stn", "Sim_varname": "flow"}}
    ref = {"general": {"Runoff_ref_source": ["Ref"]}, "Runoff": {"Ref_data_type": "grid", "Ref_varname": "flow"}}
    calls = []
    monkeypatch.setattr(drawing, f"make_scenarios_comparison_{name}", lambda *args: calls.append(args[5]))
    getattr(drawing.ComparisonProcessing_only_drawing, f"scenarios_{name}_comparison")(
        SimpleNamespace(), str(tmp_path), sim, ref, ["Runoff"], ["Overall_Score"], [], {}
    )
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], [0.8, 0.6])


def test_basic_partial_statistic_keeps_valid_role_and_redraws_na(tmp_path, monkeypatch):
    import openbench.core._comparison_basic as basic
    from openbench.visualization.only_drawing import _require_station_csv_values

    monkeypatch.setattr(basic, "_station_evaluation_frame", lambda *args: pd.DataFrame({"ID": ["A"]}))
    monkeypatch.setattr(basic, "_load_station_pair", lambda *args: (1.0, np.nan))
    monkeypatch.setattr(basic, "_comparison_callable", lambda name: lambda *args: None)
    handler = SimpleNamespace(
        main_nml={"general": {}},
        stat_mean=lambda data: data,
        _run_parallel_or_serial=lambda tasks: [fn(*args, **kw) for fn, args, kw in tasks],
    )
    sim = {"general": {"Runoff_sim_source": "Sim"}, "Runoff": {"Sim_data_type": "stn", "Sim_varname": "flow"}}
    ref = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": "stn", "Ref_varname": "flow"}}
    basic.BasicComparisonMixin.scenarios_Basic_comparison(
        handler, str(tmp_path), sim, ref, ["Runoff"], [], [], {"key": "Mean"}
    )
    path = tmp_path / "comparisons/Mean/Runoff_stn_Ref_Sim_Mean.csv"
    result = pd.read_csv(path)
    assert result.status.tolist() == ["partial"]
    assert result.status_ref_value.tolist() == ["unavailable"]
    assert result.status_sim_value.tolist() == ["ok"]
    _require_station_csv_values(str(path), "ref_value")
    _require_station_csv_values(str(path), "sim_value")
