"""Station regressions for Correlation scenario comparison."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core._comparison_tail import TailComparisonMixin
from openbench.core.statistics.stat_correlation import stat_correlation


class _CorrelationHarness(TailComparisonMixin):
    stat_correlation = stat_correlation

    def __init__(self):
        self.compare_nml = {}
        self.main_nml = {"general": {"compare_tim_res": "Day"}}
        self.compare_tim_res = "Day"
        self.time_alignment = "intersection"

    def save_result(self, output_file, method_name, result):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(output_file)


def _write_grid(path, varname, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {varname: (("time", "lat", "lon"), np.asarray(values, dtype=float).reshape(3, 1, 1))},
        coords={"time": pd.date_range("2000-01-01", periods=3), "lat": [1.0], "lon": [2.0]},
    ).to_netcdf(path)


def _write_station_pair(basedir, ref, sim, ids, status=None):
    directory = basedir / "data" / f"stn_{ref}_{sim}"
    directory.mkdir(parents=True, exist_ok=True)
    (basedir / "metrics").mkdir(exist_ok=True)
    rows = pd.DataFrame(
        {
            "ID": ids,
            "sim_lat": np.arange(len(ids), dtype=float) + 10.0,
            "sim_lon": np.arange(len(ids), dtype=float) + 100.0,
            "ref_lat": np.arange(len(ids), dtype=float) + 10.0,
            "ref_lon": np.arange(len(ids), dtype=float) + 100.0,
            "use_syear": [2000] * len(ids),
            "use_eyear": [2000] * len(ids),
        }
    )
    rows.to_csv(basedir / "metrics" / f"Runoff_stn_{ref}_{sim}_evaluations.csv", index=False)
    if status is not None:
        pd.DataFrame(status).to_csv(directory / "Runoff_evaluation_status.csv", index=False)


def _write_station_series(basedir, ref, sim, station_id, sim_values):
    directory = basedir / "data" / f"stn_{ref}_{sim}"
    times = pd.date_range("2000-01-01", periods=3)
    xr.Dataset({"flow": ("time", [1.0, 2.0, 3.0])}, coords={"time": times}).to_netcdf(
        directory / f"Runoff_ref_{station_id}_2000_2000.nc"
    )
    xr.Dataset({"flow": ("time", sim_values)}, coords={"time": times}).to_netcdf(
        directory / f"Runoff_sim_{station_id}_2000_2000.nc"
    )


def test_correlation_keeps_grid_pair_when_station_source_is_present(tmp_path, monkeypatch):
    basedir = tmp_path / "case"
    _write_grid(basedir / "data/Runoff_sim_Grid1_flow.nc", "flow", [1.0, 2.0, 3.0])
    _write_grid(basedir / "data/Runoff_sim_Grid2_flow.nc", "flow", [1.0, 3.0, 5.0])
    _write_station_pair(
        basedir,
        "Ref",
        "Station",
        ["A", "B"],
        {"ID": ["A", "B"], "status": ["ok", "unavailable"], "reason": ["", "missing station"]},
    )
    calls = []
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable", lambda name: lambda *args, **kwargs: calls.append(args)
    )
    handler = _CorrelationHarness()
    sim_nml = {
        "general": {"Runoff_sim_source": ["Grid1", "Station", "Grid2"]},
        "Runoff": {
            "Grid1_data_type": "grid",
            "Grid1_varname": "flow",
            "Grid2_data_type": "grid",
            "Grid2_varname": "flow",
            "Station_data_type": "stn",
            "Station_varname": "flow",
        },
    }
    ref_nml = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": "grid", "Ref_varname": "flow"}}

    handler.scenarios_Correlation_comparison(str(basedir), sim_nml, ref_nml, ["Runoff"], [], [], {})

    assert (basedir / "comparisons/Correlation/Correlation_Runoff_Grid1_and_Grid2.nc").exists()
    station = pd.read_csv(basedir / "comparisons/Correlation/Correlation_Runoff_stn_Ref_Grid1_and_Station.csv")
    assert station.ID.tolist() == ["A", "B"]
    assert station.Correlation.isna().all()
    assert station.status.tolist() == ["unavailable", "unavailable"]
    assert station.reason.str.contains("preprocessed station series").all()


def test_station_correlation_uses_station_union_and_real_common_times(tmp_path, monkeypatch):
    basedir = tmp_path / "case"
    _write_station_pair(
        basedir,
        "Ref",
        "Sim1",
        ["A", "B"],
        {"ID": ["A", "B"], "status": ["ok", "unavailable"], "reason": ["", "missing sim1"]},
    )
    _write_station_pair(
        basedir,
        "Ref",
        "Sim2",
        ["A", "C"],
        {"ID": ["A", "C"], "status": ["ok", "unavailable"], "reason": ["", "missing sim2"]},
    )
    _write_station_series(basedir, "Ref", "Sim1", "A", [1.0, 2.0, 3.0])
    _write_station_series(basedir, "Ref", "Sim2", "A", [1.0, 3.0, 5.0])
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable", lambda name: lambda *args, **kwargs: None
    )
    handler = _CorrelationHarness()
    sim_nml = {
        "general": {"Runoff_sim_source": ["Sim1", "Sim2"]},
        "Runoff": {
            "Sim1_data_type": "stn",
            "Sim1_varname": "flow",
            "Sim2_data_type": "stn",
            "Sim2_varname": "flow",
        },
    }
    ref_nml = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": "grid", "Ref_varname": "flow"}}

    handler.scenarios_Correlation_comparison(str(basedir), sim_nml, ref_nml, ["Runoff"], [], [], {})

    output = pd.read_csv(basedir / "comparisons/Correlation/Correlation_Runoff_stn_Ref_Sim1_and_Sim2.csv")
    assert output.ID.tolist() == ["A", "B", "C"]
    assert output.loc[0, "Correlation"] == pytest.approx(1.0)
    assert output.loc[0, "status"] == "ok"
    assert output.loc[1:, "Correlation"].isna().all()
    assert output.loc[1:, "status"].tolist() == ["unavailable", "unavailable"]
    assert "missing sim1" in output.loc[1, "reason"]
    assert "missing Sim1 station evaluation" in output.loc[2, "reason"]


def test_grid_sims_with_station_ref_write_station_csv_without_flat_grid(tmp_path, monkeypatch):
    basedir = tmp_path / "case"
    _write_station_pair(basedir, "StationRef", "Grid1", ["A"])
    _write_station_pair(basedir, "StationRef", "Grid2", ["A"])
    _write_station_series(basedir, "StationRef", "Grid1", "A", [1.0, 2.0, 3.0])
    _write_station_series(basedir, "StationRef", "Grid2", "A", [1.0, 3.0, 5.0])
    calls = []
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable",
        lambda name: lambda *args, **kwargs: calls.append((name, args, kwargs)),
    )
    handler = _CorrelationHarness()
    sim_nml = {
        "general": {"Runoff_sim_source": ["Grid1", "Grid2"]},
        "Runoff": {
            "Grid1_data_type": "grid",
            "Grid1_varname": "flow",
            "Grid2_data_type": "grid",
            "Grid2_varname": "flow",
        },
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "StationRef"},
        "Runoff": {"StationRef_data_type": "stn", "StationRef_varname": "flow"},
    }

    handler.scenarios_Correlation_comparison(str(basedir), sim_nml, ref_nml, ["Runoff"], [], [], {})

    assert not (basedir / "comparisons/Correlation/Correlation_Runoff_Grid1_and_Grid2.nc").exists()
    output_path = basedir / "comparisons/Correlation/Correlation_Runoff_stn_StationRef_Grid1_and_Grid2.csv"
    output = pd.read_csv(output_path)
    assert output.loc[0, "Correlation"] == pytest.approx(1.0)
    assert output.loc[0, "status"] == "ok"
    assert calls == [
        (
            "make_stn_plot_index",
            (str(output_path), "Correlation", handler.main_nml["general"], ("Grid1 / Grid2",), {}),
            {"value_columns": ("Correlation",)},
        )
    ]


def test_correlation_missing_expected_station_metadata_is_fatal(tmp_path, monkeypatch):
    basedir = tmp_path / "case"
    (basedir / "metrics").mkdir(parents=True)
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable", lambda name: lambda *args, **kwargs: None
    )
    handler = _CorrelationHarness()
    sim_nml = {
        "general": {"Runoff_sim_source": ["Sim1", "Sim2"]},
        "Runoff": {
            "Sim1_data_type": "stn",
            "Sim1_varname": "flow",
            "Sim2_data_type": "stn",
            "Sim2_varname": "flow",
        },
    }
    ref_nml = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": "grid", "Ref_varname": "flow"}}

    with pytest.raises(FileNotFoundError, match="Runoff_stn_Ref_Sim1_evaluations.csv"):
        handler.scenarios_Correlation_comparison(str(basedir), sim_nml, ref_nml, ["Runoff"], [], [], {})


def test_station_correlation_rejects_same_id_with_different_grid_ref_coordinates(tmp_path, monkeypatch):
    basedir = tmp_path / "case"
    _write_station_pair(basedir, "GridRef", "Sim1", ["A"])
    _write_station_pair(basedir, "GridRef", "Sim2", ["A"])
    left = basedir / "metrics/Runoff_stn_GridRef_Sim1_evaluations.csv"
    right = basedir / "metrics/Runoff_stn_GridRef_Sim2_evaluations.csv"
    pd.read_csv(left).assign(ref_lon=100.0, ref_lat=10.0).to_csv(left, index=False)
    pd.read_csv(right).assign(ref_lon=101.0, ref_lat=10.0).to_csv(right, index=False)
    _write_station_series(basedir, "GridRef", "Sim1", "A", [1.0, 2.0, 3.0])
    _write_station_series(basedir, "GridRef", "Sim2", "A", [1.0, 3.0, 5.0])
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable", lambda name: lambda *args, **kwargs: None
    )
    handler = _CorrelationHarness()
    sim_nml = {
        "general": {"Runoff_sim_source": ["Sim1", "Sim2"]},
        "Runoff": {
            "Sim1_data_type": "stn",
            "Sim1_varname": "flow",
            "Sim2_data_type": "stn",
            "Sim2_varname": "flow",
        },
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "GridRef"},
        "Runoff": {"GridRef_data_type": "grid", "GridRef_varname": "flow"},
    }

    handler.scenarios_Correlation_comparison(str(basedir), sim_nml, ref_nml, ["Runoff"], [], [], {})

    output = pd.read_csv(basedir / "comparisons/Correlation/Correlation_Runoff_stn_GridRef_Sim1_and_Sim2.csv")
    assert output.loc[0, "status"] == "unavailable"
    assert output.loc[0, "reason"] == "station coordinates differ; no common spatial support"
    assert np.isnan(output.loc[0, "Correlation"])


def test_station_correlation_without_reference_is_explicit_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable", lambda name: lambda *args, **kwargs: None
    )
    handler = _CorrelationHarness()
    sim_nml = {
        "general": {"Runoff_sim_source": ["Sim1", "Sim2"]},
        "Runoff": {
            "Sim1_data_type": "stn",
            "Sim1_varname": "flow",
            "Sim2_data_type": "grid",
            "Sim2_varname": "flow",
        },
    }
    ref_nml = {"general": {}, "Runoff": {}}

    with pytest.raises(ValueError, match="Station Correlation requires a reference source to identify station support"):
        handler.scenarios_Correlation_comparison(str(tmp_path / "case"), sim_nml, ref_nml, ["Runoff"], [], [], {})
