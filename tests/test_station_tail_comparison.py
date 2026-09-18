"""Station regressions for tail comparison methods."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core._comparison_tail import TailComparisonMixin
from openbench.core.statistics.stat_functional_response import stat_functional_response
from openbench.core.statistics.stat_mann_kendall_trend_test import stat_mann_kendall_trend_test
from openbench.core.statistics.stat_standard_deviation import stat_standard_deviation


class _TailHarness(TailComparisonMixin):
    stat_standard_deviation = stat_standard_deviation
    stat_mann_kendall_trend_test = stat_mann_kendall_trend_test
    stat_functional_response = stat_functional_response

    def __init__(self):
        self.compare_nml = {}
        self.main_nml = {"general": {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90}}
        self.compare_tim_res = "day"
        self.time_alignment = "intersection"

    def save_result(self, output_file, method_name, result):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(output_file)


def _write_station_case(tmp_path):
    basedir = tmp_path / "case"
    for folder in ("metrics", "scores", "data/stn_Ref_Sim", "data"):
        (basedir / folder).mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        {
            "ID": ["ok", "gap"],
            "sim_lat": [10.0, 20.0],
            "sim_lon": [100.0, 110.0],
            "ref_lat": [10.0, 20.0],
            "ref_lon": [100.0, 110.0],
            "use_syear": [2000, 2000],
            "use_eyear": [2000, 2000],
        }
    )
    rows.to_csv(basedir / "metrics" / "Runoff_stn_Ref_Sim_evaluations.csv", index=False)
    pd.DataFrame({"ID": ["ok", "gap"], "status": ["ok", "unavailable"], "reason": ["", "missing source"]}).to_csv(
        basedir / "data/stn_Ref_Sim/Runoff_evaluation_status.csv", index=False
    )
    times_ref = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"])
    # Same days, different hours: Functional Response must align after normalized fallback.
    times_sim = times_ref + pd.Timedelta(hours=12)
    xr.Dataset({"refvar": ("time", [1.0, 2.0, 3.0, 4.0])}, coords={"time": times_ref}).to_netcdf(
        basedir / "data/stn_Ref_Sim/Runoff_ref_ok_2000_2000.nc"
    )
    xr.Dataset({"simvar": ("time", [1.0, 3.0, 5.0, 7.0])}, coords={"time": times_sim}).to_netcdf(
        basedir / "data/stn_Ref_Sim/Runoff_sim_ok_2000_2000.nc"
    )
    return basedir


@pytest.mark.parametrize(
    "method,options,value_column",
    [
        ("Standard_Deviation", {}, "ref_value"),
        ("Mann_Kendall_Trend_Test", {"significance_level": 0.05}, "ref_trend"),
        ("Functional_Response", {"nbins": 2}, "functional_response_score"),
    ],
)
@pytest.mark.parametrize("station_side", ["ref", "sim"])
def test_tail_comparison_station_pair_csv_preserves_missing_rows(
    tmp_path, monkeypatch, method, options, value_column, station_side
):
    basedir = _write_station_case(tmp_path)
    handler = _TailHarness()
    sim_type, ref_type = ("stn", "grid") if station_side == "sim" else ("grid", "stn")
    sim_nml = {"general": {"Runoff_sim_source": "Sim"}, "Runoff": {"Sim_data_type": sim_type, "Sim_varname": "simvar"}}
    ref_nml = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": ref_type, "Ref_varname": "refvar"}}
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable", lambda name: lambda *args, **kwargs: None
    )

    getattr(handler, f"scenarios_{method}_comparison")(str(basedir), sim_nml, ref_nml, ["Runoff"], [], [], options)

    output = pd.read_csv(basedir / "comparisons" / method / f"{method}_Runoff_stn_Ref_Sim.csv")
    assert output["ID"].tolist() == ["ok", "gap"]
    assert output["status"].tolist() == ["ok", "unavailable"]
    assert output.loc[1, "reason"] == "missing source"
    assert np.isfinite(output.loc[0, value_column])
    assert np.isnan(output.loc[1, value_column])


def test_functional_response_aligns_normalized_times_only_after_exact_miss():
    handler = SimpleNamespace(compare_tim_res="Day", compare_nml={"Functional_Response": {"nbins": 2}})
    ref = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": pd.date_range("2000-01-01", periods=4)})
    sim = xr.DataArray(
        [1.0, 3.0, 5.0, 7.0],
        dims="time",
        coords={"time": pd.date_range("2000-01-01 12:00", periods=4)},
    )
    result = stat_functional_response(handler, ref, sim)
    assert np.isfinite(float(result["functional_response_score"]))


def test_radar_station_mean_uses_rehydrated_scores(tmp_path, monkeypatch):
    basedir = _write_station_case(tmp_path)
    pd.DataFrame({"ID": ["ok", "gap"], "Overall_Score": [0.8, 0.1]}).to_csv(
        basedir / "scores/Runoff_stn_Ref_Sim_evaluations.csv", index=False
    )
    handler = _TailHarness()
    sim_nml = {"general": {"Runoff_sim_source": "Sim"}, "Runoff": {"Sim_data_type": "stn", "Sim_varname": "simvar"}}
    ref_nml = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": "stn", "Ref_varname": "refvar"}}
    calls = []
    monkeypatch.setattr(
        "openbench.core._comparison_tail._comparison_callable",
        lambda name: lambda *args, **kwargs: calls.append((name, args)),
    )

    handler.scenarios_RadarMap_comparison(str(basedir), sim_nml, ref_nml, ["Runoff"], ["Overall_Score"], [], {})

    output = pd.read_csv(basedir / "comparisons/RadarMap/scenarios_Overall_Score_comparison.csv")
    assert output.loc[0, "Sim"] == pytest.approx(0.8)


def test_functional_response_hourly_alignment_does_not_collapse_to_daily():
    handler = SimpleNamespace(compare_tim_res="Hour", compare_nml={"Functional_Response": {"nbins": 2}})
    ref = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": pd.date_range("2000-01-01", periods=4, freq="h")}
    )
    sim = ref.assign_coords(time=ref.time + np.timedelta64(30, "m"))
    result = stat_functional_response(handler, ref, sim)
    assert np.isfinite(result.functional_response_score.item())


def test_mann_kendall_constant_station_keeps_undefined_tau_for_redraw(tmp_path, monkeypatch):
    from openbench.visualization.only_drawing import _require_station_csv_values

    basedir = _write_station_case(tmp_path)
    for role, variable in (("ref", "refvar"), ("sim", "simvar")):
        path = basedir / f"data/stn_Ref_Sim/Runoff_{role}_ok_2000_2000.nc"
        xr.Dataset({variable: ("time", [1.0] * 4)}, coords={"time": pd.date_range("2000-01-01", periods=4)}).to_netcdf(
            path
        )
    monkeypatch.setattr("openbench.core._comparison_tail._comparison_callable", lambda name: lambda *a, **kw: None)
    _TailHarness().scenarios_Mann_Kendall_Trend_Test_comparison(
        str(basedir),
        {"general": {"Runoff_sim_source": "Sim"}, "Runoff": {"Sim_data_type": "grid", "Sim_varname": "simvar"}},
        {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": "stn", "Ref_varname": "refvar"}},
        ["Runoff"],
        [],
        [],
        {"significance_level": 0.05},
    )
    path = basedir / "comparisons/Mann_Kendall_Trend_Test/Mann_Kendall_Trend_Test_Runoff_stn_Ref_Sim.csv"
    result = pd.read_csv(path)
    assert result.loc[0, "ref_trend"] == 0.0 and np.isnan(result.loc[0, "ref_tau"])
    for column in ("ref_tau", "sim_tau", "ref_trend", "sim_trend"):
        _require_station_csv_values(str(path), column)
    assert result.loc[0, "status_ref_tau"] == "unavailable"
    assert result.loc[0, "status"] == "partial"
