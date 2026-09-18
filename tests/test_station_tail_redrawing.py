"""Station statistic figures use the same outputs in compute and drawing-only runs."""

import numpy as np
import pandas as pd
import pytest

from openbench.visualization import only_drawing as drawing


@pytest.mark.parametrize(
    "method,columns,options",
    [
        ("Standard_Deviation", ["ref_value", "sim_value"], {}),
        ("Mann_Kendall_Trend_Test", ["ref_tau", "sim_tau", "ref_trend", "sim_trend"], {}),
        ("Functional_Response", ["functional_response_score"], {"nbins": 2}),
    ],
)
@pytest.mark.parametrize("ref_type,sim_type", [("stn", "grid"), ("grid", "stn")])
def test_station_statistic_redrawing_uses_pair_csv_not_missing_grid(
    tmp_path, monkeypatch, method, columns, options, ref_type, sim_type
):
    output = tmp_path / "comparisons" / method
    output.mkdir(parents=True)
    pd.DataFrame(
        {
            "ID": ["A", "B"],
            "ref_lat": [1.0, 2.0],
            "ref_lon": [3.0, 4.0],
            "status": ["ok", "unavailable"],
            "reason": ["", "missing flow"],
            **{name: [1.0, np.nan] for name in columns},
        }
    ).to_csv(output / f"{method}_Runoff_stn_Ref_Sim.csv", index=False)
    sim = {"general": {"Runoff_sim_source": "Sim"}, "Runoff": {"Sim_data_type": sim_type, "Sim_varname": "flow"}}
    ref = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": ref_type, "Ref_varname": "flow"}}
    calls = []
    monkeypatch.setattr(drawing, "make_stn_plot_index", lambda *args, **kwargs: calls.append((args, kwargs)))
    handler = drawing.ComparisonProcessing_only_drawing.__new__(drawing.ComparisonProcessing_only_drawing)
    handler.compare_nml = {}
    handler.main_nml = {"general": {}}
    handler.time_alignment = "intersection"
    getattr(handler, f"scenarios_{method}_comparison")(str(tmp_path), sim, ref, ["Runoff"], [], [], options)
    assert len(calls) == 1
    assert list(calls[0][1].get("value_columns", ["ref_value", "sim_value"])) == columns


@pytest.mark.parametrize("value", [0.0, 1.0, np.inf])
def test_station_redrawing_rejects_numeric_values_marked_unavailable(tmp_path, value):
    path = tmp_path / "stale.csv"
    pd.DataFrame({"ID": ["A"], "bias": [value], "status": ["unavailable"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unavailable.*non-missing"):
        drawing._require_station_csv_values(str(path), "bias")


def test_station_redrawing_uses_per_value_status(tmp_path):
    path = tmp_path / "partial.csv"
    pd.DataFrame(
        {
            "ID": ["A"],
            "bias": [np.nan],
            "status": ["partial"],
            "status_bias": ["unavailable"],
            "reason_bias": ["undefined"],
        }
    ).to_csv(path, index=False)
    drawing._require_station_csv_values(str(path), "bias")


@pytest.mark.parametrize("ref_type,sim_types", [("stn", ("grid", "grid")), ("grid", ("stn", "grid"))])
def test_station_correlation_redrawing_uses_pair_csv(tmp_path, monkeypatch, ref_type, sim_types):
    directory = tmp_path / "comparisons/Correlation"
    directory.mkdir(parents=True)
    path = directory / "Correlation_Runoff_stn_Ref_A_and_B.csv"
    pd.DataFrame(
        {
            "ID": ["gap"],
            "Correlation": [np.nan],
            "status": ["unavailable"],
            "reason": ["missing flow"],
            "ref_lon": [10.0],
            "ref_lat": [20.0],
        }
    ).to_csv(path, index=False)
    sim = {
        "general": {"Runoff_sim_source": ["A", "B"]},
        "Runoff": {"A_data_type": sim_types[0], "B_data_type": sim_types[1]},
    }
    ref = {"general": {"Runoff_ref_source": "Ref"}, "Runoff": {"Ref_data_type": ref_type}}
    calls = []
    monkeypatch.setattr(drawing, "make_stn_plot_index", lambda *a, **kw: calls.append((a, kw)))
    handler = drawing.ComparisonProcessing_only_drawing.__new__(drawing.ComparisonProcessing_only_drawing)
    handler.main_nml = {"general": {}}
    handler.scenarios_Correlation_comparison(str(tmp_path), sim, ref, ["Runoff"], [], [], {})
    assert len(calls) == 1 and calls[0][0][0] == str(path)
    assert calls[0][1]["value_columns"] == ("Correlation",)
