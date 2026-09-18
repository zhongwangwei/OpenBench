"""Station comparison summaries honor persisted station availability status."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _processor(tmp_path: Path, scores: list[str] | None = None, metrics: list[str] | None = None):
    import openbench.core.comparison as comparison_module

    return comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
                "num_cores": 1,
            }
        },
        scores or [],
        metrics or [],
    )


def _station_namelists():
    sim = {"general": {"Runoff_sim_source": ["SimA"]}, "Runoff": {"SimA_data_type": "stn", "SimA_varname": "flow"}}
    ref = {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid", "RefA_varname": "flow"}}
    return sim, ref


def _write_station_scores_with_stale_unavailable(case_dir: Path, sim_source: str = "SimA") -> None:
    (case_dir / "scores").mkdir(exist_ok=True)
    (case_dir / f"data/stn_RefA_{sim_source}").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ID": ["S1", "S2"],
            "sim_lat": [1.0, 2.0],
            "sim_lon": [3.0, 4.0],
            "ref_lat": [1.0, 2.0],
            "ref_lon": [3.0, 4.0],
            "Overall_Score": [0.8, 999.0],
        }
    ).to_csv(case_dir / f"scores/Runoff_stn_RefA_{sim_source}_evaluations.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["S1", "S2"],
            "sim_lat": [1.0, 2.0],
            "sim_lon": [3.0, 4.0],
            "ref_lat": [1.0, 2.0],
            "ref_lon": [3.0, 4.0],
            "status": ["ok", "unavailable"],
            "reason": ["", "missing station variable"],
        }
    ).to_csv(case_dir / f"data/stn_RefA_{sim_source}/Runoff_evaluation_status.csv", index=False)


@pytest.mark.parametrize(
    ("method_name", "plot_func"),
    [
        ("scenarios_Kernel_Density_Estimate_comparison", "make_scenarios_comparison_Kernel_Density_Estimate"),
        ("scenarios_Whisker_Plot_comparison", "make_scenarios_comparison_Whisker_Plot"),
        ("scenarios_Ridgeline_Plot_comparison", "make_scenarios_comparison_Ridgeline_Plot"),
    ],
)
def test_distribution_station_series_masks_status_unavailable_rows(tmp_path, monkeypatch, method_name, plot_func):
    import openbench.core.comparison as comparison_module

    _write_station_scores_with_stale_unavailable(tmp_path)
    captured = []
    monkeypatch.setattr(comparison_module, plot_func, lambda *args: captured.append(args))
    sim, ref = _station_namelists()

    getattr(_processor(tmp_path, scores=["Overall_Score"]), method_name)(
        str(tmp_path), sim, ref, ["Runoff"], ["Overall_Score"], [], {}
    )

    assert captured
    np.testing.assert_allclose(captured[0][-2][0], np.array([0.8]))


def test_parallel_coordinates_station_summary_masks_status_unavailable_rows(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module

    _write_station_scores_with_stale_unavailable(tmp_path)
    (tmp_path / "metrics").mkdir(exist_ok=True)
    pd.DataFrame({"ID": ["S1", "S2"], "bias": [2.0, 999.0]}).to_csv(
        tmp_path / "metrics/Runoff_stn_RefA_SimA_evaluations.csv", index=False
    )
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_parallel_coordinates", lambda *args: None)
    sim, ref = _station_namelists()

    _processor(tmp_path, scores=["Overall_Score"], metrics=["bias"]).scenarios_Parallel_Coordinates_comparison(
        str(tmp_path), sim, ref, ["Runoff"], ["Overall_Score"], ["bias"], {}
    )

    result = pd.read_csv(tmp_path / "comparisons/Parallel_Coordinates/Parallel_Coordinates_evaluations.csv", sep="\t")
    assert result.loc[0, "Overall_Score"] == pytest.approx(0.8)
    assert result.loc[0, "bias"] == pytest.approx(2.0)


def test_parallel_coordinates_keeps_two_finite_metrics_after_status_merge(tmp_path, monkeypatch):
    """Two valid stations must not be removed as outliers after unavailable rows are restored."""
    import openbench.core.comparison as comparison_module

    (tmp_path / "scores").mkdir()
    (tmp_path / "metrics").mkdir()
    status_dir = tmp_path / "data/stn_RefA_SimA"
    status_dir.mkdir(parents=True)
    ids = ["S1", "S2", "S3", "S4"]
    pd.DataFrame({"ID": ids, "Overall_Score": [0.8, 0.6, 999.0, 999.0]}).to_csv(
        tmp_path / "scores/Runoff_stn_RefA_SimA_evaluations.csv", index=False
    )
    pd.DataFrame({"ID": ids, "bias": [0.8348533, 1.7359673, 999.0, 999.0]}).to_csv(
        tmp_path / "metrics/Runoff_stn_RefA_SimA_evaluations.csv", index=False
    )
    pd.DataFrame(
        {
            "ID": ids,
            "status": ["ok", "ok", "unavailable", "unavailable"],
            "reason": ["", "", "missing station variable", "missing station variable"],
        }
    ).to_csv(status_dir / "Runoff_evaluation_status.csv", index=False)
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_parallel_coordinates", lambda *args: None)
    sim, ref = _station_namelists()

    _processor(tmp_path, scores=["Overall_Score"], metrics=["bias"]).scenarios_Parallel_Coordinates_comparison(
        str(tmp_path), sim, ref, ["Runoff"], ["Overall_Score"], ["bias"], {}
    )

    result = pd.read_csv(tmp_path / "comparisons/Parallel_Coordinates/Parallel_Coordinates_evaluations.csv", sep="\t")
    assert result.loc[0, "bias"] == pytest.approx(1.29)


def test_heatmap_station_summary_masks_status_unavailable_rows(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module

    _write_station_scores_with_stale_unavailable(tmp_path)
    monkeypatch.setattr(comparison_module, "make_scenarios_scores_comparison_heat_map", lambda *args: None)
    sim, ref = _station_namelists()

    _processor(tmp_path, scores=["Overall_Score"]).scenarios_HeatMap_comparison(
        str(tmp_path), sim, ref, ["Runoff"], ["Overall_Score"], [], {}
    )

    result = pd.read_csv(tmp_path / "comparisons/HeatMap/scenarios_Overall_Score_comparison.csv")
    assert result.loc[0, "SimA"] == pytest.approx(0.8)


def test_relative_score_station_summary_masks_status_unavailable_rows(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module
    from openbench.util.filenames import relative_station_scores_filename

    _write_station_scores_with_stale_unavailable(tmp_path, "SimA")
    (tmp_path / "data/stn_RefA_SimB").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ID": ["S1", "S2"],
            "sim_lat": [1.0, 2.0],
            "sim_lon": [3.0, 4.0],
            "ref_lat": [1.0, 2.0],
            "ref_lon": [3.0, 4.0],
            "Overall_Score": [0.2, 0.4],
        }
    ).to_csv(tmp_path / "scores/Runoff_stn_RefA_SimB_evaluations.csv", index=False)
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Relative_Score", lambda *args: None)
    sim = {
        "general": {"Runoff_sim_source": ["SimA", "SimB"]},
        "Runoff": {
            "SimA_data_type": "stn",
            "SimA_varname": "flow",
            "SimB_data_type": "stn",
            "SimB_varname": "flow",
        },
    }
    ref = {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid", "RefA_varname": "flow"}}

    _processor(tmp_path, scores=["Overall_Score"]).scenarios_Relative_Score_comparison(
        str(tmp_path), sim, ref, ["Runoff"], ["Overall_Score"], [], {}
    )

    result = pd.read_csv(
        tmp_path / "comparisons/Relative_Score" / relative_station_scores_filename("Runoff", "RefA", "SimA")
    )
    assert np.isfinite(result.loc[result.ID == "S1", "relative_Overall_Score_SimA"]).item()
    assert np.isnan(result.loc[result.ID == "S2", "relative_Overall_Score_SimA"]).item()
