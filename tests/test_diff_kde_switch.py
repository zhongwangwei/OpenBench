from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from openbench.cli.init_cmd import COMPARISON_OPTIONS
from openbench.config.adapter import build_fig_nml
from openbench.core import comparison
from openbench.core._comparison_diff_plot import render_grid_diff_kdes
from openbench.runner.postprocessing import _comparison_execution_order
from openbench.util.filenames import diff_grid_difference_filename
from openbench.visualization.only_drawing import ComparisonProcessing_only_drawing

SWITCH = "Diff_Kernel_Density_Estimate"


def test_diff_kde_switch_is_registered():
    assert SWITCH in COMPARISON_OPTIONS
    assert SWITCH in build_fig_nml()["Comparison"]
    assert hasattr(comparison.ComparisonProcessing, f"scenarios_{SWITCH}_comparison")
    assert hasattr(ComparisonProcessing_only_drawing, f"scenarios_{SWITCH}_comparison")


def test_diff_kde_switch_runs_diff_plot_first_without_duplicates():
    assert _comparison_execution_order([SWITCH]) == ["Diff_Plot", SWITCH]
    assert _comparison_execution_order([SWITCH, "Diff_Plot", "HeatMap"]) == ["Diff_Plot", SWITCH, "HeatMap"]
    assert _comparison_execution_order(["Diff_Plot", SWITCH]) == ["Diff_Plot", SWITCH]


def test_render_grid_diff_kdes_uses_expected_pairwise_file(tmp_path: Path, monkeypatch):
    diff_dir = tmp_path / "comparisons" / "Diff_Plot"
    diff_dir.mkdir(parents=True)
    filename = diff_grid_difference_filename("Runoff", "RefA", "SimA", "SimB", "Overall_Score")
    xr.Dataset(
        {"Overall_Score_diff": (("lat", "lon"), [[-0.2, np.nan], [0.1, 0.2]])}
    ).to_netcdf(diff_dir / filename)

    calls = []
    monkeypatch.setattr(
        comparison,
        "make_scenarios_comparison_Kernel_Density_Estimate",
        lambda *args: calls.append(args),
    )
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA", "SimB"]},
        "Runoff": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "RefA"},
        "Runoff": {"RefA_data_type": "grid"},
    }

    render_grid_diff_kdes(
        str(tmp_path),
        sim_nml,
        ref_nml,
        ["Runoff"],
        ["Overall_Score", "nSpatialScore"],
        [],
        {"title": "", "xticklabel": ""},
    )

    output_dir, item, ref, labels, variable, datasets, option = calls[0]
    assert Path(output_dir) == tmp_path / "comparisons" / "Kernel_Density_Estimate"
    assert (item, ref, labels, variable) == ("Runoff", "RefA", ["SimA − SimB"], "Overall_Score_diff")
    np.testing.assert_array_equal(datasets[0], [-0.2, 0.1, 0.2])
    assert option["title"] == "Kernel Density Estimate of Runoff Difference"
    assert option["xticklabel"] == "Overall Score Difference"
    assert len(calls) == 1
