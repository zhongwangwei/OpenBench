"""Architecture cleanup regression checks."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_comparison_tail_scenarios_live_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    tail_source = (ROOT / "src/openbench/core/_comparison_tail.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_tail import TailComparisonMixin" in comparison_source
    assert "class ComparisonProcessing(" in comparison_source
    for mixin_name in (
        "BasicComparisonMixin",
        "CommonComparisonMixin",
        "DiagramComparisonMixin",
        "DistributionComparisonMixin",
        "DiffPlotComparisonMixin",
        "HeatMapComparisonMixin",
        "ParallelCoordinatesComparisonMixin",
        "PortraitComparisonMixin",
        "RelativeScoreComparisonMixin",
        "SingleModelPerformanceIndexComparisonMixin",
        "TailComparisonMixin",
    ):
        assert mixin_name in comparison_source
    assert "def scenarios_Standard_Deviation_comparison(" in tail_source
    assert "def scenarios_Correlation_comparison(" in tail_source
    assert "def scenarios_Standard_Deviation_comparison(" not in comparison_source
    assert "def scenarios_Correlation_comparison(" not in comparison_source


def test_comparison_common_helpers_live_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    common_source = (ROOT / "src/openbench/core/_comparison_common.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_common import CommonComparisonMixin" in comparison_source
    assert "class ComparisonProcessing(" in comparison_source
    for mixin_name in (
        "BasicComparisonMixin",
        "CommonComparisonMixin",
        "DiagramComparisonMixin",
        "DistributionComparisonMixin",
        "DiffPlotComparisonMixin",
        "HeatMapComparisonMixin",
        "ParallelCoordinatesComparisonMixin",
        "PortraitComparisonMixin",
        "RelativeScoreComparisonMixin",
        "SingleModelPerformanceIndexComparisonMixin",
        "TailComparisonMixin",
    ):
        assert mixin_name in comparison_source
    assert "def _run_parallel_or_serial(" in common_source
    assert "def save_result(" in common_source
    assert "def _run_parallel_or_serial(" not in comparison_source
    assert "def save_result(" not in comparison_source


def test_comparison_taylor_target_diagrams_live_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    diagram_source = (ROOT / "src/openbench/core/_comparison_diagrams.py").read_text(encoding="utf-8")
    taylor_source = (ROOT / "src/openbench/core/_comparison_taylor.py").read_text(encoding="utf-8")
    target_source = (ROOT / "src/openbench/core/_comparison_target.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_diagrams import DiagramComparisonMixin" in comparison_source
    assert "DiagramComparisonMixin" in comparison_source
    assert "class DiagramComparisonMixin(TaylorDiagramComparisonMixin, TargetDiagramComparisonMixin):" in diagram_source
    assert "def scenarios_Taylor_Diagram_comparison(" in taylor_source
    assert "def scenarios_Target_Diagram_comparison(" in target_source
    assert "def scenarios_Taylor_Diagram_comparison(" not in comparison_source
    assert "def scenarios_Target_Diagram_comparison(" not in comparison_source


def test_comparison_distribution_scenarios_live_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    distribution_source = (ROOT / "src/openbench/core/_comparison_distributions.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_distributions import DistributionComparisonMixin" in comparison_source
    assert "DistributionComparisonMixin" in comparison_source
    assert "def scenarios_Kernel_Density_Estimate_comparison(" in distribution_source
    assert "def scenarios_Whisker_Plot_comparison(" in distribution_source
    assert "def scenarios_Ridgeline_Plot_comparison(" in distribution_source
    assert "def scenarios_Kernel_Density_Estimate_comparison(" not in comparison_source
    assert "def scenarios_Whisker_Plot_comparison(" not in comparison_source
    assert "def scenarios_Ridgeline_Plot_comparison(" not in comparison_source


def test_comparison_diff_plot_lives_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    diff_source = (ROOT / "src/openbench/core/_comparison_diff.py").read_text(encoding="utf-8")
    diff_plot_source = (ROOT / "src/openbench/core/_comparison_diff_plot.py").read_text(encoding="utf-8")
    diff_station_source = (ROOT / "src/openbench/core/_comparison_diff_station.py").read_text(encoding="utf-8")
    diff_grid_source = (ROOT / "src/openbench/core/_comparison_diff_grid.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_diff import DiffPlotComparisonMixin" in comparison_source
    assert "DiffPlotComparisonMixin" in comparison_source
    assert "class DiffPlotComparisonMixin(DiffPlotScenarioMixin):" in diff_source
    assert "def scenarios_Diff_Plot_comparison(" in diff_plot_source
    assert "def process_station_diff_plot(" in diff_station_source
    assert "def process_grid_diff_plot(" in diff_grid_source
    assert "diff_station_difference_filename(" in diff_station_source
    assert "diff_grid_difference_filename(" in diff_grid_source
    assert "def scenarios_Diff_Plot_comparison(" not in comparison_source


def test_comparison_heatmap_lives_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    heatmap_source = (ROOT / "src/openbench/core/_comparison_heatmap.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_heatmap import HeatMapComparisonMixin" in comparison_source
    assert "HeatMapComparisonMixin" in comparison_source
    assert "def scenarios_HeatMap_comparison(" in heatmap_source
    assert "make_scenarios_scores_comparison_heat_map" in heatmap_source
    assert "def scenarios_HeatMap_comparison(" not in comparison_source


def test_comparison_basic_lives_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    basic_source = (ROOT / "src/openbench/core/_comparison_basic.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_basic import BasicComparisonMixin" in comparison_source
    assert "BasicComparisonMixin" in comparison_source
    assert "def scenarios_Basic_comparison(" in basic_source
    assert "make_geo_plot_index" in basic_source
    assert "def scenarios_Basic_comparison(" not in comparison_source


def test_comparison_smpi_lives_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    smpi_source = (ROOT / "src/openbench/core/_comparison_smpi.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_smpi import SingleModelPerformanceIndexComparisonMixin" in comparison_source
    assert "SingleModelPerformanceIndexComparisonMixin" in comparison_source
    assert "def scenarios_Single_Model_Performance_Index_comparison(" in smpi_source
    assert "make_scenarios_comparison_Single_Model_Performance_Index" in smpi_source
    assert "def scenarios_Single_Model_Performance_Index_comparison(" not in comparison_source


def test_comparison_relative_score_lives_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    relative_source = (ROOT / "src/openbench/core/_comparison_relative.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_relative import RelativeScoreComparisonMixin" in comparison_source
    assert "RelativeScoreComparisonMixin" in comparison_source
    assert "def scenarios_Relative_Score_comparison(" in relative_source
    assert "relative_station_scores_filename" in relative_source
    assert "make_scenarios_comparison_Relative_Score" in relative_source
    assert "def scenarios_Relative_Score_comparison(" not in comparison_source


def test_comparison_portrait_lives_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    portrait_source = (ROOT / "src/openbench/core/_comparison_portrait.py").read_text(encoding="utf-8")
    portrait_seasonal_source = (ROOT / "src/openbench/core/_comparison_portrait_seasonal.py").read_text(
        encoding="utf-8"
    )
    portrait_calc_source = (ROOT / "src/openbench/core/_comparison_portrait_calculations.py").read_text(
        encoding="utf-8"
    )

    assert "from openbench.core._comparison_portrait import PortraitComparisonMixin" in comparison_source
    assert "PortraitComparisonMixin" in comparison_source
    assert "class PortraitComparisonMixin(PortraitSeasonalComparisonMixin):" in portrait_source
    assert "def scenarios_Portrait_Plot_seasonal_comparison(" in portrait_seasonal_source
    assert "def process_portrait_metric(" in portrait_calc_source
    assert "def process_portrait_score(" in portrait_calc_source
    assert "make_scenarios_comparison_Portrait_Plot_seasonal" in portrait_seasonal_source
    assert "def scenarios_Portrait_Plot_seasonal_comparison(" not in comparison_source


def test_comparison_parallel_coordinates_lives_outside_comparison_god_module():
    comparison_source = (ROOT / "src/openbench/core/comparison.py").read_text(encoding="utf-8")
    parallel_source = (ROOT / "src/openbench/core/_comparison_parallel.py").read_text(encoding="utf-8")

    assert "from openbench.core._comparison_parallel import ParallelCoordinatesComparisonMixin" in comparison_source
    assert "ParallelCoordinatesComparisonMixin" in comparison_source
    assert "def scenarios_Parallel_Coordinates_comparison(" in parallel_source
    assert "make_scenarios_comparison_parallel_coordinates" in parallel_source
    assert "def scenarios_Parallel_Coordinates_comparison(" not in comparison_source
