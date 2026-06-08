"""Plotting and figure generation."""

# Default to a non-interactive backend only for the common Linux/headless CLI
# case, and never override a backend chosen by the caller.  `matplotlib.use(...,
# force=False)` still changes a caller-selected backend when pyplot has not yet
# been imported, so guard explicitly instead of relying on force=False.
import os as _os
import sys as _sys

import matplotlib as _matplotlib


def _backend_name(value) -> str:
    return str(value).lower()


def _caller_selected_backend() -> bool:
    return bool(_os.environ.get("MPLBACKEND")) or _backend_name(_matplotlib.get_backend()) != _backend_name(
        _matplotlib.rcParamsOrig.get("backend")
    )


def _should_default_to_agg() -> bool:
    if "matplotlib.pyplot" in _sys.modules or _caller_selected_backend():
        return False
    if not _sys.platform.startswith("linux"):
        return False
    return not (_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))


if _should_default_to_agg():
    _matplotlib.use("Agg", force=False)

from .Fig_ANOVA import make_ANOVA  # noqa: E402
from .Fig_Basic_Plot import make_Basic, make_plot_index_grid, make_plot_index_stn, plot_stn
from .Fig_Correlation import make_Correlation
from .Fig_Diff_Plot import make_scenarios_comparison_Diff_Plot
from .Fig_Functional_Response import make_Functional_Response
from .Fig_geo_plot_index import make_geo_plot_index
from .Fig_heatmap import make_scenarios_scores_comparison_heat_map
from .Fig_Hellinger_Distance import make_Hellinger_Distance
from .Fig_kernel_density_estimate import make_scenarios_comparison_Kernel_Density_Estimate
from .Fig_LC_based_heat_map import make_CZ_based_heat_map, make_LC_based_heat_map
from .Fig_Mann_Kendall_Trend_Test import make_Mann_Kendall_Trend_Test
from .Fig_parallel_coordinates import make_scenarios_comparison_parallel_coordinates
from .Fig_Partial_Least_Squares_Regression import make_Partial_Least_Squares_Regression
from .Fig_portrait_plot_seasonal import make_scenarios_comparison_Portrait_Plot_seasonal
from .Fig_radarmap import make_scenarios_comparison_radar_map
from .Fig_Relative_Score import make_scenarios_comparison_Relative_Score
from .Fig_Ridgeline_Plot import make_scenarios_comparison_Ridgeline_Plot
from .Fig_Single_Model_Performance_Index import make_scenarios_comparison_Single_Model_Performance_Index
from .Fig_Standard_Deviation import make_Standard_Deviation
from .Fig_stn_plot_index import make_stn_plot_index
from .Fig_target_diagram import make_scenarios_comparison_Target_Diagram
from .Fig_taylor_diagram import make_scenarios_comparison_Taylor_Diagram
from .Fig_Three_Cornered_Hat import make_Three_Cornered_Hat
from .Fig_Whisker_Plot import make_scenarios_comparison_Whisker_Plot

__all__ = [
    "make_scenarios_comparison_Portrait_Plot_seasonal",
    "make_scenarios_scores_comparison_heat_map",
    "make_scenarios_comparison_parallel_coordinates",
    "make_scenarios_comparison_Taylor_Diagram",
    "make_scenarios_comparison_Target_Diagram",
    "make_scenarios_comparison_Kernel_Density_Estimate",
    "make_geo_plot_index",
    "make_stn_plot_index",
    "make_LC_based_heat_map",
    "make_CZ_based_heat_map",
    "make_scenarios_comparison_Whisker_Plot",
    "make_scenarios_comparison_Single_Model_Performance_Index",
    "make_scenarios_comparison_Relative_Score",
    "make_scenarios_comparison_Ridgeline_Plot",
    "make_scenarios_comparison_Diff_Plot",
    "make_scenarios_comparison_radar_map",
    "make_Mann_Kendall_Trend_Test",
    "make_Correlation",
    "make_Standard_Deviation",
    "make_Hellinger_Distance",
    "make_Functional_Response",
    "make_Partial_Least_Squares_Regression",
    "make_plot_index_stn",
    "make_plot_index_grid",
    "plot_stn",
    "make_Basic",
    "make_ANOVA",
    "make_Three_Cornered_Hat",
]
