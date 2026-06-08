# -*- coding: utf-8 -*-
import logging
import os
import re

from joblib import Parallel

from openbench.core._comparison_basic import BasicComparisonMixin
from openbench.core._comparison_common import CommonComparisonMixin
from openbench.core._comparison_diagrams import DiagramComparisonMixin
from openbench.core._comparison_diff import DiffPlotComparisonMixin
from openbench.core._comparison_distributions import DistributionComparisonMixin
from openbench.core._comparison_heatmap import HeatMapComparisonMixin
from openbench.core._comparison_parallel import ParallelCoordinatesComparisonMixin
from openbench.core._comparison_portrait import PortraitComparisonMixin
from openbench.core._comparison_relative import RelativeScoreComparisonMixin
from openbench.core._comparison_smpi import SingleModelPerformanceIndexComparisonMixin
from openbench.core._comparison_tail import TailComparisonMixin
from openbench.core._visualization_bridge import visualization_callable
from openbench.core.metrics import metrics
from openbench.core.scores import scores
from openbench.core.statistics import statistics_calculate

# Kept as a module-level symbol for tests/downstream code that monkeypatches
# openbench.core.comparison.Parallel.  The split mixins resolve it lazily.
Parallel = Parallel

make_Correlation = visualization_callable("make_Correlation")
make_Functional_Response = visualization_callable("make_Functional_Response")
make_geo_plot_index = visualization_callable("make_geo_plot_index")
make_Mann_Kendall_Trend_Test = visualization_callable("make_Mann_Kendall_Trend_Test")
make_scenarios_comparison_Diff_Plot = visualization_callable("make_scenarios_comparison_Diff_Plot")
make_scenarios_comparison_Kernel_Density_Estimate = visualization_callable(
    "make_scenarios_comparison_Kernel_Density_Estimate"
)
make_scenarios_comparison_parallel_coordinates = visualization_callable(
    "make_scenarios_comparison_parallel_coordinates"
)
make_scenarios_comparison_Portrait_Plot_seasonal = visualization_callable(
    "make_scenarios_comparison_Portrait_Plot_seasonal"
)
make_scenarios_comparison_radar_map = visualization_callable("make_scenarios_comparison_radar_map")
make_scenarios_comparison_Relative_Score = visualization_callable("make_scenarios_comparison_Relative_Score")
make_scenarios_comparison_Ridgeline_Plot = visualization_callable("make_scenarios_comparison_Ridgeline_Plot")
make_scenarios_comparison_Single_Model_Performance_Index = visualization_callable(
    "make_scenarios_comparison_Single_Model_Performance_Index"
)
make_scenarios_comparison_Target_Diagram = visualization_callable("make_scenarios_comparison_Target_Diagram")
make_scenarios_comparison_Taylor_Diagram = visualization_callable("make_scenarios_comparison_Taylor_Diagram")
make_scenarios_comparison_Whisker_Plot = visualization_callable("make_scenarios_comparison_Whisker_Plot")
make_scenarios_scores_comparison_heat_map = visualization_callable("make_scenarios_scores_comparison_heat_map")
make_Standard_Deviation = visualization_callable("make_Standard_Deviation")
make_stn_plot_index = visualization_callable("make_stn_plot_index")


class ComparisonProcessing(
    BasicComparisonMixin,
    CommonComparisonMixin,
    DiagramComparisonMixin,
    DistributionComparisonMixin,
    DiffPlotComparisonMixin,
    HeatMapComparisonMixin,
    ParallelCoordinatesComparisonMixin,
    PortraitComparisonMixin,
    RelativeScoreComparisonMixin,
    SingleModelPerformanceIndexComparisonMixin,
    TailComparisonMixin,
    metrics,
    scores,
    statistics_calculate,
):
    def __init__(self, main_nml, scores, metrics):
        self.name = "ComparisonDataHandler"
        self.version = "0.3"
        self.release = "0.3"
        self.date = "June 2024"
        self.author = "Zhongwang Wei"
        self.main_nml = main_nml
        self.general_config = self.main_nml["general"]
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        self.compare_nml = {}
        # Add default weight attribute
        self.weight = self.main_nml["general"].get("weight", "none")  # Default to 'none' if not specified
        self.time_alignment = self.main_nml["general"].get("time_alignment", "intersection")

        # Frequency mapping for time resolution parsing
        self.freq_map = {
            "year": "Y",
            "yr": "Y",
            "y": "Y",
            "month": "M",
            "mon": "M",
            "m": "M",
            "week": "W",
            "wk": "W",
            "w": "W",
            "day": "D",
            "d": "D",
            "hour": "H",
            "hr": "H",
            "h": "H",
        }

        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml["general"]["compare_grid_res"]
        self.compare_tim_res = self.main_nml["general"].get("compare_tim_res", "1").lower()
        self.casedir = os.path.join(self.main_nml["general"]["basedir"], self.main_nml["general"]["basename"])

        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ["climatology-year", "climatology-month"]:
            logging.info(
                f"ComparisonProcessing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion"
            )
        else:
            # this should be done in read_namelist
            # adjust the time frequency
            match = re.match(r"(\d*)\s*([a-zA-Z]+)", self.compare_tim_res)
            if not match:
                logging.error("Invalid time resolution format. Use '3month', '6hr', etc.")
                raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")

            value, unit = match.groups()
            if not value:
                value = 1
            else:
                value = int(value)  # Convert the numerical value to an integer
            # Get the corresponding pandas frequency
            freq = self.freq_map.get(unit.lower())
            if not freq:
                raise ValueError(f"Unsupported time unit: {unit}")
            self.compare_tim_res = f"{value}{freq}E"

        self.metrics = metrics
        self.scores = scores

    # `to_dict` was removed: it had zero callers across the codebase
    # and its naive `return self.__dict__` would have leaked mutable
    # internal state if it had ever been used externally.

    from openbench.data.coordinates import COORDINATE_MAP_WITH_VERTICAL

    coordinate_map = dict(COORDINATE_MAP_WITH_VERTICAL)

    # NOTE: freq_map is defined as an INSTANCE attribute in __init__
    # (line ~66). The class-body version that previously lived here was
    # incomplete (missing year/yr/y keys etc.) and would only ever be
    # consulted if a subclass bypassed __init__. Removed to eliminate
    # the silent two-source inconsistency.
