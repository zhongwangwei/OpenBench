"""Diff Plot comparison scenario dispatcher."""

from __future__ import annotations

import logging
import os
import sys

from openbench.core._comparison_diff_grid import process_grid_diff_plot
from openbench.core._comparison_diff_station import process_station_diff_plot


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class DiffPlotScenarioMixin:
    def scenarios_Diff_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """Compare metrics and scores between simulations, then render Diff Plot outputs."""
        dir_path = os.path.join(f"{basedir}", "comparisons", "Diff_Plot")
        os.makedirs(dir_path, exist_ok=True)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                data_types = [sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"] for sim_source in sim_sources]
                if "stn" in data_types and any(data_type != "stn" for data_type in data_types):
                    logging.warning(f"Cannot compare station and gridded data together for {evaluation_item}")
                    logging.warning(
                        "All simulation sources must be either station data or gridded data; skipping this evaluation item"
                    )
                    continue

                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                if ref_data_type == "stn":
                    process_station_diff_plot(
                        basedir=basedir,
                        dir_path=dir_path,
                        sim_nml=sim_nml,
                        evaluation_item=evaluation_item,
                        ref_source=ref_source,
                        sim_sources=sim_sources,
                        metrics=metrics,
                        scores=scores,
                    )
                else:
                    process_grid_diff_plot(
                        basedir=basedir,
                        dir_path=dir_path,
                        evaluation_item=evaluation_item,
                        ref_source=ref_source,
                        sim_sources=sim_sources,
                        metrics=metrics,
                        scores=scores,
                    )

                try:
                    _comparison_callable("make_scenarios_comparison_Diff_Plot")(
                        dir_path,
                        metrics,
                        scores,
                        evaluation_item,
                        ref_source,
                        sim_sources,
                        self.general_config,
                        sim_nml,
                        ref_data_type,
                        option,
                    )
                except (ValueError, RuntimeError, IOError, OSError) as e:
                    logging.error(f"Error creating Diff Plot for {evaluation_item}/{ref_source}: {e}")
                    raise
