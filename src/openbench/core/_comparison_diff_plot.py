"""Diff Plot comparison scenario dispatcher."""

from __future__ import annotations

import logging
import os
import sys

from openbench.core._comparison_helpers import _comparison_sim_groups
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
                groups = _comparison_sim_groups(evaluation_item, sim_sources, ref_source, sim_nml, ref_nml)
                for data_type, sources in groups.items():
                    if not sources:
                        continue
                    kwargs = dict(
                        basedir=basedir,
                        dir_path=dir_path,
                        evaluation_item=evaluation_item,
                        ref_source=ref_source,
                        sim_sources=sources,
                        metrics=metrics,
                        scores=scores,
                    )
                    if data_type == "stn":
                        process_station_diff_plot(sim_nml=sim_nml, **kwargs)
                    else:
                        process_grid_diff_plot(**kwargs)
                    _comparison_callable("make_scenarios_comparison_Diff_Plot")(
                        dir_path,
                        metrics,
                        scores,
                        evaluation_item,
                        ref_source,
                        sources,
                        self.general_config,
                        sim_nml,
                        data_type,
                        option,
                    )
                if len(groups) > 1:
                    logging.warning(
                        "%s/%s: station and grid comparisons were produced separately; "
                        "cross-representation differences require a common spatial support",
                        evaluation_item,
                        ref_source,
                    )
