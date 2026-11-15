"""Diff Plot comparison scenario dispatcher."""

from __future__ import annotations

import logging
import os
import sys

import xarray as xr

from openbench.core._comparison_helpers import _comparison_sim_groups
from openbench.core._comparison_diff_grid import process_grid_diff_plot
from openbench.core._comparison_diff_station import process_station_diff_plot
from openbench.util.filenames import diff_grid_difference_filename
from openbench.visualization._validation import finite_values


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


def render_grid_diff_kdes(basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
    """Render KDE curves from the pairwise gridded Diff_Plot NetCDF outputs."""
    diff_dir = os.path.join(basedir, "comparisons", "Diff_Plot")
    output_dir = os.path.join(basedir, "comparisons", "Kernel_Density_Estimate")
    item_types = [*metrics, *(score for score in scores if score != "nSpatialScore")]

    for evaluation_item in evaluation_items:
        sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
        ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
        if isinstance(sim_sources, str):
            sim_sources = [sim_sources]
        if isinstance(ref_sources, str):
            ref_sources = [ref_sources]

        for ref_source in ref_sources:
            groups = _comparison_sim_groups(evaluation_item, sim_sources, ref_source, sim_nml, ref_nml)
            grid_sources = groups.get("grid", [])
            if groups.get("stn"):
                logging.info(
                    "Skipping Diff Kernel Density Estimate for station group %s/%s; only gridded diff NetCDF is supported",
                    evaluation_item,
                    ref_source,
                )
            for index, sim1 in enumerate(grid_sources):
                for sim2 in grid_sources[index + 1 :]:
                    curve_label = f"{sim1} − {sim2}"
                    for item_type in item_types:
                        variable = f"{item_type}_diff"
                        path = os.path.join(
                            diff_dir,
                            diff_grid_difference_filename(evaluation_item, ref_source, sim1, sim2, item_type),
                        )
                        with xr.open_dataset(path) as dataset:
                            if variable not in dataset:
                                raise KeyError(f"{variable!r} not found in Diff_Plot file: {path}")
                            values = finite_values(
                                dataset[variable].values,
                                label=f"{evaluation_item}/{ref_source}/{curve_label}/{variable} KDE",
                            )

                        plot_option = option.copy()
                        plot_option["title"] = (
                            f"Kernel Density Estimate of {evaluation_item.replace('_', ' ')} Difference"
                        )
                        plot_option["xticklabel"] = f"{item_type.replace('_', ' ')} Difference"
                        _comparison_callable("make_scenarios_comparison_Kernel_Density_Estimate")(
                            output_dir,
                            evaluation_item,
                            ref_source,
                            [curve_label],
                            variable,
                            [values],
                            plot_option,
                        )


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

    def scenarios_Diff_Kernel_Density_Estimate_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        """Render KDE figures from pairwise gridded Diff_Plot outputs."""
        render_grid_diff_kdes(basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option)
