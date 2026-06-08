"""HeatMap comparison scenario."""

from __future__ import annotations

import gc
import os
import sys

import numpy as np

from openbench.core._comparison_helpers import _atomic_text_writer, _grid_score_mean, _station_csv_column_mean


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class HeatMapComparisonMixin:
    def scenarios_HeatMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "HeatMap")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for score in scores:
                output_file_path = os.path.join(dir_path, f"scenarios_{score}_comparison.csv")
                with _atomic_text_writer(output_file_path) as output_file:
                    # Collect all unique sim_sources across all evaluation items
                    all_sim_sources = []
                    for evaluation_item in evaluation_items:
                        sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        for s in sim_sources:
                            if s not in all_sim_sources:
                                all_sim_sources.append(s)
                    # Write header without trailing tab
                    header = ["Item", "Reference"] + all_sim_sources
                    output_file.write("\t".join(header) + "\n")

                    for evaluation_item in evaluation_items:
                        sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                        ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                        # if the sim_sources and ref_sources are not list, then convert them to list
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        if isinstance(ref_sources, str):
                            ref_sources = [ref_sources]

                        for ref_source in ref_sources:
                            output_file.write(f"{evaluation_item}\t")
                            output_file.write(f"{ref_source}\t")
                            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                            if isinstance(sim_sources, str):
                                sim_sources = [sim_sources]

                            values = []
                            for sim_source in sim_sources:
                                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                                if ref_data_type == "stn" or sim_data_type == "stn":
                                    file = f"{casedir}/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                                    overall_mean = _station_csv_column_mean(file, score, label="Score")
                                else:
                                    overall_mean = _grid_score_mean(
                                        self, casedir, evaluation_item, ref_source, sim_source, ref_varname, score
                                    )

                                overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"
                                values.append(overall_mean_str)
                            # Write values without trailing tab
                            output_file.write("\t".join(values) + "\n")

                _comparison_callable("make_scenarios_scores_comparison_heat_map")(output_file_path, score, option)
        finally:
            gc.collect()  # Clean up memory after processing
