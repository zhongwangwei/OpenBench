"""Distribution-style comparison scenarios."""

from __future__ import annotations

import gc
import logging
import os
import sys

import pandas as pd
import xarray as xr

from openbench.core._comparison_helpers import _finite_distribution_values
from openbench.util.converttype import Convert_Type


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class DistributionComparisonMixin:
    def scenarios_Kernel_Density_Estimate_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Kernel_Density_Estimate")
            os.makedirs(dir_path, exist_ok=True)

            # fixme: add the Kernel Density Estimate
            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for score in scores:
                        try:
                            # Skip nSpatialScore since it's a constant value
                            if score == "nSpatialScore":
                                logging.info(f"Skipping {score} for Kernel Density Estimate - it's a constant value")
                                continue
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                # read the file_path data and select the score
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                df = Convert_Type.convert_Frame(df)
                                                data = df[score].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                                                )
                                                with xr.open_dataset(file_path) as ds_file:
                                                    ds = Convert_Type.convert_nc(ds_file.load())
                                                data = ds[score].values
                                            datasets_filtered.append(
                                                _finite_distribution_values(
                                                    data,
                                                    plot="Kernel Density Estimate",
                                                    item=evaluation_item,
                                                    ref_source=ref_source,
                                                    sim_source=sim_source,
                                                    variable=score,
                                                )
                                            )
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        _comparison_callable("make_scenarios_comparison_Kernel_Density_Estimate")(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            score,
                                            datasets_filtered,
                                            option,
                                        )
                                    except (ValueError, RuntimeError, IOError, OSError) as e:
                                        logging.error(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Kernel Density Estimate failed: {e}"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each score

                    for metric in metrics:
                        try:
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                # read the file_path data and select the metric
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                data = df[metric].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc",
                                                )
                                                with xr.open_dataset(file_path) as ds_file:
                                                    ds = Convert_Type.convert_nc(ds_file.load())
                                                data = ds[metric].values

                                            if metric == "percent_bias":
                                                data = data[(data >= -100) & (data <= 100)]
                                            datasets_filtered.append(
                                                _finite_distribution_values(
                                                    data,
                                                    plot="Kernel Density Estimate",
                                                    item=evaluation_item,
                                                    ref_source=ref_source,
                                                    sim_source=sim_source,
                                                    variable=metric,
                                                )
                                            )
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        _comparison_callable("make_scenarios_comparison_Kernel_Density_Estimate")(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            metric,
                                            datasets_filtered,
                                            option,
                                        )
                                    except (ValueError, RuntimeError, IOError, OSError) as e:
                                        logging.error(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Kernel Density Estimate failed: {e}"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each metric
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Whisker_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Whisker_Plot")
            os.makedirs(dir_path, exist_ok=True)

            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                    # If the sim_sources and ref_sources are not lists, convert them to lists
                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for score in scores:
                        try:
                            # Skip nSpatialScore since it's a constant value
                            if score == "nSpatialScore":
                                logging.info(f"Skipping {score} for Whisker Plot - it's a constant value")
                                continue
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # Create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                # Read the file_path data and select the score
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                df = Convert_Type.convert_Frame(df)
                                                data = df[score].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                                                )
                                                with xr.open_dataset(file_path) as ds_file:
                                                    ds = Convert_Type.convert_nc(ds_file.load())
                                                data = ds[score].values
                                            datasets_filtered.append(
                                                _finite_distribution_values(
                                                    data,
                                                    plot="Whisker Plot",
                                                    item=evaluation_item,
                                                    ref_source=ref_source,
                                                    sim_source=sim_source,
                                                    variable=score,
                                                )
                                            )
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        _comparison_callable("make_scenarios_comparison_Whisker_Plot")(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            score,
                                            datasets_filtered,
                                            option,
                                        )
                                    except (ValueError, RuntimeError, IOError, OSError) as e:
                                        logging.error(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Whisker Plot failed: {e}"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each score

                    for metric in metrics:
                        try:
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # Create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                # Read the file_path data and select the metric
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                data = df[metric].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc",
                                                )
                                                with xr.open_dataset(file_path) as ds_file:
                                                    ds = Convert_Type.convert_nc(ds_file.load())
                                                data = ds[metric].values

                                            if metric == "percent_bias":
                                                data = data[(data >= -100) & (data <= 100)]
                                            datasets_filtered.append(
                                                _finite_distribution_values(
                                                    data,
                                                    plot="Whisker Plot",
                                                    item=evaluation_item,
                                                    ref_source=ref_source,
                                                    sim_source=sim_source,
                                                    variable=metric,
                                                )
                                            )
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        _comparison_callable("make_scenarios_comparison_Whisker_Plot")(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            metric,
                                            datasets_filtered,
                                            option,
                                        )
                                    except (ValueError, RuntimeError, IOError, OSError) as e:
                                        logging.error(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Whisker Plot failed: {e}"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each metric
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Ridgeline_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f"{basedir}", "comparisons", "Ridgeline_Plot")
        # if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        # print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
            # if the sim_sources and ref_sources are not list, then convert them to list
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]
            for score in scores:
                # Skip nSpatialScore since it's a constant value
                if score == "nSpatialScore":
                    logging.info(f"Skipping {score} for Ridgeline Plot - it's a constant value")
                    continue
                for ref_source in ref_sources:
                    datasets_filtered = []
                    # create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        if isinstance(ref_sources, str):
                            ref_sources = [ref_sources]
                        # create a numpy matrix to store the data

                        if ref_data_type == "stn" or sim_data_type == "stn":
                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            if sim_varname is None or sim_varname == "":
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == "":
                                ref_varname = evaluation_item
                            file_path = os.path.join(
                                basedir, "scores", f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            )
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=",", header=0)
                            df = Convert_Type.convert_Frame(df)
                            data = df[score].values
                        else:
                            file_path = os.path.join(
                                basedir, "scores", f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                            )
                            with xr.open_dataset(file_path) as ds_file:
                                ds = Convert_Type.convert_nc(ds_file.load())
                            data = ds[score].values
                        datasets_filtered.append(
                            _finite_distribution_values(
                                data,
                                plot="Ridgeline Plot",
                                item=evaluation_item,
                                ref_source=ref_source,
                                sim_source=sim_source,
                                variable=score,
                            )
                        )

                    try:
                        _comparison_callable("make_scenarios_comparison_Ridgeline_Plot")(
                            dir_path, evaluation_item, ref_source, sim_sources, score, datasets_filtered, option
                        )
                    except (ValueError, RuntimeError, IOError, OSError) as e:
                        logging.error(
                            f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Ridgeline_Plot failed: {e}"
                        )
                        raise

            for metric in metrics:
                for ref_source in ref_sources:
                    dir_path = os.path.join(f"{basedir}", "comparisons", "Ridgeline_Plot")
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    datasets_filtered = []
                    # create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        if isinstance(ref_sources, str):
                            ref_sources = [ref_sources]
                        # create a numpy matrix to store the data
                        if ref_data_type == "stn" or sim_data_type == "stn":
                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            if sim_varname is None or sim_varname == "":
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == "":
                                ref_varname = evaluation_item
                            file_path = os.path.join(
                                basedir, "metrics", f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            )
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=",", header=0)
                            data = df[metric].values
                        else:
                            file_path = os.path.join(
                                basedir, "metrics", f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc"
                            )

                            with xr.open_dataset(file_path) as ds_file:
                                ds = Convert_Type.convert_nc(ds_file.load())
                            data = ds[metric].values
                        if metric == "percent_bias":
                            data = data[(data >= -100) & (data <= 100)]
                        datasets_filtered.append(
                            _finite_distribution_values(
                                data,
                                plot="Ridgeline Plot",
                                item=evaluation_item,
                                ref_source=ref_source,
                                sim_source=sim_source,
                                variable=metric,
                            )
                        )

                    try:
                        _comparison_callable("make_scenarios_comparison_Ridgeline_Plot")(
                            dir_path, evaluation_item, ref_source, sim_sources, metric, datasets_filtered, option
                        )
                    except (ValueError, RuntimeError, IOError, OSError) as e:
                        logging.error(
                            f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Ridgeline Plot failed: {e}"
                        )
                        raise
