"""Relative Score comparison scenario."""

from __future__ import annotations

import gc
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from openbench.core._comparison_helpers import _finite_distribution_values, _write_csv_atomic
from openbench.util.converttype import Convert_Type
from openbench.util.filenames import relative_grid_score_filename, relative_station_scores_filename
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class RelativeScoreComparisonMixin:
    def scenarios_Relative_Score_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "Relative_Score")
            os.makedirs(dir_path, exist_ok=True)

            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        try:
                            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                            if isinstance(sim_sources, str):
                                sim_sources = [sim_sources]

                            for sim_source in sim_sources:
                                try:
                                    ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                    sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                    if ref_data_type == "stn" or sim_data_type == "stn":
                                        file_pattern = os.path.join(
                                            casedir, "scores", f"{evaluation_item}_stn_{ref_source}_*_evaluations.csv"
                                        )
                                        all_files = glob.glob(file_pattern)

                                        if not all_files:
                                            logging.warning(f"No files found for pattern: {file_pattern}")
                                            continue
                                        if len(all_files) < 2:
                                            logging.warning(f"Files less than 2, passing stn {ref_source}-{sim_source}")
                                            continue

                                        combined_relative_scores = pd.DataFrame()
                                        filex = os.path.join(
                                            casedir,
                                            "scores",
                                            f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                        )
                                        df_sim = pd.read_csv(filex, sep=",", header=0)
                                        df_sim = Convert_Type.convert_Frame(df_sim)

                                        ID = df_sim["ID"]
                                        combined_relative_scores["ID"] = df_sim["ID"]
                                        df_sim.set_index("ID", inplace=True)

                                        # Read all files
                                        for score in scores:
                                            try:
                                                dfs = []
                                                for i, file in enumerate(all_files):
                                                    df = pd.read_csv(file, sep=",", header=0)
                                                    df = Convert_Type.convert_Frame(df)

                                                    df.set_index("ID", inplace=True)
                                                    df = df.reindex(ID)
                                                    dfs.append(df[f"{score}"])
                                                # Empty-list check belongs AFTER the file loop, not
                                                # inside it (the original `if not dfs` ran after the
                                                # first append so it was always False). When the file
                                                # list is empty, skip the concat — otherwise
                                                # pd.concat(axis=1, []) raises.
                                                if not dfs:
                                                    logging.warning(
                                                        f"No valid data found for {evaluation_item}, {ref_source}, {sim_source}, {score}"
                                                    )
                                                    continue
                                                # Combine all dataframes
                                                combined_df = pd.concat(dfs, axis=1)
                                                score_mean = combined_df.mean(axis=1, skipna=True).astype("float32")
                                                score_std = combined_df.std(axis=1, skipna=True).astype("float32")
                                                # Calculate relative scores for each file
                                                with np.errstate(divide="ignore", invalid="ignore"):
                                                    relative_scores = (
                                                        df_sim[f"{score}"].values - score_mean.values
                                                    ) / score_std.values
                                                _finite_distribution_values(
                                                    relative_scores,
                                                    plot="Relative Score",
                                                    item=evaluation_item,
                                                    ref_source=ref_source,
                                                    sim_source=sim_source,
                                                    variable=f"relative_{score}",
                                                )
                                                relative_scores = np.where(
                                                    np.isfinite(relative_scores), relative_scores, np.nan
                                                )

                                                # Add the relative scores as a new column to the combined dataframe
                                                combined_relative_scores[f"relative_{score}_{sim_source}"] = (
                                                    relative_scores
                                                )
                                            finally:
                                                gc.collect()  # Clean up memory after processing each score

                                        # Check if any valid relative scores were calculated.
                                        # The ID column is populated before score calculation,
                                        # so DataFrame.empty is not a valid signal here.
                                        relative_score_columns = [
                                            column
                                            for column in combined_relative_scores.columns
                                            if column.startswith("relative_")
                                        ]
                                        has_valid_relative_scores = bool(relative_score_columns) and bool(
                                            combined_relative_scores[relative_score_columns].notna().any().any()
                                        )
                                        if has_valid_relative_scores:
                                            ilat_lon = []
                                            for file in all_files:
                                                df = pd.read_csv(file, sep=",", header=0)
                                                del_col = ["ID", "sim_lat", "sim_lon", "ref_lon", "ref_lat"]
                                                df.drop(
                                                    columns=[col for col in df.columns if col not in del_col],
                                                    inplace=True,
                                                )
                                                ilat_lon.append(df)
                                            # Combine all dataframes
                                            merged_df = pd.concat(ilat_lon).groupby("ID").first().reset_index()
                                            # Save the combined relative scores to a single file
                                            try:
                                                lon_mapping = merged_df.set_index("ID")["ref_lon"].to_dict()
                                                lat_mapping = merged_df.set_index("ID")["ref_lat"].to_dict()
                                                combined_relative_scores["ref_lon"] = combined_relative_scores[
                                                    "ID"
                                                ].map(lon_mapping)
                                                combined_relative_scores["ref_lat"] = combined_relative_scores[
                                                    "ID"
                                                ].map(lat_mapping)
                                            except (KeyError, ValueError) as e:
                                                logging.debug(f"Using sim coordinates instead of ref coordinates: {e}")
                                                lon_mapping = merged_df.set_index("ID")["sim_lon"].to_dict()
                                                lat_mapping = merged_df.set_index("ID")["sim_lat"].to_dict()
                                                combined_relative_scores["sim_lon"] = combined_relative_scores[
                                                    "ID"
                                                ].map(lon_mapping)
                                                combined_relative_scores["sim_lat"] = combined_relative_scores[
                                                    "ID"
                                                ].map(lat_mapping)
                                            combined_relative_scores = Convert_Type.convert_Frame(
                                                combined_relative_scores
                                            )
                                            output_path = os.path.join(
                                                dir_path,
                                                relative_station_scores_filename(
                                                    evaluation_item, ref_source, sim_source
                                                ),
                                            )
                                            _write_csv_atomic(combined_relative_scores, output_path, index=False)
                                        else:
                                            logging.warning(
                                                f"No valid relative scores found for {evaluation_item}, {ref_source}, {sim_source}"
                                            )
                                            continue

                                        try:
                                            _comparison_callable("make_scenarios_comparison_Relative_Score")(
                                                dir_path,
                                                evaluation_item,
                                                ref_source,
                                                sim_source,
                                                scores,
                                                "stn",
                                                self.main_nml["general"],
                                                option,
                                            )
                                        except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
                                            logging.error(f"Error creating relative score plot: {e}")
                                            raise

                                    else:
                                        for score in scores:
                                            try:
                                                file_pattern = os.path.join(
                                                    casedir,
                                                    "scores",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_*_{score}.nc",
                                                )
                                                all_files = glob.glob(file_pattern)

                                                if not all_files:
                                                    logging.warning(f"No files found for pattern: {file_pattern}")
                                                    continue
                                                if len(all_files) < 2:
                                                    logging.warning(
                                                        f"Files less than 2, passing {score} {ref_source}-{sim_source}"
                                                    )
                                                    continue

                                                # Read all files and combine into a single dataset
                                                datasets = []
                                                for file in all_files:
                                                    with xr.open_dataset(file) as ds_file:
                                                        ds = Convert_Type.convert_nc(ds_file.load())
                                                    datasets.append(ds)

                                                if not datasets:
                                                    logging.warning(
                                                        f"No valid data found for {evaluation_item}, {ref_source}, {sim_source}, {score}"
                                                    )
                                                    continue

                                                combined_ds = xr.concat(datasets, dim="file")

                                                # Calculate mean and standard deviation for each grid point
                                                score_mean = (
                                                    combined_ds[score].mean(dim="file", skipna=True).astype("float32")
                                                )
                                                score_std = (
                                                    combined_ds[score].std(dim="file", skipna=True).astype("float32")
                                                )

                                                file = os.path.join(
                                                    casedir,
                                                    "scores",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                                                )
                                                with xr.open_dataset(file) as ds_file:
                                                    ds = Convert_Type.convert_nc(ds_file.load())
                                                relative_score = xr.where(
                                                    score_std != 0,
                                                    (ds[score] - score_mean) / score_std,
                                                    np.nan,
                                                )
                                                relative_score = relative_score.where(
                                                    np.isfinite(relative_score), np.nan
                                                )
                                                _finite_distribution_values(
                                                    relative_score.values,
                                                    plot="Relative Score",
                                                    item=evaluation_item,
                                                    ref_source=ref_source,
                                                    sim_source=sim_source,
                                                    variable=f"relative_{score}",
                                                )

                                                # Create a new dataset to store the relative score
                                                result_ds = xr.Dataset()
                                                result_ds[f"relative_{score}"] = Convert_Type.convert_nc(relative_score)

                                                output_file = os.path.join(
                                                    dir_path,
                                                    relative_grid_score_filename(
                                                        evaluation_item, ref_source, sim_source, score
                                                    ),
                                                )
                                                _write_netcdf_atomic(result_ds, output_file)
                                            finally:
                                                gc.collect()  # Clean up memory after processing each score

                                        try:
                                            _comparison_callable("make_scenarios_comparison_Relative_Score")(
                                                dir_path,
                                                evaluation_item,
                                                ref_source,
                                                sim_source,
                                                scores,
                                                "grid",
                                                self.main_nml["general"],
                                                option,
                                            )
                                        except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
                                            logging.error(f"Error creating relative score plot: {e}")
                                            raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each simulation source
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method
