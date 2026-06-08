"""Parallel Coordinates comparison scenario."""

from __future__ import annotations

import gc
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from openbench.core._comparison_helpers import _finite_reduced_value, _write_csv_atomic
from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_file_atomic as _write_file_atomic


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class ParallelCoordinatesComparisonMixin:
    def scenarios_Parallel_Coordinates_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Parallel_Coordinates")
            os.makedirs(dir_path, exist_ok=True)

            output_file_path = os.path.join(dir_path, "Parallel_Coordinates_evaluations.csv")

            def _write_evaluations_csv(tmp_output_file_path):
                with open(tmp_output_file_path, "w") as output_file:
                    output_file.write("Item\t")
                    output_file.write("Reference\t")
                    output_file.write("Simulation\t")
                    for score in scores:
                        output_file.write(f"{score}\t")
                    for metric in metrics:
                        output_file.write(f"{metric}\t")
                    output_file.write("\n")

                    # read the simulation source and reference source
                    for evaluation_item in evaluation_items:
                        try:
                            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                            # if the sim_sources and ref_sources are not list, then convert them to list
                            if isinstance(sim_sources, str):
                                sim_sources = [sim_sources]
                            if isinstance(ref_sources, str):
                                ref_sources = [ref_sources]

                            for ref_source in ref_sources:
                                try:
                                    for i, sim_source in enumerate(sim_sources):
                                        try:
                                            output_file.write(f"{evaluation_item}\t")
                                            output_file.write(f"{ref_source}\t")
                                            output_file.write(f"{sim_source}\t")
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
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                df = Convert_Type.convert_Frame(df)

                                                for score in scores:
                                                    kk = _finite_reduced_value(
                                                        df[score].values,
                                                        reducer="mean",
                                                        plot="Parallel Coordinates",
                                                        item=evaluation_item,
                                                        ref_source=ref_source,
                                                        sim_source=sim_source,
                                                        variable=score,
                                                    )
                                                    kk_str = f"{kk:.2f}"
                                                    output_file.write(f"{kk_str}\t")

                                                for metric in metrics:
                                                    df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)
                                                    if df[metric].shape[0] > 2:
                                                        q_low, q_high = df[metric].quantile([0.05, 0.95])
                                                        df[metric] = df[metric].where(
                                                            (df[metric] >= q_low) & (df[metric] <= q_high), np.nan
                                                        )

                                                    kk = _finite_reduced_value(
                                                        df[metric].values,
                                                        reducer="median",
                                                        plot="Parallel Coordinates",
                                                        item=evaluation_item,
                                                        ref_source=ref_source,
                                                        sim_source=sim_source,
                                                        variable=metric,
                                                    )
                                                    kk_str = f"{kk:.2f}"
                                                    output_file.write(f"{kk_str}\t")

                                                output_file.write("\n")
                                            else:
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                ref_path = self._ref_data_path(
                                                    basedir, evaluation_item, ref_source, ref_varname, sim_source
                                                )
                                                sim_path = os.path.join(
                                                    basedir,
                                                    "data",
                                                    f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc",
                                                )

                                                with xr.open_dataset(ref_path) as ref_ds:
                                                    reffile = ref_ds[ref_varname].load()
                                                with xr.open_dataset(sim_path) as sim_ds:
                                                    simfile = sim_ds[sim_varname].load()
                                                reffile = Convert_Type.convert_nc(reffile)
                                                simfile = Convert_Type.convert_nc(simfile)

                                                for score in scores:
                                                    score_path = os.path.join(
                                                        self.casedir,
                                                        "scores",
                                                        f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                                                    )
                                                    with xr.open_dataset(score_path) as ds_file:
                                                        ds = Convert_Type.convert_nc(ds_file.load())

                                                    if self.weight.lower() == "area":
                                                        weights = np.cos(np.deg2rad(reffile.lat))
                                                        kk = (
                                                            ds[score]
                                                            .where(np.isfinite(ds[score]), np.nan)
                                                            .weighted(weights)
                                                            .mean(skipna=True)
                                                            .values
                                                        )
                                                    elif self.weight.lower() == "mass":
                                                        area_weights = np.cos(np.deg2rad(reffile.lat))
                                                        flux_weights = np.abs(reffile.mean("time"))
                                                        combined_weights = area_weights * flux_weights
                                                        normalized_weights = combined_weights / combined_weights.sum()
                                                        kk = (
                                                            ds[score]
                                                            .where(np.isfinite(ds[score]), np.nan)
                                                            .weighted(normalized_weights.fillna(0))
                                                            .mean(skipna=True)
                                                            .values
                                                        )
                                                    else:
                                                        kk = ds[score].mean(skipna=True).values

                                                    kk = _finite_reduced_value(
                                                        [kk],
                                                        reducer="mean",
                                                        plot="Parallel Coordinates",
                                                        item=evaluation_item,
                                                        ref_source=ref_source,
                                                        sim_source=sim_source,
                                                        variable=score,
                                                    )
                                                    kk_str = f"{kk:.2f}"
                                                    output_file.write(f"{kk_str}\t")

                                                for metric in metrics:
                                                    metric_path = os.path.join(
                                                        self.casedir,
                                                        "metrics",
                                                        f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc",
                                                    )
                                                    with xr.open_dataset(metric_path) as ds_file:
                                                        ds = Convert_Type.convert_nc(ds_file.load())
                                                    ds = ds.where(np.isfinite(ds), np.nan)
                                                    q_value = ds[metric].quantile(
                                                        [0.05, 0.95], dim=["lat", "lon"], skipna=True
                                                    )
                                                    ds = ds.where((ds >= q_value[0]) & (ds <= q_value[1]), np.nan)
                                                    kk = _finite_reduced_value(
                                                        [ds[metric].median(skipna=True).values],
                                                        reducer="median",
                                                        plot="Parallel Coordinates",
                                                        item=evaluation_item,
                                                        ref_source=ref_source,
                                                        sim_source=sim_source,
                                                        variable=metric,
                                                    )
                                                    kk_str = f"{kk:.2f}"
                                                    output_file.write(f"{kk_str}\t")

                                                output_file.write("\n")
                                        finally:
                                            pass  # Memory cleanup handled at method level
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each evaluation item

            _write_file_atomic(output_file_path, _write_evaluations_csv, suffix=".tmp.csv")

            # Drop only value columns that are entirely missing. Dropping columns
            # with *any* NaN silently removed a metric for every item/simulation
            # when just one row could not produce it.
            df = pd.read_csv(output_file_path, sep="	", header=0)
            all_missing_value_columns = [
                column for column in [*scores, *metrics] if column in df.columns and df[column].isna().all()
            ]
            if all_missing_value_columns:
                logging.warning(
                    "Dropping Parallel Coordinates columns with no finite values: %s",
                    ", ".join(all_missing_value_columns),
                )
                df = df.drop(columns=all_missing_value_columns)
            # If scores or metrics were dropped, then remove the corresponding entries.
            scores = [score for score in scores if score in df.columns]
            metrics = [metric for metric in metrics if metric in df.columns]

            output_file_path1 = os.path.join(dir_path, "Parallel_Coordinates_evaluations_remove_nan.csv")
            _write_csv_atomic(df, output_file_path1, sep="\t", index=False)

            _comparison_callable("make_scenarios_comparison_parallel_coordinates")(
                output_file_path1, self.casedir, evaluation_items, scores, metrics, option
            )
        finally:
            gc.collect()  # Final cleanup for the entire method
