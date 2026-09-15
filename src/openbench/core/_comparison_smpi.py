"""Single Model Performance Index comparison scenario."""

from __future__ import annotations

import gc
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from joblib import delayed

from openbench.core._comparison_helpers import (
    _apply_pairwise_valid_mask,
    _atomic_text_writer,
    _load_station_pair,
    _station_evaluation_frame,
    _write_csv_atomic,
)
from openbench.data.station_missing import StationDataUnavailable
from openbench.util.converttype import Convert_Type
from openbench.util.names import select_data_array
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def _station_metadata_for_results(station_list: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Keep station metadata only before appending freshly computed result columns."""
    metadata_columns = [
        "ID",
        "sim_lat",
        "sim_lon",
        "ref_lon",
        "ref_lat",
        "lon",
        "lat",
        "use_syear",
        "use_eyear",
    ]
    return station_list[[column for column in metadata_columns if column in station_list.columns]]


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


def _smpi_normalized_diff(s, o):
    obs_var = o.var(dim="time", ddof=1)
    s_climate = s.mean(dim="time")
    o_climate = o.mean(dim="time")
    diff_squared = (s_climate - o_climate) ** 2
    return diff_squared / obs_var.where(obs_var != 0)


def _smpi_scalar(value) -> float:
    arr = np.asarray(value).squeeze()
    if arr.size != 1:
        return np.nan
    return float(arr.item())


def _smpi_spatial_weights(weight: str, reference: xr.DataArray) -> xr.DataArray | None:
    mode = str(weight or "none").lower()
    if mode == "area":
        return xr.DataArray(
            np.cos(np.deg2rad(reference["lat"])),
            coords={"lat": reference["lat"]},
            dims=("lat",),
        )
    if mode == "mass":
        area_weights = xr.DataArray(
            np.cos(np.deg2rad(reference["lat"])),
            coords={"lat": reference["lat"]},
            dims=("lat",),
        )
        combined_weights = area_weights * np.abs(reference.mean("time"))
        total = _smpi_scalar(combined_weights.sum(skipna=True).values)
        if not np.isfinite(total) or total == 0:
            return combined_weights.fillna(0)
        return (combined_weights / total).fillna(0)
    return None


def _smpi_weighted_mean(normalized_diff: xr.DataArray, weights: xr.DataArray | None) -> float:
    normalized_diff = normalized_diff.where(np.isfinite(normalized_diff))
    if weights is None:
        return _smpi_scalar(normalized_diff.mean(skipna=True).values)
    return _smpi_scalar(normalized_diff.weighted(weights).mean(skipna=True).values)


def _smpi_percentile_interval(samples) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    lower, upper = np.percentile(values, [5, 95])
    return float(lower), float(upper)


def _smpi_grid_summary(s, o, *, weight: str, n_bootstrap: int = 1000):
    normalized_diff = _smpi_normalized_diff(s, o)
    weights = _smpi_spatial_weights(weight, o)
    smpi = _smpi_weighted_mean(normalized_diff, weights)

    if n_bootstrap <= 0 or s.sizes.get("time", 0) == 0:
        return smpi, np.nan, np.nan, normalized_diff

    bootstrap_smpi = []
    n_time = len(s["time"])
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_time, size=n_time, replace=True)
        s_boot = s.isel(time=idx)
        o_boot = o.isel(time=idx)
        smpi_boot = _smpi_weighted_mean(_smpi_normalized_diff(s_boot, o_boot), weights)
        if np.isfinite(smpi_boot):
            bootstrap_smpi.append(smpi_boot)

    if not bootstrap_smpi:
        return smpi, np.nan, np.nan, normalized_diff

    smpi_lower, smpi_upper = _smpi_percentile_interval(bootstrap_smpi)
    return smpi, smpi_lower, smpi_upper, normalized_diff


class SingleModelPerformanceIndexComparisonMixin:
    def scenarios_Single_Model_Performance_Index_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        dir_path = os.path.join(f"{basedir}", "comparisons", "Single_Model_Performance_Index")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        def calculate_smpi(s, o):
            normalized_diff = _smpi_normalized_diff(s, o)
            smpi = float(normalized_diff.mean(skipna=True))

            # Bootstrap on time dimension for uncertainty estimation
            n_bootstrap = 100
            n_time = len(s["time"])
            bootstrap_smpi = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(n_time, size=n_time, replace=True)
                s_boot = s.isel(time=idx)
                o_boot = o.isel(time=idx)
                smpi_boot = float(_smpi_normalized_diff(s_boot, o_boot).mean(skipna=True))
                bootstrap_smpi.append(smpi_boot)

            smpi_lower, smpi_upper = _smpi_percentile_interval(bootstrap_smpi)

            return smpi, smpi_lower, smpi_upper

        def process_smpi(casedir, item, ref_source, sim_source, s, o):
            smpi, smpi_lower, smpi_upper, normalized_diff = _smpi_grid_summary(
                s,
                o,
                weight=self.weight,
                n_bootstrap=1000,
            )

            # Save grid-based SMPI
            try:
                smpi_da = normalized_diff.rename("SMPI")
                output_path = os.path.join(
                    casedir,
                    "comparisons",
                    "Single_Model_Performance_Index",
                    f"{item}_ref_{ref_source}_sim_{sim_source}_SMPI_grid.nc",
                )
                _write_netcdf_atomic(smpi_da, output_path)
                del smpi_da  # Release memory
                gc.collect()  # Force garbage collection
            except Exception as e:
                logging.error(f"Error saving grid-based SMPI: {e}")
                raise

            return smpi, smpi_lower, smpi_upper

        output_file_path = f"{dir_path}/SMPI_comparison.csv"

        with _atomic_text_writer(output_file_path) as output_file:
            output_file.write("Item\tReference\tSimulation\tSMPI\tLower_CI\tUpper_CI\n")
            for evaluation_item in evaluation_items:
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        output_file.write(f"{evaluation_item}\t{ref_source}\t{sim_source}\t")
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                        if ref_data_type == "stn" or sim_data_type == "stn":
                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            if sim_varname is None or sim_varname == "":
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == "":
                                ref_varname = evaluation_item
                            station_list = _station_evaluation_frame(
                                basedir, evaluation_item, ref_source, sim_source, kind="metrics"
                            )

                            def _process_station_data_parallel(
                                casedir, ref_source, sim_source, item, sim_varname, ref_varname, station_row
                            ):
                                try:
                                    s, o = _load_station_pair(
                                        self,
                                        casedir,
                                        item,
                                        ref_source,
                                        sim_source,
                                        station_row,
                                        ref_varname,
                                        sim_varname,
                                    )
                                    smpi, lower, upper = calculate_smpi(s, o)
                                    result = {"SMPI": smpi, "Lower_CI": lower, "Upper_CI": upper}
                                    if not np.isfinite(list(result.values())).any():
                                        result.update(
                                            {
                                                "status": "unavailable",
                                                "reason": "computed SMPI metrics are undefined",
                                            }
                                        )
                                    else:
                                        result.update({"status": "ok", "reason": ""})
                                    return result
                                except StationDataUnavailable as exc:
                                    return {
                                        "SMPI": np.nan,
                                        "Lower_CI": np.nan,
                                        "Upper_CI": np.nan,
                                        "status": "unavailable",
                                        "reason": str(exc),
                                    }
                                finally:
                                    gc.collect()  # Clean up memory after processing

                            results = self._run_parallel_or_serial(
                                delayed(_process_station_data_parallel)(
                                    basedir,
                                    ref_source,
                                    sim_source,
                                    evaluation_item,
                                    sim_varname,
                                    ref_varname,
                                    station_row,
                                )
                                for _, station_row in station_list.iterrows()
                            )
                            result_frame = pd.DataFrame(results)
                            station_list = pd.concat(
                                [_station_metadata_for_results(station_list, result_frame), result_frame],
                                axis=1,
                            )
                            station_list = Convert_Type.convert_Frame(station_list)
                            output_stn_path = os.path.join(
                                dir_path, f"SMPI_{evaluation_item}_stn_{ref_source}_{sim_source}.csv"
                            )
                            _write_csv_atomic(station_list, output_stn_path, index=False)

                            smpi_values = np.asarray(station_list["SMPI"], dtype=float)
                            lower_values = np.asarray(station_list["Lower_CI"], dtype=float)
                            upper_values = np.asarray(station_list["Upper_CI"], dtype=float)
                            mean_smpi = float(np.nanmean(smpi_values)) if np.isfinite(smpi_values).any() else np.nan
                            mean_lower = float(np.nanmean(lower_values)) if np.isfinite(lower_values).any() else np.nan
                            mean_upper = float(np.nanmean(upper_values)) if np.isfinite(upper_values).any() else np.nan
                            output_file.write(f"{mean_smpi:.4f}\t{mean_lower:.4f}\t{mean_upper:.4f}\n")

                        else:
                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            if sim_varname is None or sim_varname == "":
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == "":
                                ref_varname = evaluation_item
                            o_path = self._ref_data_path(basedir, evaluation_item, ref_source, ref_varname, sim_source)
                            s_path = os.path.join(
                                basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                            )

                            with xr.open_dataset(o_path) as o_ds:
                                o = select_data_array(o_ds, ref_varname, evaluation_item).load()
                            with xr.open_dataset(s_path) as s_ds:
                                s = select_data_array(s_ds, sim_varname, evaluation_item).load()

                            o = Convert_Type.convert_nc(o)
                            s = Convert_Type.convert_nc(s)

                            # Align time axes safely. The previous code did an unconditional

                            # s["time"] = o["time"], which (a) raises when lengths differ and

                            # (b) silently pairs values against wrong timestamps when lengths

                            # match but coords are offset. Use inner-join intersect instead.

                            if s.sizes.get("time") != o.sizes.get("time") or not np.array_equal(
                                s["time"].values, o["time"].values
                            ):
                                s, o = xr.align(s, o, join="inner")
                            s, o = _apply_pairwise_valid_mask(s, o)

                            smpi, lower, upper = process_smpi(basedir, evaluation_item, ref_source, sim_source, s, o)
                            output_file.write(f"{smpi:.4f}\t{lower:.4f}\t{upper:.4f}\n")

                logging.info(f"Completed SMPI calculation for {evaluation_item}")
                logging.info("===============================================================================")
        _comparison_callable("make_scenarios_comparison_Single_Model_Performance_Index")(
            basedir, evaluation_items, ref_nml, sim_nml, option
        )

        return
