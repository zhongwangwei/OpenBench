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

from openbench.core._comparison_helpers import _apply_pairwise_valid_mask, _atomic_text_writer
from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


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
    return xr.where(obs_var != 0, diff_squared / obs_var, np.nan)


class SingleModelPerformanceIndexComparisonMixin:
    def scenarios_Single_Model_Performance_Index_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        dir_path = os.path.join(f"{basedir}", "comparisons", "Single_Model_Performance_Index")
        # if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        # print(f"Re-creating output directory: {dir_path}")
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

            bootstrap_smpi = np.array(bootstrap_smpi)
            smpi_lower, smpi_upper = np.percentile(bootstrap_smpi, [5, 95])

            return smpi, smpi_lower, smpi_upper

        def process_smpi(casedir, item, ref_source, sim_source, s, o):
            normalized_diff = _smpi_normalized_diff(s, o)

            # Calculate overall SMPI
            smpi = normalized_diff.mean(skipna=True).values
            # Bootstrap for uncertainty estimation — flatten to sample grid points
            n_bootstrap = 1000
            flat_diff = normalized_diff.values.ravel()
            flat_diff = flat_diff[~np.isnan(flat_diff)]
            n_points = len(flat_diff)
            bootstrap_smpi = []
            if n_points == 0:
                smpi_lower, smpi_upper = np.nan, np.nan
            else:
                for _ in range(n_bootstrap):
                    bootstrap_indices = np.random.choice(n_points, size=n_points, replace=True)
                    bootstrap_smpi.append(np.nanmean(flat_diff[bootstrap_indices]))

                bootstrap_smpi = np.array(bootstrap_smpi)
                smpi_lower, smpi_upper = np.percentile(bootstrap_smpi, [5, 95])

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
                            stnlist = os.path.join(
                                basedir, "metrics", f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            )
                            station_list = pd.read_csv(stnlist, header=0)
                            station_list = Convert_Type.convert_Frame(station_list)
                            del_col = ["ID", "sim_lat", "sim_lon", "ref_lon", "ref_lat", "use_syear", "use_eyear"]
                            station_list.drop(
                                columns=[col for col in station_list.columns if col not in del_col], inplace=True
                            )

                            def _process_station_data_parallel(
                                casedir, ref_source, sim_source, item, sim_varname, ref_varname, station_list, iik
                            ):
                                try:
                                    sim_path = os.path.join(
                                        casedir,
                                        "data",
                                        f"stn_{ref_source}_{sim_source}",
                                        f"{item}_sim_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc",
                                    )
                                    ref_path = os.path.join(
                                        casedir,
                                        "data",
                                        f"stn_{ref_source}_{sim_source}",
                                        f"{item}_ref_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc",
                                    )
                                    with xr.open_dataset(sim_path) as sim_ds:
                                        s = sim_ds[sim_varname].squeeze().load()
                                    with xr.open_dataset(ref_path) as ref_ds:
                                        o = ref_ds[ref_varname].squeeze().load()
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

                                    return calculate_smpi(s, o)
                                except (FileNotFoundError, KeyError, ValueError, OSError) as e:
                                    logging.debug(f"Station {station_list['ID'][iik]} skipped in SMPI: {e}")
                                    return None
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
                                    station_list,
                                    iik,
                                )
                                for iik in range(len(station_list["ID"]))
                            )
                            valid_results = [r for r in results if r is not None]
                            if not valid_results:
                                logging.warning("No valid SMPI results for %s %s", evaluation_item, sim_source)
                                continue
                            smpi_values, lower_values, upper_values = zip(*valid_results)
                            mean_smpi = np.nanmean(smpi_values).astype("float32")
                            mean_lower = np.nanmean(lower_values).astype("float32")
                            mean_upper = np.nanmean(upper_values).astype("float32")
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
                                o = o_ds[f"{ref_varname}"].load()
                            with xr.open_dataset(s_path) as s_ds:
                                s = s_ds[f"{sim_varname}"].load()

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
        # After all calculations are done, call the plotting function
        _comparison_callable("make_scenarios_comparison_Single_Model_Performance_Index")(
            basedir, evaluation_items, ref_nml, sim_nml, option
        )

        return
