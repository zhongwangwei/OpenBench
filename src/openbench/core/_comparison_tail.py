"""Tail-end comparison scenarios split out of the main comparison module."""

from __future__ import annotations

import gc
import logging
import os
import sys

import numpy as np
import xarray as xr

from openbench.core._comparison_helpers import (
    _atomic_text_writer,
    _grid_score_mean,
    _require_stat_method,
    _station_csv_column_mean,
)
from openbench.util.converttype import Convert_Type


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class TailComparisonMixin:
    def scenarios_Mann_Kendall_Trend_Test_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        method_name = "Mann_Kendall_Trend_Test"
        method_function = _require_stat_method(self, method_name)
        dir_path = os.path.join(basedir, "comparisons", "Mann_Kendall_Trend_Test")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.compare_nml["Mann_Kendall_Trend_Test"] = {}
        self.compare_nml["Mann_Kendall_Trend_Test"]["significance_level"] = option["significance_level"]
        for evaluation_item in evaluation_items:
            # Get simulation sources
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

            # Convert to lists if needed
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for sim_source in sim_sources:
                # Skip if only one simulation source

                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                if sim_data_type != "stn":
                    try:
                        sim_path = os.path.join(basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc")
                        with xr.open_dataset(sim_path) as sim_ds:
                            sim = sim_ds[f"{sim_varname}"].load()
                        sim = Convert_Type.convert_nc(sim)

                        result = method_function(*[sim])
                        output_file = os.path.join(
                            dir_path, f"Mann_Kendall_Trend_Test_{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                        )
                        self.save_result(output_file, method_name, Convert_Type.convert_nc(result))
                        _comparison_callable("make_Mann_Kendall_Trend_Test")(
                            output_file, method_name, sim_source, self.main_nml["general"], option
                        )
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {sim_source}: {e}"
                        )
                        raise
                else:
                    logging.info(
                        "Skipping %s for %s simulation source %s: station data type is not supported",
                        method_name,
                        evaluation_item,
                        sim_source,
                    )
            for ref_source in ref_sources:
                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                if ref_data_type != "stn":
                    try:
                        ref_path = os.path.join(basedir, "data", f"{evaluation_item}_ref_{ref_source}_{ref_varname}.nc")
                        with xr.open_dataset(ref_path) as ref_ds:
                            ref = ref_ds[f"{ref_varname}"].load()
                        ref = Convert_Type.convert_nc(ref)
                        result = method_function(*[ref])
                        output_file = os.path.join(
                            dir_path, f"Mann_Kendall_Trend_Test_{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                        )
                        self.save_result(output_file, method_name, Convert_Type.convert_nc(result))
                        _comparison_callable("make_Mann_Kendall_Trend_Test")(
                            output_file, method_name, ref_source, self.main_nml["general"], option
                        )
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {ref_source}: {e}"
                        )
                        raise
                else:
                    logging.info(
                        "Skipping %s for %s reference source %s: station data type is not supported",
                        method_name,
                        evaluation_item,
                        ref_source,
                    )

    def scenarios_Standard_Deviation_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            method_name = "Standard_Deviation"
            method_function = _require_stat_method(self, method_name)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for sim_source in sim_sources:
                    try:
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                        if sim_data_type != "stn":
                            # Use os.path.join for file paths
                            sim_path = os.path.join(
                                basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                            )
                            if not os.path.exists(sim_path):
                                raise FileNotFoundError(
                                    f"{method_name}: required simulation input is missing for "
                                    f"{evaluation_item}/{sim_source}: {sim_path}"
                                )
                            with xr.open_dataset(sim_path) as sim_ds:
                                sim = sim_ds[sim_varname].load()
                            sim = Convert_Type.convert_nc(sim)

                            result = method_function(*[sim])

                            output_file = os.path.join(
                                dir_path, f"{method_name}_{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                            )

                            self.save_result(output_file, method_name, Convert_Type.convert_nc(result))
                            _comparison_callable("make_Standard_Deviation")(
                                output_file, method_name, sim_source, self.main_nml["general"], option
                            )
                        else:
                            logging.info(
                                f"Skipping {method_name} for {evaluation_item} {sim_source}: station data type."
                            )
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {sim_source}: {e}"
                        )
                        raise
                    finally:
                        # Clean up memory after each simulation source
                        gc.collect()

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]

                        if ref_data_type != "stn":
                            # Use os.path.join for file paths
                            ref_path = os.path.join(
                                basedir, "data", f"{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                            )
                            if not os.path.exists(ref_path):
                                raise FileNotFoundError(
                                    f"{method_name}: required reference input is missing for "
                                    f"{evaluation_item}/{ref_source}: {ref_path}"
                                )
                            with xr.open_dataset(ref_path) as ref_ds:
                                ref = ref_ds[ref_varname].load()
                            ref = Convert_Type.convert_nc(ref)

                            result = method_function(*[ref])

                            output_file = os.path.join(
                                dir_path, f"{method_name}_{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                            )

                            self.save_result(output_file, method_name, Convert_Type.convert_nc(result))
                            _comparison_callable("make_Standard_Deviation")(
                                output_file, method_name, ref_source, self.main_nml["general"], option
                            )
                        else:
                            logging.info(
                                f"Skipping {method_name} for {evaluation_item} {ref_source}: station data type."
                            )
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {ref_source}: {e}"
                        )
                        raise
                    finally:
                        # Clean up memory after each reference source
                        gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()

    def scenarios_Functional_Response_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        self.compare_nml["Functional_Response"] = {}
        self.compare_nml["Functional_Response"]["nbins"] = option["nbins"]
        try:
            method_name = "Functional_Response"
            method_function = _require_stat_method(self, method_name)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]

                        if ref_data_type != "stn":
                            # Use os.path.join for file paths
                            ref_path = os.path.join(
                                basedir, "data", f"{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                            )
                            with xr.open_dataset(ref_path) as ref_ds:
                                ref = ref_ds[ref_varname].load()
                            ref = Convert_Type.convert_nc(ref)

                            for sim_source in sim_sources:
                                try:
                                    sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                    sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                    if sim_data_type != "stn":
                                        # Use os.path.join for file paths
                                        sim_path = os.path.join(
                                            basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                                        )
                                        with xr.open_dataset(sim_path) as sim_ds:
                                            sim = sim_ds[sim_varname].load()
                                        sim = Convert_Type.convert_nc(sim)

                                        result = method_function(*[ref, sim])

                                        output_file = os.path.join(
                                            dir_path,
                                            f"{method_name}_{evaluation_item}_ref_{ref_source}_sim_{sim_source}.nc",
                                        )

                                        self.save_result(output_file, method_name, Convert_Type.convert_nc(result))
                                        _comparison_callable("make_Functional_Response")(
                                            output_file, method_name, sim_source, self.main_nml["general"], option
                                        )
                                except Exception as e:
                                    logging.error(
                                        f"Error processing {method_name} calculations for {evaluation_item} {ref_source} {sim_source}: {e}"
                                    )
                                    raise
                                finally:
                                    # Clean up memory after each simulation source
                                    gc.collect()
                    finally:
                        # Clean up memory after each reference source
                        gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()

    def scenarios_RadarMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "RadarMap")
            os.makedirs(dir_path, exist_ok=True)

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
                # try:
                _comparison_callable("make_scenarios_comparison_radar_map")(output_file_path, score, option)
                # except Exception as e:
                #     logging.error(f"Error processing RadarMap for {score}: {e}")
        finally:
            gc.collect()  # Clean up memory after processing

    def scenarios_Correlation_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            method_name = "Correlation"
            method_function = _require_stat_method(self, method_name)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if len(sim_sources) < 2:
                    continue

                for i, sim1 in enumerate(sim_sources):
                    for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                        try:
                            sim_varname1 = sim_nml[f"{evaluation_item}"][f"{sim1}_varname"]
                            sim_varname2 = sim_nml[f"{evaluation_item}"][f"{sim2}_varname"]
                            sim_data_type1 = sim_nml[f"{evaluation_item}"][f"{sim1}_data_type"]
                            sim_data_type2 = sim_nml[f"{evaluation_item}"][f"{sim2}_data_type"]
                            if sim_data_type1 == "stn" or sim_data_type2 == "stn":
                                logging.warning(
                                    f"Cannot compare station and gridded data together for {evaluation_item}"
                                )
                                logging.warning(
                                    "All simulation sources must be gridded data; skipping this evaluation item"
                                )
                                continue

                            # If a varname is empty, use the evaluation item
                            # locally without mutating the caller's sim_nml.
                            if sim_varname1 is None or sim_varname1 == "":
                                sim_varname1 = evaluation_item
                            if sim_varname2 is None or sim_varname2 == "":
                                sim_varname2 = evaluation_item
                            # Use os.path.join for file paths

                            ds1_path = os.path.join(basedir, "data", f"{evaluation_item}_sim_{sim1}_{sim_varname1}.nc")
                            ds2_path = os.path.join(basedir, "data", f"{evaluation_item}_sim_{sim2}_{sim_varname2}.nc")

                            with xr.open_dataset(ds1_path) as ds1_file:
                                ds1 = ds1_file[sim_varname1].load()
                            with xr.open_dataset(ds2_path) as ds2_file:
                                ds2 = ds2_file[sim_varname2].load()

                            ds1 = Convert_Type.convert_nc(ds1)
                            ds2 = Convert_Type.convert_nc(ds2)

                            result = method_function(*[ds1, ds2])

                            output_file = os.path.join(
                                dir_path, f"{method_name}_{evaluation_item}_{sim1}_and_{sim2}.nc"
                            )

                            self.save_result(output_file, method_name, Convert_Type.convert_nc(result))
                            _comparison_callable("make_Correlation")(
                                output_file, method_name, self.main_nml["general"], option
                            )

                        except Exception as e:
                            logging.error(
                                f"Error processing {method_name} calculations for {evaluation_item} {sim1} and {sim2}: {e}"
                            )
                            raise
                        finally:
                            # Clean up memory after each iteration
                            gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()
