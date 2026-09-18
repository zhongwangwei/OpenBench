"""Target diagram comparison scenario."""

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
    _atomic_text_writer,
    _load_station_pair,
    _station_evaluation_frame,
    _write_csv_atomic,
)
from openbench.data.station_missing import StationDataUnavailable
from openbench.util.converttype import Convert_Type
from openbench.util.names import select_data_array
from openbench.util.filenames import join_filename_components


def _to_float(value):
    arr = np.asarray(value).squeeze()
    return float(arr.item()) if arr.size == 1 else np.nan


def _mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


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


class TargetDiagramComparisonMixin:
    def scenarios_Target_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "Target_Diagram")
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
                            output_file_path = os.path.join(
                                dir_path,
                                f"{join_filename_components('target_diagram', evaluation_item, ref_source)}.csv",
                            )

                            with _atomic_text_writer(output_file_path) as output_file:
                                output_file.write("Item\t")
                                output_file.write("Reference\t")
                                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                                if isinstance(sim_sources, str):
                                    sim_sources = [sim_sources]
                                for sim_source in sim_sources:
                                    # Column order must match the values written below:
                                    # bias, then total RMSE (=RMSD), then centered CRMSD.
                                    output_file.write(f"{sim_source}_bias\t")
                                    output_file.write(f"{sim_source}_rmsd\t")
                                    output_file.write(f"{sim_source}_crmsd\t")

                                output_file.write("\n")  # Move "All" to the first line
                                output_file.write(f"{evaluation_item}\t")
                                output_file.write(f"{ref_source}\t")
                                biases = np.zeros(len(sim_sources))
                                rmses = np.zeros(len(sim_sources))
                                crmsds = np.zeros(len(sim_sources))
                                for i, sim_source in enumerate(sim_sources):
                                    try:
                                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                        if isinstance(sim_sources, str):
                                            sim_sources = [sim_sources]
                                        if isinstance(ref_sources, str):
                                            ref_sources = [ref_sources]
                                        if ref_data_type == "stn" or sim_data_type == "stn":
                                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                            if sim_varname is None or sim_varname == "":
                                                sim_varname = evaluation_item
                                            if ref_varname is None or ref_varname == "":
                                                ref_varname = evaluation_item
                                            station_list = _station_evaluation_frame(
                                                casedir, evaluation_item, ref_source, sim_source, kind="metrics"
                                            )

                                            def _make_validation_parallel(
                                                casedir,
                                                ref_source,
                                                sim_source,
                                                item,
                                                sim_varname,
                                                ref_varname,
                                                station_row,
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
                                                    result = {
                                                        "CRMSD": _to_float(self.CRMSD(s, o)),
                                                        "bias": _to_float(self.bias(s, o)),
                                                        "rmse": _to_float(self.RMSE(s, o)),
                                                    }
                                                    if not np.isfinite(list(result.values())).any():
                                                        result.update(
                                                            {
                                                                "status": "unavailable",
                                                                "reason": "computed target metrics are undefined",
                                                            }
                                                        )
                                                    else:
                                                        result.update({"status": "ok", "reason": ""})
                                                    return result
                                                except StationDataUnavailable as exc:
                                                    return {
                                                        "CRMSD": np.nan,
                                                        "bias": np.nan,
                                                        "rmse": np.nan,
                                                        "status": "unavailable",
                                                        "reason": str(exc),
                                                    }

                                            results = self._run_parallel_or_serial(
                                                delayed(_make_validation_parallel)(
                                                    casedir,
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
                                                [
                                                    _station_metadata_for_results(station_list, result_frame),
                                                    result_frame,
                                                ],
                                                axis=1,
                                            )
                                            station_list = Convert_Type.convert_Frame(station_list)

                                            output_stn_path = os.path.join(
                                                dir_path,
                                                f"target_diagram_{evaluation_item}_stn_{ref_source}_{sim_source}.csv",
                                            )
                                            _write_csv_atomic(station_list, output_stn_path, index=False)

                                            bias_sim = _mean_or_nan(station_list["bias"])
                                            output_file.write(f"{bias_sim}	")
                                            biases[i] = bias_sim

                                            rmse_sim = _mean_or_nan(station_list["rmse"])
                                            output_file.write(f"{rmse_sim}	")
                                            rmses[i] = rmse_sim

                                            crmsd_sim = _mean_or_nan(station_list["CRMSD"])
                                            output_file.write(f"{crmsd_sim}	")
                                            crmsds[i] = crmsd_sim
                                        else:
                                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                            if sim_varname is None or sim_varname == "":
                                                sim_varname = evaluation_item
                                            if ref_varname is None or ref_varname == "":
                                                ref_varname = evaluation_item

                                            ref_path = self._ref_data_path(
                                                casedir, evaluation_item, ref_source, ref_varname, sim_source
                                            )
                                            sim_path = os.path.join(
                                                casedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                                            )

                                            with xr.open_dataset(ref_path) as ref_ds:
                                                reffile = select_data_array(ref_ds, ref_varname).load()
                                            with xr.open_dataset(sim_path) as sim_ds:
                                                simfile = select_data_array(sim_ds, sim_varname).load()
                                            reffile = Convert_Type.convert_nc(reffile)
                                            simfile = Convert_Type.convert_nc(simfile)

                                            bias_sim = self.bias(simfile, reffile).mean(skipna=True).values
                                            output_file.write(f"{bias_sim}\t")
                                            biases[i] = bias_sim
                                            rmse_sim = self.RMSE(simfile, reffile).mean(skipna=True).values
                                            output_file.write(f"{rmse_sim}\t")
                                            rmses[i] = rmse_sim
                                            crmsd_sim = self.CRMSD(simfile, reffile).mean(skipna=True).values
                                            output_file.write(f"{crmsd_sim}\t")
                                            crmsds[i] = crmsd_sim
                                    finally:
                                        pass  # Memory cleanup handled at method level

                                output_file.write("\n")
                                try:
                                    _comparison_callable("make_scenarios_comparison_Target_Diagram")(
                                        dir_path,
                                        evaluation_item,
                                        biases,
                                        # Target diagram expects (bias, crmsd, rmsd):
                                        # crmsd slot = centered/unbiased RMSD (x-axis uRMSD),
                                        # rmsd slot = total RMSD. Do NOT swap these.
                                        crmsds,
                                        rmses,
                                        ref_source,
                                        sim_sources,
                                        option,
                                    )
                                except (ValueError, RuntimeError, IOError, OSError) as e:
                                    logging.error(
                                        f"Error: {evaluation_item} {ref_source} Target diagram generation failed: {e}"
                                    )
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()
