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
    _apply_pairwise_valid_mask,
    _atomic_text_writer,
    _require_station_diagram_results,
    _write_csv_atomic,
)
from openbench.util.converttype import Convert_Type
from openbench.util.filenames import join_filename_components


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

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                try:
                    # read the simulation source and reference source
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                    # if the sim_sources and ref_sources are not list, then convert them to list
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
                                # ill determine the number of simulation sources
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
                                            stnlist = os.path.join(
                                                casedir,
                                                "metrics",
                                                f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                            )
                                            station_list = pd.read_csv(stnlist, header=0)
                                            station_list = Convert_Type.convert_Frame(station_list)
                                            del_col = [
                                                "ID",
                                                "sim_lat",
                                                "sim_lon",
                                                "ref_lon",
                                                "ref_lat",
                                                "use_syear",
                                                "use_eyear",
                                            ]
                                            station_list.drop(
                                                columns=[col for col in station_list.columns if col not in del_col],
                                                inplace=True,
                                            )

                                            def _make_validation_parallel(
                                                casedir,
                                                ref_source,
                                                sim_source,
                                                item,
                                                sim_varname,
                                                ref_varname,
                                                station_list,
                                                iik,
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

                                                    if s.sizes.get("time") != o.sizes.get("time") or not np.array_equal(
                                                        s["time"].values, o["time"].values
                                                    ):
                                                        s, o = xr.align(s, o, join="inner")
                                                    s, o = _apply_pairwise_valid_mask(s, o)
                                                    row = {}
                                                    try:
                                                        row["CRMSD"] = self.CRMSD(s, o).values
                                                    except (ValueError, RuntimeError, AttributeError) as e:
                                                        logging.debug(f"CRMSD calculation failed: {e}")
                                                        row["CRMSD"] = np.nan
                                                    try:
                                                        row["bias"] = self.bias(s, o).values
                                                    except (ValueError, RuntimeError, AttributeError) as e:
                                                        logging.debug(f"bias calculation failed: {e}")
                                                        row["bias"] = np.nan
                                                    try:
                                                        row["rmse"] = self.RMSE(s, o).values
                                                    except (ValueError, RuntimeError, AttributeError) as e:
                                                        logging.debug(f"rmse calculation failed: {e}")
                                                        row["rmse"] = np.nan
                                                    return row
                                                except (FileNotFoundError, KeyError, ValueError, OSError) as e:
                                                    logging.debug(
                                                        f"Station {station_list['ID'][iik]} skipped in Target: {e}"
                                                    )
                                                    return None
                                                finally:
                                                    pass  # Memory cleanup handled at higher level

                                            results = self._run_parallel_or_serial(
                                                delayed(_make_validation_parallel)(
                                                    casedir,
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

                                            # Replace None results with empty dicts so pd.DataFrame
                                            # produces zero-length rows that align by ID rather than
                                            # 0-column placeholder rows that break column alignment
                                            # in the subsequent concat.
                                            results_clean = [r if isinstance(r, dict) else {} for r in results]
                                            _require_station_diagram_results(
                                                results_clean,
                                                diagram="Target",
                                                item=evaluation_item,
                                                ref_source=ref_source,
                                                sim_source=sim_source,
                                            )
                                            results_df = pd.DataFrame(results_clean, index=station_list.index)
                                            station_list = pd.concat([station_list, results_df], axis=1)
                                            station_list = Convert_Type.convert_Frame(station_list)

                                            output_stn_path = os.path.join(
                                                dir_path,
                                                f"target_diagram_{evaluation_item}_stn_{ref_source}_{sim_source}.csv",
                                            )
                                            _write_csv_atomic(station_list, output_stn_path)

                                            station_list = pd.read_csv(output_stn_path, header=0)
                                            station_list = Convert_Type.convert_Frame(station_list)
                                            bias_sim = station_list["bias"].mean(skipna=True)
                                            output_file.write(f"{bias_sim}\t")
                                            biases[i] = bias_sim

                                            rmse_sim = station_list["rmse"].mean(skipna=True)
                                            output_file.write(f"{rmse_sim}\t")
                                            rmses[i] = rmse_sim

                                            crmsd_sim = station_list["CRMSD"].mean(skipna=True)
                                            output_file.write(f"{crmsd_sim}\t")
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
                                                reffile = ref_ds[ref_varname].load()
                                            with xr.open_dataset(sim_path) as sim_ds:
                                                simfile = sim_ds[sim_varname].load()
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
