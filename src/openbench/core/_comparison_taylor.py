"""Taylor diagram comparison scenario."""

from __future__ import annotations

import gc
import logging
import os
import sys
from typing import NamedTuple

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


def _taylor_standard_deviation(data):
    if isinstance(data, xr.Dataset):
        data = list(data.data_vars.values())[0]
    return data.std(dim="time", ddof=0)


class TaylorSummaryStatistics(NamedTuple):
    std_sim: float
    cor_sim: float
    diagram_crmsd: float
    mean_crmsd: float
    std_ref: float


def _to_scalar_float(value) -> float:
    arr = np.asarray(value).squeeze()
    if arr.size != 1:
        return np.nan
    return float(arr.item())


def _taylor_implied_crmsd(std_sim, cor_sim, std_ref) -> float:
    std_sim = _to_scalar_float(std_sim)
    cor_sim = _to_scalar_float(cor_sim)
    std_ref = _to_scalar_float(std_ref)
    if not np.all(np.isfinite([std_sim, cor_sim, std_ref])):
        return np.nan
    cor_sim = float(np.clip(cor_sim, -1.0, 1.0))
    radicand = std_sim**2 + std_ref**2 - 2.0 * std_sim * std_ref * cor_sim
    return float(np.sqrt(max(radicand, 0.0)))


def _taylor_summary_statistics(std_sim, cor_sim, mean_crmsd, std_ref) -> TaylorSummaryStatistics:
    std_sim = _to_scalar_float(std_sim)
    cor_sim = _to_scalar_float(cor_sim)
    mean_crmsd = _to_scalar_float(mean_crmsd)
    std_ref = _to_scalar_float(std_ref)
    return TaylorSummaryStatistics(
        std_sim=std_sim,
        cor_sim=cor_sim,
        diagram_crmsd=_taylor_implied_crmsd(std_sim, cor_sim, std_ref),
        mean_crmsd=mean_crmsd,
        std_ref=std_ref,
    )


def _write_taylor_summary(output_file, stds, cors, RMSs, index: int, summary: TaylorSummaryStatistics) -> None:
    output_file.write(f"{summary.std_sim}\t")
    output_file.write(f"{summary.cor_sim}\t")
    output_file.write(f"{summary.diagram_crmsd}\t")
    output_file.write(f"{summary.mean_crmsd}\t")
    output_file.write(f"{summary.std_ref}\t")
    stds[index] = summary.std_sim
    cors[index] = summary.cor_sim
    RMSs[index] = summary.diagram_crmsd


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class TaylorDiagramComparisonMixin:
    def scenarios_Taylor_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "Taylor_Diagram")
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
                                f"{join_filename_components('taylor_diagram', evaluation_item, ref_source)}.csv",
                            )
                            with _atomic_text_writer(output_file_path) as output_file:
                                output_file.write("Item\t")
                                output_file.write("Reference\t")
                                for sim_source in sim_sources:
                                    output_file.write(f"{sim_source}_std\t")
                                    output_file.write(f"{sim_source}_COR\t")
                                    output_file.write(f"{sim_source}_RMS\t")
                                    output_file.write(f"{sim_source}_RMS_mean\t")
                                    output_file.write(f"{sim_source}_std_ref\t")

                                output_file.write("Reference_std\t")
                                output_file.write("\n")  # Move "All" to the first line
                                output_file.write(f"{evaluation_item}\t")
                                output_file.write(f"{ref_source}\t")
                                stds = np.zeros(len(sim_sources) + 1)
                                cors = np.zeros(len(sim_sources) + 1)
                                RMSs = np.zeros(len(sim_sources) + 1)
                                for i, sim_source in enumerate(sim_sources):
                                    try:
                                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                        sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                        # ugly code, need to be improved
                                        # if self.sim_varname is empty, then set it to item
                                        if sim_varname is None or sim_varname == "":
                                            sim_varname = evaluation_item
                                        if ref_varname is None or ref_varname == "":
                                            ref_varname = evaluation_item
                                        if ref_data_type == "stn" or sim_data_type == "stn":
                                            stnlist = os.path.join(
                                                casedir,
                                                "metrics",
                                                f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                            )
                                            station_list = pd.read_csv(stnlist, header=0)
                                            station_list = Convert_Type.convert_Frame(station_list)
                                            # Keep existing evaluation columns and required station metadata.
                                            required_cols = {"ID", "use_syear", "use_eyear"}
                                            missing_cols = required_cols.difference(station_list.columns)
                                            if missing_cols:
                                                raise KeyError(
                                                    f"Station evaluation CSV missing required Taylor columns: {sorted(missing_cols)}"
                                                )
                                            # this should be moved to other place
                                            if ref_source.lower() == "grdc" and {"lon", "lat"}.issubset(
                                                station_list.columns
                                            ):
                                                station_list["ref_lon"] = station_list["lon"]
                                                station_list["ref_lat"] = station_list["lat"]

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
                                                        row["std_s"] = _taylor_standard_deviation(s).values
                                                    except (ValueError, RuntimeError, AttributeError) as e:
                                                        logging.debug(f"std_s calculation failed: {e}")
                                                        row["std_s"] = np.nan
                                                    try:
                                                        row["std_o"] = _taylor_standard_deviation(o).values
                                                    except (ValueError, RuntimeError, AttributeError) as e:
                                                        logging.debug(f"std_o calculation failed: {e}")
                                                        row["std_o"] = np.nan
                                                    try:
                                                        row["CRMSD"] = self.CRMSD(s, o).values
                                                    except (ValueError, RuntimeError, AttributeError) as e:
                                                        logging.debug(f"CRMSD calculation failed: {e}")
                                                        row["CRMSD"] = np.nan
                                                    try:
                                                        row["correlation"] = self.correlation(s, o).values
                                                    except (ValueError, RuntimeError, AttributeError) as e:
                                                        logging.debug(f"correlation calculation failed: {e}")
                                                        row["correlation"] = np.nan
                                                    return row
                                                except (FileNotFoundError, KeyError, ValueError, OSError) as e:
                                                    logging.debug(
                                                        f"Station {station_list['ID'][iik]} skipped in Taylor: {e}"
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
                                                diagram="Taylor",
                                                item=evaluation_item,
                                                ref_source=ref_source,
                                                sim_source=sim_source,
                                            )
                                            results_df = pd.DataFrame(results_clean, index=station_list.index)
                                            station_list = pd.concat([station_list, results_df], axis=1)
                                            station_list = Convert_Type.convert_Frame(station_list)

                                            output_stn_path = os.path.join(
                                                dir_path,
                                                f"taylor_diagram_{evaluation_item}_stn_{ref_source}_{sim_source}.csv",
                                            )
                                            _write_csv_atomic(station_list, output_stn_path)

                                            station_list = pd.read_csv(output_stn_path, header=0)
                                            std_sim = station_list["std_s"].mean(skipna=True)
                                            cor_sim = station_list["correlation"].mean(skipna=True)
                                            std_ref = station_list["std_o"].mean(skipna=True)
                                            mean_crmsd = station_list["CRMSD"].mean(skipna=True)
                                            summary = _taylor_summary_statistics(
                                                std_sim=std_sim,
                                                cor_sim=cor_sim,
                                                mean_crmsd=mean_crmsd,
                                                std_ref=std_ref,
                                            )
                                            _write_taylor_summary(output_file, stds, cors, RMSs, i + 1, summary)

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

                                            std_sim_result = _taylor_standard_deviation(simfile)
                                            cor_result = self.correlation(simfile, reffile)
                                            RMS_result = self.CRMSD(simfile, reffile)

                                            if self.weight.lower() == "area":
                                                weights = np.cos(np.deg2rad(reffile.lat))
                                                std_sim = (
                                                    std_sim_result.where(np.isfinite(std_sim_result))
                                                    .weighted(weights)
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                                cor_sim = (
                                                    cor_result.where(np.isfinite(cor_result))
                                                    .weighted(weights)
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                                mean_crmsd = (
                                                    RMS_result.where(np.isfinite(RMS_result))
                                                    .weighted(weights)
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                            elif self.weight.lower() == "mass":
                                                # Calculate area weights (cosine of latitude)
                                                area_weights = np.cos(np.deg2rad(reffile.lat))
                                                # Calculate absolute flux weights
                                                flux_weights = np.abs(reffile.mean("time"))
                                                # Combine area and flux weights
                                                combined_weights = area_weights * flux_weights
                                                # Normalize weights to sum to 1
                                                normalized_weights = combined_weights / combined_weights.sum()
                                                # Calculate weighted mean
                                                std_sim = (
                                                    std_sim_result.where(np.isfinite(std_sim_result))
                                                    .weighted(normalized_weights.fillna(0))
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                                cor_sim = (
                                                    cor_result.where(np.isfinite(cor_result))
                                                    .weighted(normalized_weights.fillna(0))
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                                mean_crmsd = (
                                                    RMS_result.where(np.isfinite(RMS_result))
                                                    .weighted(normalized_weights.fillna(0))
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                            else:
                                                std_sim = (
                                                    std_sim_result.where(np.isfinite(std_sim_result))
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                                cor_sim = (
                                                    cor_result.where(np.isfinite(cor_result)).mean(skipna=True).values
                                                )
                                                mean_crmsd = (
                                                    RMS_result.where(np.isfinite(RMS_result)).mean(skipna=True).values
                                                )

                                            if self.weight.lower() == "area":
                                                weights = np.cos(np.deg2rad(reffile.lat))
                                                std_ref = (
                                                    _taylor_standard_deviation(reffile)
                                                    .where(np.isfinite(_taylor_standard_deviation(reffile)))
                                                    .weighted(weights)
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                            elif self.weight.lower() == "mass":
                                                # Calculate area weights (cosine of latitude)
                                                area_weights = np.cos(np.deg2rad(reffile.lat))
                                                # Calculate absolute flux weights
                                                flux_weights = np.abs(reffile.mean("time"))
                                                # Combine area and flux weights
                                                combined_weights = area_weights * flux_weights
                                                # Normalize weights to sum to 1
                                                normalized_weights = combined_weights / combined_weights.sum()
                                                # Calculate weighted mean
                                                std_ref = (
                                                    _taylor_standard_deviation(reffile)
                                                    .where(np.isfinite(_taylor_standard_deviation(reffile)))
                                                    .weighted(normalized_weights.fillna(0))
                                                    .mean(skipna=True)
                                                    .values
                                                )
                                            else:
                                                std_ref = _taylor_standard_deviation(reffile).mean(skipna=True).values

                                            summary = _taylor_summary_statistics(
                                                std_sim=std_sim,
                                                cor_sim=cor_sim,
                                                mean_crmsd=mean_crmsd,
                                                std_ref=std_ref,
                                            )
                                            _write_taylor_summary(output_file, stds, cors, RMSs, i + 1, summary)

                                        stds[0] = _to_scalar_float(std_ref)
                                    finally:
                                        pass  # Memory cleanup handled at method level
                                output_file.write(f"{stds[0]}\t")
                                output_file.write("\n")
                            try:
                                _comparison_callable("make_scenarios_comparison_Taylor_Diagram")(
                                    casedir, evaluation_item, stds, RMSs, cors, ref_source, sim_sources, option
                                )
                            except (ValueError, RuntimeError, IOError, OSError) as e:
                                logging.error(
                                    f"Error: {evaluation_item} {ref_source} Taylor diagram generation failed: {e}"
                                )
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()
