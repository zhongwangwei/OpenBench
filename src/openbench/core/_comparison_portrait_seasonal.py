"""Portrait Plot seasonal comparison scenario."""

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
    _finite_reduced_value,
)
from openbench.core._comparison_portrait_calculations import (
    process_portrait_metric,
    process_portrait_score,
)
from openbench.util.converttype import Convert_Type


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class PortraitSeasonalComparisonMixin:
    def scenarios_Portrait_Plot_seasonal_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Portrait_Plot_seasonal")
            os.makedirs(dir_path, exist_ok=True)

            output_file_path = os.path.join(dir_path, "Portrait_Plot_seasonal.csv")
            with _atomic_text_writer(output_file_path) as output_file:
                output_file.write("Item\t")
                output_file.write("Reference\t")
                output_file.write("Simulation\t")

                for metric in metrics:
                    output_file.write(f"{metric}_DJF\t")
                    output_file.write(f"{metric}_MAM\t")
                    output_file.write(f"{metric}_JJA\t")
                    output_file.write(f"{metric}_SON\t")

                for score in scores:
                    output_file.write(f"{score}_DJF\t")
                    output_file.write(f"{score}_MAM\t")
                    output_file.write(f"{score}_JJA\t")
                    output_file.write(f"{score}_SON\t")

                output_file.write("\n")

                for evaluation_item in evaluation_items:
                    try:
                        logging.info(f"now processing the evaluation item: {evaluation_item}")
                        sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                        ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
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

                                            stnlist = os.path.join(
                                                basedir,
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

                                            def _process_station_data_parallel(
                                                casedir,
                                                ref_source,
                                                sim_source,
                                                item,
                                                sim_varname,
                                                ref_varname,
                                                station_list,
                                                iik,
                                                metric_or_score,
                                                season,
                                                metric=None,
                                                score=None,
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

                                                    s_season = s.sel(time=s["time.season"] == season)
                                                    o_season = o.sel(time=o["time.season"] == season)

                                                    if metric_or_score == "metric":
                                                        return process_portrait_metric(
                                                            self,
                                                            casedir,
                                                            item,
                                                            ref_source,
                                                            sim_source,
                                                            metric,
                                                            s_season,
                                                            o_season,
                                                        )
                                                    elif metric_or_score == "score":
                                                        return process_portrait_score(
                                                            self,
                                                            casedir,
                                                            item,
                                                            ref_source,
                                                            sim_source,
                                                            score,
                                                            s_season,
                                                            o_season,
                                                        )
                                                except (FileNotFoundError, KeyError, ValueError, OSError) as e:
                                                    logging.debug(
                                                        f"Station {station_list['ID'][iik]} skipped in Portrait: {e}"
                                                    )
                                                    return None
                                                finally:
                                                    pass  # Memory cleanup handled at higher level

                                            seasons = ["DJF", "MAM", "JJA", "SON"]
                                            for metric in metrics:
                                                try:
                                                    for season in seasons:
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
                                                                "metric",
                                                                season,
                                                                metric=metric,
                                                            )
                                                            for iik in range(len(station_list["ID"]))
                                                        )
                                                        results = np.array(
                                                            [r if r is not None else np.nan for r in results],
                                                            dtype=float,
                                                        )
                                                        if results[~np.isnan(results)].shape[0] > 2:
                                                            q1, q3 = np.percentile(results[~np.isnan(results)], [5, 95])
                                                            results = np.where(
                                                                (results >= q1) & (results <= q3), results, np.nan
                                                            )

                                                        mean_value = _finite_reduced_value(
                                                            results,
                                                            reducer="median",
                                                            plot="Portrait Plot seasonal",
                                                            item=evaluation_item,
                                                            ref_source=ref_source,
                                                            sim_source=sim_source,
                                                            variable=f"{metric}_{season}",
                                                        )
                                                        kk_str = f"{mean_value:.2f}"
                                                        output_file.write(f"{kk_str}\t")
                                                finally:
                                                    gc.collect()  # Clean up memory after processing each metric

                                            for score in scores:
                                                try:
                                                    for season in seasons:
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
                                                                "score",
                                                                season,
                                                                score=score,
                                                            )
                                                            for iik in range(len(station_list["ID"]))
                                                        )
                                                        results_clean = np.array(
                                                            [r if r is not None else np.nan for r in results],
                                                            dtype=float,
                                                        )
                                                        mean_value = _finite_reduced_value(
                                                            results_clean,
                                                            reducer="mean",
                                                            plot="Portrait Plot seasonal",
                                                            item=evaluation_item,
                                                            ref_source=ref_source,
                                                            sim_source=sim_source,
                                                            variable=f"{score}_{season}",
                                                        )
                                                        kk_str = f"{mean_value:.2f}"
                                                        output_file.write(f"{kk_str}\t")
                                                finally:
                                                    gc.collect()  # Clean up memory after processing each score
                                        else:
                                            try:
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
                                                    o = ref_ds[ref_varname].load()
                                                with xr.open_dataset(sim_path) as sim_ds:
                                                    s = sim_ds[sim_varname].load()
                                                o = Convert_Type.convert_nc(o)
                                                s = Convert_Type.convert_nc(s)

                                                o = o.where(np.isfinite(o), np.nan)
                                                s = s.where(np.isfinite(s), np.nan)
                                                # Align time axes safely. The previous code did an unconditional
                                                # s["time"] = o["time"], which (a) raises when lengths differ and
                                                # (b) silently pairs values against wrong timestamps when lengths
                                                # match but coords are offset. Use inner-join intersect instead.
                                                if s.sizes.get("time") != o.sizes.get("time") or not np.array_equal(
                                                    s["time"].values, o["time"].values
                                                ):
                                                    s, o = xr.align(s, o, join="inner")

                                                s, o = _apply_pairwise_valid_mask(s, o)

                                                s_DJF = s.sel(time=s["time.season"] == "DJF")
                                                o_DJF = o.sel(time=o["time.season"] == "DJF")
                                                s_MAM = s.sel(time=s["time.season"] == "MAM")
                                                o_MAM = o.sel(time=o["time.season"] == "MAM")
                                                s_JJA = s.sel(time=s["time.season"] == "JJA")
                                                o_JJA = o.sel(time=o["time.season"] == "JJA")
                                                s_SON = s.sel(time=s["time.season"] == "SON")
                                                o_SON = o.sel(time=o["time.season"] == "SON")

                                                for metric in metrics:
                                                    try:
                                                        if hasattr(self, metric):
                                                            k = process_portrait_metric(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                metric,
                                                                s_DJF,
                                                                o_DJF,
                                                                vkey="_DJF",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")

                                                            k = process_portrait_metric(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                metric,
                                                                s_MAM,
                                                                o_MAM,
                                                                vkey="_MAM",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")

                                                            k = process_portrait_metric(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                metric,
                                                                s_JJA,
                                                                o_JJA,
                                                                vkey="_JJA",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")

                                                            k = process_portrait_metric(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                metric,
                                                                s_SON,
                                                                o_SON,
                                                                vkey="_SON",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")
                                                        else:
                                                            raise ValueError(f"No such metric: {metric}")
                                                    finally:
                                                        gc.collect()  # Clean up memory after processing each metric

                                                for score in scores:
                                                    try:
                                                        if hasattr(self, score):
                                                            k = process_portrait_score(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                score,
                                                                s_DJF,
                                                                o_DJF,
                                                                vkey="_DJF",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")

                                                            k = process_portrait_score(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                score,
                                                                s_MAM,
                                                                o_MAM,
                                                                vkey="_MAM",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")

                                                            k = process_portrait_score(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                score,
                                                                s_JJA,
                                                                o_JJA,
                                                                vkey="_JJA",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")

                                                            k = process_portrait_score(
                                                                self,
                                                                basedir,
                                                                evaluation_item,
                                                                ref_source,
                                                                sim_source,
                                                                score,
                                                                s_SON,
                                                                o_SON,
                                                                vkey="_SON",
                                                            )
                                                            kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                                            output_file.write(f"{kk_str}\t")
                                                        else:
                                                            raise ValueError(f"No such score: {score}")
                                                    finally:
                                                        gc.collect()  # Clean up memory after processing each score
                                            finally:
                                                gc.collect()  # Clean up memory after processing grid data
                                        output_file.write("\n")
                                    finally:
                                        pass  # Memory cleanup handled at method level
                            finally:
                                gc.collect()  # Clean up memory after processing each reference source
                    finally:
                        gc.collect()  # Clean up memory after processing each evaluation item

            _comparison_callable("make_scenarios_comparison_Portrait_Plot_seasonal")(
                output_file_path, self.casedir, evaluation_items, scores, metrics, option
            )
        finally:
            gc.collect()
