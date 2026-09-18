"""Portrait Plot seasonal comparison scenario."""

from __future__ import annotations

import gc
import logging
import os
import sys

import numpy as np
import xarray as xr
from joblib import delayed

from openbench.core._comparison_helpers import (
    _apply_pairwise_valid_mask,
    _atomic_text_writer,
    _finite_reduced_value,
    _load_station_pair,
    _station_evaluation_frame,
)
from openbench.data.station_missing import StationDataUnavailable
from openbench.core._comparison_portrait_calculations import (
    process_portrait_metric,
    process_portrait_score,
)
from openbench.util.converttype import Convert_Type
from openbench.util.names import select_data_array


def _has_finite_pair(s, o) -> bool:
    return bool(np.isfinite(np.asarray(s)).any() and np.isfinite(np.asarray(o)).any())


def _select_season_pair(s, o, season: str):
    s_season = s.sel(time=s["time.season"] == season)
    o_season = o.sel(time=o["time.season"] == season)
    if s_season.sizes.get("time", 0) == 0 or o_season.sizes.get("time", 0) == 0:
        raise StationDataUnavailable(f"No {season} station data")
    s_season, o_season = _apply_pairwise_valid_mask(s_season, o_season)
    if not _has_finite_pair(s_season, o_season):
        raise StationDataUnavailable(f"No valid {season} station pairs")
    return s_season, o_season


def _nan_or_reduced(values, *, reducer, plot, item, ref_source, sim_source, variable):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.nan
    return _finite_reduced_value(
        values,
        reducer=reducer,
        plot=plot,
        item=item,
        ref_source=ref_source,
        sim_source=sim_source,
        variable=variable,
    )


def _format_portrait_value(value):
    return "nan" if not np.isfinite(value) else f"{value:.2f}"


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

                                            station_list = _station_evaluation_frame(
                                                basedir, evaluation_item, ref_source, sim_source, kind="metrics"
                                            )

                                            def _process_station_data_parallel(
                                                casedir,
                                                ref_source,
                                                sim_source,
                                                item,
                                                sim_varname,
                                                ref_varname,
                                                station_row,
                                                metric_or_score,
                                                season,
                                                metric=None,
                                                score=None,
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
                                                    s_season, o_season = _select_season_pair(s, o, season)
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
                                                            allow_empty=True,
                                                        )
                                                    if metric_or_score == "score":
                                                        return process_portrait_score(
                                                            self,
                                                            casedir,
                                                            item,
                                                            ref_source,
                                                            sim_source,
                                                            score,
                                                            s_season,
                                                            o_season,
                                                            allow_empty=True,
                                                        )
                                                    raise ValueError(
                                                        f"Unsupported portrait statistic kind: {metric_or_score}"
                                                    )
                                                except StationDataUnavailable as e:
                                                    logging.debug(
                                                        "Station %s has no Portrait seasonal value for %s/%s: %s",
                                                        station_row.get("ID", "<unknown>"),
                                                        metric or score,
                                                        season,
                                                        e,
                                                    )
                                                    return np.nan

                                            station_rows = [row for _, row in station_list.iterrows()]
                                            seasons = ["DJF", "MAM", "JJA", "SON"]
                                            for metric in metrics:
                                                try:
                                                    if not hasattr(self, metric):
                                                        raise ValueError(f"No such metric: {metric}")
                                                    for season in seasons:
                                                        results = self._run_parallel_or_serial(
                                                            delayed(_process_station_data_parallel)(
                                                                basedir,
                                                                ref_source,
                                                                sim_source,
                                                                evaluation_item,
                                                                sim_varname,
                                                                ref_varname,
                                                                station_row,
                                                                "metric",
                                                                season,
                                                                metric=metric,
                                                            )
                                                            for station_row in station_rows
                                                        )
                                                        results = np.asarray(results, dtype=float)
                                                        finite = results[np.isfinite(results)]
                                                        if finite.size > 2:
                                                            q1, q3 = np.percentile(finite, [5, 95])
                                                            results = np.where(
                                                                (results >= q1) & (results <= q3), results, np.nan
                                                            )

                                                        mean_value = _nan_or_reduced(
                                                            results,
                                                            reducer="median",
                                                            plot="Portrait Plot seasonal",
                                                            item=evaluation_item,
                                                            ref_source=ref_source,
                                                            sim_source=sim_source,
                                                            variable=f"{metric}_{season}",
                                                        )
                                                        output_file.write(f"{_format_portrait_value(mean_value)}\t")
                                                finally:
                                                    gc.collect()  # Clean up memory after processing each metric

                                            for score in scores:
                                                try:
                                                    if not hasattr(self, score):
                                                        raise ValueError(f"No such score: {score}")
                                                    for season in seasons:
                                                        results = self._run_parallel_or_serial(
                                                            delayed(_process_station_data_parallel)(
                                                                basedir,
                                                                ref_source,
                                                                sim_source,
                                                                evaluation_item,
                                                                sim_varname,
                                                                ref_varname,
                                                                station_row,
                                                                "score",
                                                                season,
                                                                score=score,
                                                            )
                                                            for station_row in station_rows
                                                        )
                                                        mean_value = _nan_or_reduced(
                                                            results,
                                                            reducer="mean",
                                                            plot="Portrait Plot seasonal",
                                                            item=evaluation_item,
                                                            ref_source=ref_source,
                                                            sim_source=sim_source,
                                                            variable=f"{score}_{season}",
                                                        )
                                                        output_file.write(f"{_format_portrait_value(mean_value)}\t")
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
                                                    o = select_data_array(ref_ds, ref_varname).load()
                                                with xr.open_dataset(sim_path) as sim_ds:
                                                    s = select_data_array(sim_ds, sim_varname).load()
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
