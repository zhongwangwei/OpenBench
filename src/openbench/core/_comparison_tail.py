"""Tail-end comparison scenarios split out of the main comparison module."""

from __future__ import annotations

import csv
import gc
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from openbench.core._comparison_helpers import (
    _atomic_text_writer,
    _STATION_STATISTIC_COLUMNS,
    _station_statistic_sources,
    _grid_score_mean,
    _load_station_pair,
    _require_stat_method,
    _station_evaluation_frame,
    _write_csv_atomic,
)
from openbench.data.station_missing import StationDataUnavailable
from openbench.util.converttype import Convert_Type
from openbench.util.names import select_data_array


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


_MK_COLUMNS = ("trend", "significance", "p_value", "tau", "s_statistic", "z_score", "sen_slope")


def _scalar(value) -> float:
    array = value.to_array().values if isinstance(value, xr.Dataset) else getattr(value, "values", value)
    return float(np.asarray(array).squeeze())


def _station_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["ID", "sim_lat", "sim_lon", "ref_lon", "ref_lat", "use_syear", "use_eyear", "status", "reason"]
    return frame[[column for column in columns if column in frame]].copy()


def _station_tail_path(dir_path: str, method_name: str, item: str, ref_source: str, sim_source: str) -> str:
    return os.path.join(dir_path, f"{method_name}_{item}_stn_{ref_source}_{sim_source}.csv")


def _load_station_rows(basedir: str, item: str, ref_source: str, sim_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _station_evaluation_frame(basedir, item, ref_source, sim_source)
    output = _station_metadata(frame)
    output["status"] = output["status"].fillna("ok") if "status" in output else "ok"
    output["reason"] = output["reason"].fillna("") if "reason" in output else ""
    return frame, output


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
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]
            grid_ref_sources = [
                ref_source
                for ref_source in ref_sources
                if ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"] != "stn"
            ]
            grid_sim_sources = [
                sim_source
                for sim_source in sim_sources
                if sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"] != "stn"
            ]

            for sim_source in sim_sources:
                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                if sim_data_type == "stn" or not grid_ref_sources:
                    logging.debug(
                        "%s simulation source %s is written in pair-specific station CSVs",
                        method_name,
                        sim_source,
                    )
                else:
                    try:
                        sim_path = os.path.join(basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc")
                        with xr.open_dataset(sim_path) as sim_ds:
                            sim = select_data_array(sim_ds, sim_varname, evaluation_item).load()
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
            for ref_source in ref_sources:
                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                if ref_data_type == "stn" or not grid_sim_sources:
                    logging.debug(
                        "%s reference source %s is written in pair-specific station CSVs",
                        method_name,
                        ref_source,
                    )
                elif self.time_alignment == "per_pair" and getattr(self, "unified_mask", True):
                    for sim_source in grid_sim_sources:
                        try:
                            ref_path = self._ref_data_path(
                                basedir, evaluation_item, ref_source, ref_varname, sim_source
                            )
                            with xr.open_dataset(ref_path) as ref_ds:
                                ref = select_data_array(ref_ds, ref_varname, evaluation_item).load()
                            ref = Convert_Type.convert_nc(ref)
                            result = method_function(*[ref])
                            output_file = os.path.join(
                                dir_path,
                                f"Mann_Kendall_Trend_Test_{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{ref_varname}.nc",
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
                    try:
                        ref_path = self._ref_data_path(basedir, evaluation_item, ref_source, ref_varname)
                        with xr.open_dataset(ref_path) as ref_ds:
                            ref = select_data_array(ref_ds, ref_varname, evaluation_item).load()
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
                for sim_source in sim_sources:
                    sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                    if ref_data_type == "stn" or sim_data_type == "stn":
                        sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                        station_list, output = _load_station_rows(basedir, evaluation_item, ref_source, sim_source)
                        for prefix in ("ref", "sim"):
                            for column in _MK_COLUMNS:
                                output[f"{prefix}_{column}"] = np.nan
                        for index, row in station_list.iterrows():
                            try:
                                s, o = _load_station_pair(
                                    self,
                                    basedir,
                                    evaluation_item,
                                    ref_source,
                                    sim_source,
                                    row,
                                    ref_varname,
                                    sim_varname,
                                )
                                ref_result = method_function(o)
                                sim_result = method_function(s)
                                values = {}
                                for prefix, result in (("ref", ref_result), ("sim", sim_result)):
                                    for column in _MK_COLUMNS:
                                        values[f"{prefix}_{column}"] = _scalar(result[column])
                                for column, value in values.items():
                                    finite = np.isfinite(value)
                                    output.loc[index, column] = value if finite else np.nan
                                    output.loc[index, f"status_{column}"] = "ok" if finite else "unavailable"
                                    output.loc[index, f"reason_{column}"] = (
                                        "" if finite else f"{column} is undefined for available samples"
                                    )
                                if all(np.isfinite(value) for value in values.values()):
                                    output.loc[index, ["status", "reason"]] = ["ok", ""]
                                elif any(np.isfinite(value) for value in values.values()):
                                    output.loc[index, ["status", "reason"]] = [
                                        "partial",
                                        "Some statistics are undefined",
                                    ]
                                else:
                                    output.loc[index, ["status", "reason"]] = [
                                        "unavailable",
                                        f"{method_name} is undefined for available samples",
                                    ]
                            except StationDataUnavailable as exc:
                                output.loc[index, ["status", "reason"]] = ["unavailable", str(exc)]
                        _write_csv_atomic(
                            output,
                            _station_tail_path(dir_path, method_name, evaluation_item, ref_source, sim_source),
                            index=False,
                        )
                        columns = _STATION_STATISTIC_COLUMNS[method_name]
                        _comparison_callable("make_stn_plot_index")(
                            _station_tail_path(dir_path, method_name, evaluation_item, ref_source, sim_source),
                            method_name,
                            self.main_nml["general"],
                            _station_statistic_sources(columns, ref_source, sim_source),
                            option,
                            value_columns=columns,
                        )

    def scenarios_Standard_Deviation_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            method_name = "Standard_Deviation"
            method_function = _require_stat_method(self, method_name)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            os.makedirs(dir_path, exist_ok=True)

            for evaluation_item in evaluation_items:
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]
                grid_ref_sources = [
                    ref_source
                    for ref_source in ref_sources
                    if ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"] != "stn"
                ]
                grid_sim_sources = [
                    sim_source
                    for sim_source in sim_sources
                    if sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"] != "stn"
                ]

                for sim_source in sim_sources:
                    try:
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                        if sim_data_type != "stn" and grid_ref_sources:
                            sim_path = os.path.join(
                                basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                            )
                            if not os.path.exists(sim_path):
                                raise FileNotFoundError(
                                    f"{method_name}: required simulation input is missing for "
                                    f"{evaluation_item}/{sim_source}: {sim_path}"
                                )
                            with xr.open_dataset(sim_path) as sim_ds:
                                sim = select_data_array(sim_ds, sim_varname).load()
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
                            logging.debug(
                                "%s simulation source %s is written in pair-specific station CSVs",
                                method_name,
                                sim_source,
                            )
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {sim_source}: {e}"
                        )
                        raise
                    finally:
                        gc.collect()

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]

                        if ref_data_type != "stn" and grid_sim_sources:
                            ref_sim_sources = (
                                grid_sim_sources
                                if self.time_alignment == "per_pair" and getattr(self, "unified_mask", True)
                                else [None]
                            )
                            for ref_sim_source in ref_sim_sources:
                                ref_path = self._ref_data_path(
                                    basedir, evaluation_item, ref_source, ref_varname, ref_sim_source
                                )
                                if not os.path.exists(ref_path):
                                    raise FileNotFoundError(
                                        f"{method_name}: required reference input is missing for "
                                        f"{evaluation_item}/{ref_source}: {ref_path}"
                                    )
                                with xr.open_dataset(ref_path) as ref_ds:
                                    ref = select_data_array(ref_ds, ref_varname).load()
                                ref = Convert_Type.convert_nc(ref)

                                result = method_function(*[ref])
                                suffix = (
                                    f"_sim_{ref_sim_source}_{ref_varname}.nc"
                                    if ref_sim_source
                                    else f"_{ref_varname}.nc"
                                )
                                output_file = os.path.join(
                                    dir_path, f"{method_name}_{evaluation_item}_ref_{ref_source}{suffix}"
                                )

                                self.save_result(output_file, method_name, Convert_Type.convert_nc(result))
                                _comparison_callable("make_Standard_Deviation")(
                                    output_file, method_name, ref_source, self.main_nml["general"], option
                                )
                        else:
                            logging.debug(
                                "%s reference source %s is written in pair-specific station CSVs",
                                method_name,
                                ref_source,
                            )
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {ref_source}: {e}"
                        )
                        raise
                    finally:
                        gc.collect()

                    for sim_source in sim_sources:
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        if ref_data_type != "stn" and sim_data_type != "stn":
                            continue
                        sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                        station_list, output = _load_station_rows(basedir, evaluation_item, ref_source, sim_source)
                        output["ref_value"] = np.nan
                        output["sim_value"] = np.nan
                        for index, row in station_list.iterrows():
                            try:
                                s, o = _load_station_pair(
                                    self,
                                    basedir,
                                    evaluation_item,
                                    ref_source,
                                    sim_source,
                                    row,
                                    ref_varname,
                                    sim_varname,
                                )
                                ref_value = _scalar(method_function(o))
                                sim_value = _scalar(method_function(s))
                                output.loc[index, ["ref_value", "sim_value"]] = [ref_value, sim_value]
                                if np.isfinite(ref_value) and np.isfinite(sim_value):
                                    output.loc[index, ["status", "reason"]] = ["ok", ""]
                                else:
                                    output.loc[index, ["status", "reason"]] = [
                                        "unavailable",
                                        f"{method_name} is undefined for available samples",
                                    ]
                            except StationDataUnavailable as exc:
                                output.loc[index, ["status", "reason"]] = ["unavailable", str(exc)]
                        _write_csv_atomic(
                            output,
                            _station_tail_path(dir_path, method_name, evaluation_item, ref_source, sim_source),
                            index=False,
                        )
                        columns = _STATION_STATISTIC_COLUMNS[method_name]
                        _comparison_callable("make_stn_plot_index")(
                            _station_tail_path(dir_path, method_name, evaluation_item, ref_source, sim_source),
                            method_name,
                            self.main_nml["general"],
                            _station_statistic_sources(columns, ref_source, sim_source),
                            option,
                            value_columns=columns,
                        )
        finally:
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
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                        for sim_source in sim_sources:
                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                            if ref_data_type != "stn" and sim_data_type != "stn":
                                continue
                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            station_list, output = _load_station_rows(basedir, evaluation_item, ref_source, sim_source)
                            output["functional_response_score"] = np.nan
                            for index, row in station_list.iterrows():
                                try:
                                    s, o = _load_station_pair(
                                        self,
                                        basedir,
                                        evaluation_item,
                                        ref_source,
                                        sim_source,
                                        row,
                                        ref_varname,
                                        sim_varname,
                                    )
                                    value = _scalar(method_function(o, s)["functional_response_score"])
                                    output.loc[index, "functional_response_score"] = value
                                    if np.isfinite(value):
                                        output.loc[index, ["status", "reason"]] = ["ok", ""]
                                    else:
                                        output.loc[index, ["status", "reason"]] = [
                                            "unavailable",
                                            f"{method_name} is undefined for available samples",
                                        ]
                                except StationDataUnavailable as exc:
                                    output.loc[index, ["status", "reason"]] = ["unavailable", str(exc)]
                            _write_csv_atomic(
                                output,
                                _station_tail_path(dir_path, method_name, evaluation_item, ref_source, sim_source),
                                index=False,
                            )
                            columns = _STATION_STATISTIC_COLUMNS[method_name]
                            _comparison_callable("make_stn_plot_index")(
                                _station_tail_path(dir_path, method_name, evaluation_item, ref_source, sim_source),
                                method_name,
                                self.main_nml["general"],
                                _station_statistic_sources(columns, ref_source, sim_source),
                                option,
                                value_columns=columns,
                            )

                        if ref_data_type != "stn":
                            for sim_source in sim_sources:
                                try:
                                    sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                    sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                    if sim_data_type != "stn":
                                        ref_path = self._ref_data_path(
                                            basedir, evaluation_item, ref_source, ref_varname, sim_source
                                        )
                                        with xr.open_dataset(ref_path) as ref_ds:
                                            ref = select_data_array(ref_ds, ref_varname).load()
                                        ref = Convert_Type.convert_nc(ref)
                                        sim_path = os.path.join(
                                            basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                                        )
                                        with xr.open_dataset(sim_path) as sim_ds:
                                            sim = select_data_array(sim_ds, sim_varname).load()
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
                                    gc.collect()
                    finally:
                        gc.collect()
        finally:
            gc.collect()

    def scenarios_RadarMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            if not scores:
                raise ValueError("RadarMap comparison requires at least one score")

            dir_path = os.path.join(casedir, "comparisons", "RadarMap")
            os.makedirs(dir_path, exist_ok=True)

            for score in scores:
                output_file_path = os.path.join(dir_path, f"scenarios_{score}_comparison.csv")
                with _atomic_text_writer(output_file_path) as output_file:
                    writer = csv.writer(output_file, lineterminator="\n")
                    all_sim_sources = []
                    for evaluation_item in evaluation_items:
                        sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        for s in sim_sources:
                            if s not in all_sim_sources:
                                all_sim_sources.append(s)
                    header = ["Item", "Reference"] + all_sim_sources
                    writer.writerow(header)

                    for evaluation_item in evaluation_items:
                        sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                        ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        if isinstance(ref_sources, str):
                            ref_sources = [ref_sources]

                        for ref_source in ref_sources:
                            values = []
                            for sim_source in all_sim_sources:
                                if sim_source not in sim_sources:
                                    values.append("N/A")
                                    continue

                                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                                if ref_data_type == "stn" or sim_data_type == "stn":
                                    frame = _station_evaluation_frame(
                                        casedir, evaluation_item, ref_source, sim_source, "scores"
                                    )
                                    if score not in frame.columns:
                                        raise KeyError(
                                            f"Score '{score}' not found in station file: "
                                            f"{casedir}/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                                        )
                                    overall_mean = frame[score].mean(skipna=True)
                                else:
                                    overall_mean = _grid_score_mean(
                                        self, casedir, evaluation_item, ref_source, sim_source, ref_varname, score
                                    )

                                overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"
                                values.append(overall_mean_str)
                            writer.writerow([evaluation_item, ref_source, *values])
                _comparison_callable("make_scenarios_comparison_radar_map")(output_file_path, score, option)
        finally:
            gc.collect()  # Clean up memory after processing

    def scenarios_Correlation_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            from openbench.core._comparison_helpers import _station_frames_aligned_by_id
            from openbench.data.time_utils import align_station_times

            method_name = "Correlation"
            method_function = _require_stat_method(self, method_name)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            os.makedirs(dir_path, exist_ok=True)

            def _sources(nml, item, kind):
                values = nml.get("general", {}).get(f"{item}_{kind}_source", [])
                if isinstance(values, str):
                    return [values]
                return list(values)

            def _varname(nml, item, source):
                value = nml[item].get(f"{source}_varname")
                return value or item

            def _station_path(item, ref_source, sim1, sim2):
                return os.path.join(dir_path, f"{method_name}_{item}_stn_{ref_source}_{sim1}_and_{sim2}.csv")

            def _station_frame(item, ref_source, sim_source, ref_type, sim_type):
                if ref_type != "stn" and sim_type != "stn":
                    return None
                frame, metadata = _load_station_rows(basedir, item, ref_source, sim_source)
                if metadata["ID"].isna().any() or metadata["ID"].duplicated().any():
                    raise ValueError(f"{ref_source}/{sim_source} station file contains missing or duplicate IDs")
                return frame, metadata

            def _station_output(metas):
                present = {label: meta for label, meta in metas.items() if meta is not None}
                aligned = _station_frames_aligned_by_id(present)
                output = next(iter(aligned.values())).drop(columns=["status", "reason"], errors="ignore")
                output["status"] = "unavailable"
                output["reason"] = ""
                output["Correlation"] = np.nan
                return output, aligned

            def _raw_row(source, station_id):
                if source is None:
                    return None
                rows = source[1].loc[source[1]["ID"] == station_id]
                return None if rows.empty else rows.iloc[0]

            def _site_coordinate(row):
                if row is None:
                    return None
                for lon_col, lat_col in (("ref_lon", "ref_lat"), ("sim_lon", "sim_lat")):
                    if lon_col in row and lat_col in row and pd.notna(row[lon_col]) and pd.notna(row[lat_col]):
                        lon, lat = float(row[lon_col]), float(row[lat_col])
                        if np.isfinite(lon) and np.isfinite(lat):
                            return lon, lat
                return None

            def _coordinates_differ(left, right):
                left_coord = _site_coordinate(left)
                right_coord = _site_coordinate(right)
                return left_coord is not None and right_coord is not None and left_coord != right_coord

            def _unavailable_reason(label, row, *, unsupported=False):
                if unsupported:
                    return f"{label} has no preprocessed station series for Correlation"
                if row is None:
                    return f"missing {label} station evaluation"
                if row.get("status") == "unavailable":
                    return str(row.get("reason") or f"{label} station data unavailable")
                return ""

            def _sim_series(item, ref_source, sim_source, row, ref_varname, sim_varname):
                sim, _ref = _load_station_pair(
                    self, basedir, item, ref_source, sim_source, row, ref_varname, sim_varname
                )
                return sim

            def _write_station_correlation(
                item, ref_source, ref_type, ref_varname, sim1, sim2, sim_type1, sim_type2, sim_var1, sim_var2
            ):
                left = _station_frame(item, ref_source, sim1, ref_type, sim_type1)
                right = _station_frame(item, ref_source, sim2, ref_type, sim_type2)
                metas = {sim1: left[1] if left else None, sim2: right[1] if right else None}
                output, aligned = _station_output(metas)
                for index, row in output.iterrows():
                    station_id = row["ID"]
                    row1 = _raw_row(left, station_id)
                    row2 = _raw_row(right, station_id)
                    reason1 = _unavailable_reason(sim1, row1, unsupported=left is None)
                    reason2 = _unavailable_reason(sim2, row2, unsupported=right is None)
                    if reason1 or reason2:
                        output.loc[index, ["status", "reason"]] = [
                            "unavailable",
                            "; ".join(reason for reason in (reason1, reason2) if reason),
                        ]
                        continue
                    if (
                        ref_type != "stn"
                        and sim_type1 == "stn"
                        and sim_type2 == "stn"
                        and _coordinates_differ(_raw_row(left, station_id), _raw_row(right, station_id))
                    ):
                        output.loc[index, ["status", "reason"]] = [
                            "unavailable",
                            "station coordinates differ; no common spatial support",
                        ]
                        continue
                    try:
                        data1 = _sim_series(item, ref_source, sim1, row1, ref_varname, sim_var1)
                        data2 = _sim_series(item, ref_source, sim2, row2, ref_varname, sim_var2)
                        data1, data2 = align_station_times(
                            data1, data2, station_id, getattr(self, "compare_tim_res", "")
                        )
                        value = _scalar(method_function(data1, data2)[method_name])
                        if np.isfinite(value):
                            output.loc[index, ["Correlation", "status", "reason"]] = [value, "ok", ""]
                        else:
                            output.loc[index, ["status", "reason"]] = [
                                "unavailable",
                                f"{method_name} is undefined for available samples",
                            ]
                    except StationDataUnavailable as exc:
                        output.loc[index, ["status", "reason"]] = ["unavailable", str(exc)]
                output_file = _station_path(item, ref_source, sim1, sim2)
                _write_csv_atomic(output, output_file, index=False)
                _comparison_callable("make_stn_plot_index")(
                    output_file,
                    method_name,
                    self.main_nml["general"],
                    (f"{sim1} / {sim2}",),
                    option,
                    value_columns=("Correlation",),
                )

            for evaluation_item in evaluation_items:
                sim_sources = _sources(sim_nml, evaluation_item, "sim")
                ref_sources = _sources(ref_nml, evaluation_item, "ref")
                if len(sim_sources) < 2:
                    continue
                grid_refs = [
                    ref_source
                    for ref_source in ref_sources
                    if ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"] != "stn"
                ]

                for i, sim1 in enumerate(sim_sources):
                    for sim2 in sim_sources[i + 1 :]:
                        try:
                            sim_varname1 = _varname(sim_nml, evaluation_item, sim1)
                            sim_varname2 = _varname(sim_nml, evaluation_item, sim2)
                            sim_data_type1 = sim_nml[f"{evaluation_item}"][f"{sim1}_data_type"]
                            sim_data_type2 = sim_nml[f"{evaluation_item}"][f"{sim2}_data_type"]

                            if "stn" in (sim_data_type1, sim_data_type2) and not ref_sources:
                                raise ValueError(
                                    "Station Correlation requires a reference source to identify station support"
                                )

                            if sim_data_type1 != "stn" and sim_data_type2 != "stn" and (not ref_sources or grid_refs):
                                ds1_path = os.path.join(
                                    basedir, "data", f"{evaluation_item}_sim_{sim1}_{sim_varname1}.nc"
                                )
                                ds2_path = os.path.join(
                                    basedir, "data", f"{evaluation_item}_sim_{sim2}_{sim_varname2}.nc"
                                )

                                with xr.open_dataset(ds1_path) as ds1_file:
                                    ds1 = select_data_array(ds1_file, sim_varname1, evaluation_item).load()
                                with xr.open_dataset(ds2_path) as ds2_file:
                                    ds2 = select_data_array(ds2_file, sim_varname2, evaluation_item).load()

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

                            for ref_source in ref_sources:
                                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                if ref_data_type != "stn" and sim_data_type1 != "stn" and sim_data_type2 != "stn":
                                    continue
                                ref_varname = _varname(ref_nml, evaluation_item, ref_source)
                                _write_station_correlation(
                                    evaluation_item,
                                    ref_source,
                                    ref_data_type,
                                    ref_varname,
                                    sim1,
                                    sim2,
                                    sim_data_type1,
                                    sim_data_type2,
                                    sim_varname1,
                                    sim_varname2,
                                )

                        except Exception as e:
                            logging.error(
                                f"Error processing {method_name} calculations for {evaluation_item} {sim1} and {sim2}: {e}"
                            )
                            raise
                        finally:
                            gc.collect()
        finally:
            gc.collect()
