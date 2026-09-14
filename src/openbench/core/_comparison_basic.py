"""Basic comparison scenarios."""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from joblib import delayed

from openbench.core._comparison_helpers import (
    _load_station_pair,
    _station_evaluation_frame,
    _require_stat_method,
    _write_csv_atomic,
)
from openbench.data.station_missing import StationDataUnavailable
from openbench.util.converttype import Convert_Type


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class BasicComparisonMixin:
    def scenarios_Basic_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """
        Calculate all the data (including input data,metrics,scores):
        1. Calculate ensemble mean, median, min, max
        2. Calculate sum value for each input
        4. Plot the results
        """
        basic_method = option["key"]
        dir_path = os.path.join(f"{basedir}", "comparisons", basic_method)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        def calculate_basic_parallel(
            station_list, iik, evaluation_item, ref_source, sim_source, ref_varname, sim_varname
        ):
            try:
                s, o = _load_station_pair(
                    self,
                    basedir,
                    evaluation_item,
                    ref_source,
                    sim_source,
                    station_list.iloc[iik],
                    ref_varname,
                    sim_varname,
                )
            except StationDataUnavailable as exc:
                return {"ref_value": np.nan, "sim_value": np.nan, "status": "unavailable", "reason": str(exc)}
            method = _require_stat_method(self, basic_method)
            ref_value, sim_value = float(method(o)), float(method(s))
            ref_valid, sim_valid = np.isfinite(ref_value), np.isfinite(sim_value)
            available = ref_valid and sim_valid
            reason = f"{basic_method} is undefined for available samples"
            return {
                "ref_value": ref_value if ref_valid else np.nan,
                "sim_value": sim_value if sim_valid else np.nan,
                "status": "ok" if available else "partial" if ref_valid or sim_valid else "unavailable",
                "reason": "" if available else reason,
                "status_ref_value": "ok" if ref_valid else "unavailable",
                "status_sim_value": "ok" if sim_valid else "unavailable",
                "reason_ref_value": "" if ref_valid else reason,
                "reason_sim_value": "" if sim_valid else reason,
            }

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]

                station_sources = [
                    source
                    for source in sim_sources
                    if ref_data_type == "stn" or sim_nml[evaluation_item][f"{source}_data_type"] == "stn"
                ]
                grid_sources = [source for source in sim_sources if source not in station_sources]
                if station_sources:
                    for sim_source in station_sources:
                        try:
                            station_list = _station_evaluation_frame(basedir, evaluation_item, ref_source, sim_source)
                            metadata = [
                                "ID",
                                "sim_lat",
                                "sim_lon",
                                "ref_lon",
                                "ref_lat",
                                "use_syear",
                                "use_eyear",
                                "status",
                                "reason",
                            ]
                            station_list = station_list[[col for col in metadata if col in station_list]]

                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            results = self._run_parallel_or_serial(
                                delayed(calculate_basic_parallel)(
                                    station_list, iik, evaluation_item, ref_source, sim_source, ref_varname, sim_varname
                                )
                                for iik in range(len(station_list["ID"]))
                            )
                            basic_data = pd.concat(
                                [
                                    station_list.drop(columns=["status", "reason"], errors="ignore"),
                                    pd.DataFrame(results),
                                ],
                                axis=1,
                            )
                            basic_data = Convert_Type.convert_Frame(basic_data)
                            output_path = (
                                f"{dir_path}/{evaluation_item}_stn_{ref_source}_{sim_source}_{basic_method}.csv"
                            )
                            logging.info(f"Saving evaluation to {output_path}")
                            _write_csv_atomic(basic_data, output_path, index=False)
                            _comparison_callable("make_stn_plot_index")(
                                output_path, basic_method, self.main_nml["general"], (ref_source, sim_source), option
                            )
                        except Exception as e:
                            logging.error(f"Error processing station {basic_method} calculations for {ref_source}: {e}")
                            raise
                if grid_sources:
                    # In per_pair mode the runner writes one masked ref
                    # file per (ref, sim) pair so Basic comparison must
                    # read and write the pair-specific ref result. In
                    # intersection/strict mode all sims share one ref file,
                    # preserving the historical single-output behavior.
                    ref_sim_sources = grid_sources if self.time_alignment == "per_pair" else [None]
                    for ref_sim_source in ref_sim_sources:
                        try:
                            ref_path = self._ref_data_path(
                                basedir,
                                evaluation_item,
                                ref_source,
                                ref_varname,
                                ref_sim_source,
                            )
                            with xr.open_dataset(ref_path) as ds_file:
                                ds = ds_file[f"{ref_varname}"].load()
                            ds = Convert_Type.convert_nc(ds)
                            method_function = _require_stat_method(self, basic_method)
                            result = method_function(*[ds])
                            if ref_sim_source is None:
                                filename = f"{evaluation_item}_ref_{ref_source}_{ref_varname}_{basic_method}.nc"
                            else:
                                filename = (
                                    f"{evaluation_item}_ref_{ref_source}_sim_{ref_sim_source}_"
                                    f"{ref_varname}_{basic_method}.nc"
                                )
                            output_path = os.path.join(dir_path, filename)
                            self.save_result(output_path, basic_method, Convert_Type.convert_nc(result))
                            # Skip global map plotting for nSpatialScore since it's constant globally
                            if basic_method != "nSpatialScore":
                                _comparison_callable("make_geo_plot_index")(
                                    output_path, basic_method, self.main_nml["general"], option
                                )
                        except Exception as e:
                            logging.error(f"Error processing Grid {basic_method} calculations for {ref_source}: {e}")
                            raise

            for sim_source in sim_sources:
                if len(sim_sources) < 2:
                    continue
                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                if sim_data_type != "stn" and any(
                    ref_nml[evaluation_item][f"{source}_data_type"] != "stn" for source in ref_sources
                ):
                    try:
                        with xr.open_dataset(
                            os.path.join(basedir, "data", f"{evaluation_item}_sim_{sim_source}_{sim_varname}.nc")
                        ) as ds_file:
                            ds = ds_file[f"{sim_varname}"].load()
                        ds = Convert_Type.convert_nc(ds)
                        method_function = _require_stat_method(self, basic_method)
                        result = method_function(*[ds])
                        output_path = os.path.join(
                            dir_path, f"{evaluation_item}_sim_{sim_source}_{sim_varname}_{basic_method}.nc"
                        )
                        self.save_result(output_path, basic_method, Convert_Type.convert_nc(result))
                        # Skip global map plotting for nSpatialScore since it's constant globally
                        if basic_method != "nSpatialScore":
                            _comparison_callable("make_geo_plot_index")(
                                output_path, basic_method, self.main_nml["general"], option
                            )
                    except Exception as e:
                        logging.error(f"Error processing station {basic_method} calculations for {sim_source}: {e}")
                        raise
