"""Relative Score comparison scenario."""

from __future__ import annotations

import gc
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from openbench.core._comparison_helpers import (
    _comparison_sim_groups,
    _station_evaluation_frame,
    _station_frames_aligned_by_id,
    _write_csv_atomic,
)
from openbench.util.converttype import Convert_Type
from openbench.util.filenames import relative_grid_score_filename, relative_station_scores_filename
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


class RelativeScoreComparisonMixin:
    def scenarios_Relative_Score_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "Relative_Score")
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
                            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                            if isinstance(sim_sources, str):
                                sim_sources = [sim_sources]
                            sim_groups = _comparison_sim_groups(
                                evaluation_item, sim_sources, ref_source, sim_nml, ref_nml
                            )

                            station_sources = sim_groups.get("stn", [])
                            for sim_source in station_sources:
                                try:
                                    frames = _station_frames_aligned_by_id(
                                        {
                                            source: _station_evaluation_frame(
                                                casedir, evaluation_item, ref_source, source, kind="scores"
                                            )
                                            for source in station_sources
                                        }
                                    )
                                    base = frames[sim_source]
                                    combined_relative_scores = pd.DataFrame({"ID": base["ID"]})
                                    coord_cols = [
                                        col
                                        for col in ("ref_lon", "ref_lat", "sim_lon", "sim_lat")
                                        if col in base.columns
                                    ]
                                    for col in coord_cols:
                                        combined_relative_scores[col] = base[col]

                                    for score in scores:
                                        try:
                                            score_frames = _station_frames_aligned_by_id(frames, score)
                                            values = pd.DataFrame(
                                                {source: score_frames[source][score] for source in station_sources}
                                            ).where(lambda x: np.isfinite(x))
                                            n_models = values.count(axis=1)
                                            score_mean = values.mean(axis=1, skipna=True)
                                            score_std = values.std(axis=1, skipna=True)
                                            relative_scores = ((values[sim_source] - score_mean) / score_std).where(
                                                (n_models >= 2) & (score_std > 0)
                                            )

                                            value_col = f"relative_{score}_{sim_source}"
                                            status_col = f"status_{value_col}"
                                            reason_col = f"reason_{value_col}"
                                            combined_relative_scores[value_col] = relative_scores.astype("float32")
                                            own_missing_reason = (
                                                score_frames[sim_source]
                                                .get("reason", pd.Series("", index=relative_scores.index))
                                                .fillna("")
                                                .replace("", "no finite station evaluation")
                                            )
                                            reasons = pd.Series("", index=relative_scores.index, dtype="object")
                                            reasons = reasons.mask(values[sim_source].isna(), own_missing_reason)
                                            reasons = reasons.mask(
                                                values[sim_source].notna() & (n_models < 2),
                                                "fewer than two finite model evaluations",
                                            )
                                            reasons = reasons.mask(
                                                values[sim_source].notna() & (n_models >= 2) & ~(score_std > 0),
                                                "zero across-model variance",
                                            )
                                            combined_relative_scores[status_col] = np.where(
                                                relative_scores.notna(), "ok", "unavailable"
                                            )
                                            combined_relative_scores[reason_col] = reasons.where(
                                                relative_scores.isna(), ""
                                            )
                                        finally:
                                            gc.collect()  # Clean up memory after processing each score

                                    value_columns = [
                                        column
                                        for column in combined_relative_scores.columns
                                        if column.startswith("relative_")
                                        and not column.endswith(("_status", "_reason"))
                                    ]
                                    status_columns = [f"status_{column}" for column in value_columns]
                                    reason_columns = [f"reason_{column}" for column in value_columns]
                                    valid_counts = (
                                        combined_relative_scores[status_columns].eq("ok").sum(axis=1)
                                        if status_columns
                                        else pd.Series(0, index=combined_relative_scores.index)
                                    )
                                    combined_relative_scores["status"] = np.select(
                                        [valid_counts == len(status_columns), valid_counts > 0],
                                        ["ok", "partial"],
                                        default="unavailable",
                                    )
                                    combined_relative_scores["reason"] = ""
                                    if reason_columns:
                                        combined_relative_scores.loc[
                                            combined_relative_scores["status"] == "unavailable", "reason"
                                        ] = combined_relative_scores.loc[
                                            combined_relative_scores["status"] == "unavailable", reason_columns
                                        ].agg("; ".join, axis=1)
                                    combined_relative_scores = Convert_Type.convert_Frame(combined_relative_scores)
                                    output_path = os.path.join(
                                        dir_path,
                                        relative_station_scores_filename(evaluation_item, ref_source, sim_source),
                                    )
                                    _write_csv_atomic(combined_relative_scores, output_path, index=False)

                                    try:
                                        _comparison_callable("make_scenarios_comparison_Relative_Score")(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_source,
                                            scores,
                                            "stn",
                                            self.main_nml["general"],
                                            option,
                                        )
                                    except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
                                        logging.error(f"Error creating relative score plot: {e}")
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each simulation source

                            grid_sources = sim_groups.get("grid", [])
                            for sim_source in grid_sources:
                                try:
                                    for score in scores:
                                        try:
                                            datasets = []
                                            for source in grid_sources:
                                                file = os.path.join(
                                                    casedir,
                                                    "scores",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{source}_{score}.nc",
                                                )
                                                with xr.open_dataset(file) as ds_file:
                                                    ds = Convert_Type.convert_nc(ds_file.load())
                                                datasets.append(ds)

                                            combined_ds = xr.concat(datasets, dim="file")

                                            score_mean = (
                                                combined_ds[score].mean(dim="file", skipna=True).astype("float32")
                                            )
                                            score_std = (
                                                combined_ds[score].std(dim="file", skipna=True).astype("float32")
                                            )

                                            file = os.path.join(
                                                casedir,
                                                "scores",
                                                f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                                            )
                                            with xr.open_dataset(file) as ds_file:
                                                ds = Convert_Type.convert_nc(ds_file.load())
                                            relative_score = xr.where(
                                                (len(grid_sources) >= 2) & (score_std != 0),
                                                (ds[score] - score_mean) / score_std,
                                                np.nan,
                                            )
                                            relative_score = relative_score.where(np.isfinite(relative_score), np.nan)

                                            result_ds = xr.Dataset()
                                            result_ds[f"relative_{score}"] = Convert_Type.convert_nc(relative_score)

                                            output_file = os.path.join(
                                                dir_path,
                                                relative_grid_score_filename(
                                                    evaluation_item, ref_source, sim_source, score
                                                ),
                                            )
                                            _write_netcdf_atomic(result_ds, output_file)
                                        finally:
                                            gc.collect()  # Clean up memory after processing each score

                                    try:
                                        _comparison_callable("make_scenarios_comparison_Relative_Score")(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_source,
                                            scores,
                                            "grid",
                                            self.main_nml["general"],
                                            option,
                                        )
                                    except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
                                        logging.error(f"Error creating relative score plot: {e}")
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each simulation source
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method
