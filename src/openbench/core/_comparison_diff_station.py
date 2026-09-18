"""Station-data calculations for Diff Plot comparison."""

from __future__ import annotations

from itertools import combinations
import os

import numpy as np
import pandas as pd

from openbench.core._comparison_helpers import (
    _station_evaluation_frame,
    _station_frames_aligned_by_id,
    _write_csv_atomic,
)
from openbench.util.filenames import diff_station_anomaly_filename, diff_station_difference_filename


def process_station_diff_plot(
    *,
    basedir: str,
    dir_path: str,
    sim_nml: dict,
    evaluation_item: str,
    ref_source: str,
    sim_sources: list[str],
    metrics: list[str],
    scores: list[str],
) -> None:
    for kind, variables in (("metrics", metrics), ("scores", scores)):
        for variable in variables:
            frames = _station_frames_aligned_by_id(
                {
                    source: _station_evaluation_frame(basedir, evaluation_item, ref_source, source, kind)
                    for source in sim_sources
                },
                variable,
            )
            values = pd.DataFrame({source: frames[source][variable] for source in sim_sources})
            values = values.where(np.isfinite(values))
            base = frames[sim_sources[0]]
            n_models = values.count(axis=1)
            mean = values.mean(axis=1)

            def missing_reason(frame):
                return (
                    frame.get("reason", pd.Series("", index=frame.index))
                    .fillna("")
                    .replace("", "no finite station evaluation")
                )

            missing_causes = pd.concat(
                [source + ": " + missing_reason(frames[source]) for source in sim_sources], axis=1
            ).agg("; ".join, axis=1)
            _write_csv_atomic(
                pd.DataFrame(
                    {
                        "ID": base.ID,
                        f"{variable}_ensemble_mean": mean,
                        "n_models": n_models,
                        "status": np.where(n_models > 0, "ok", "unavailable"),
                        "reason": missing_causes.where(n_models == 0, ""),
                    }
                ),
                os.path.join(dir_path, f"{evaluation_item}_stn_{ref_source}_ensemble_mean_{variable}.csv"),
                index=False,
            )

            def output_frame(frame, data, column, reason):
                lat = frame["ref_lat"] if "ref_lat" in frame else frame["sim_lat"]
                lon = frame["ref_lon"] if "ref_lon" in frame else frame["sim_lon"]
                return pd.DataFrame(
                    {
                        "ID": frame.ID,
                        "lat": lat,
                        "lon": lon,
                        column: data,
                        "status": np.where(data.notna(), "ok", "unavailable"),
                        "reason": reason.where(data.isna(), ""),
                    }
                )

            for source in sim_sources:
                frame = frames[source]
                # A lone available model has no meaningful inter-model anomaly.
                anomaly = (values[source] - mean).where(n_models >= 2)
                reason = missing_reason(frame).where(values[source].isna(), "fewer than two finite model evaluations")
                output = output_frame(frame, anomaly, f"{variable}_anomaly", reason)
                output["n_models"] = n_models
                _write_csv_atomic(
                    output,
                    os.path.join(
                        dir_path, diff_station_anomaly_filename(evaluation_item, ref_source, source, variable)
                    ),
                    index=False,
                )

            for left, right in combinations(sim_sources, 2):
                difference = values[left] - values[right]
                reason = pd.Series("", index=base.index)
                for source in (left, right):
                    unavailable = values[source].isna()
                    detail = source + ": " + missing_reason(frames[source])
                    reason = reason + detail.where(unavailable, "") + np.where(unavailable, "; ", "")
                output = output_frame(base, difference, f"{variable}_diff", reason.str.rstrip("; "))
                _write_csv_atomic(
                    output,
                    os.path.join(
                        dir_path,
                        diff_station_difference_filename(
                            evaluation_item,
                            ref_source,
                            left,
                            sim_nml[evaluation_item][f"{left}_varname"],
                            right,
                            sim_nml[evaluation_item][f"{right}_varname"],
                            variable,
                        ),
                    ),
                    index=False,
                )
