# -*- coding: utf-8 -*-
"""Shared helpers for comparison processing.

Kept outside the large ``comparison.py`` orchestration class so filename,
station-alignment, and atomic-write behavior can be reviewed independently.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.station_missing import StationDataUnavailable
from openbench.data.time_utils import align_station_times
from openbench.util.names import select_data_array
from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_file_atomic as _write_file_atomic


_STATION_STATISTIC_COLUMNS = {
    "Standard_Deviation": ("ref_value", "sim_value"),
    "Mann_Kendall_Trend_Test": ("ref_tau", "sim_tau", "ref_trend", "sim_trend"),
    "Functional_Response": ("functional_response_score",),
    "Correlation": ("Correlation",),
}


def _station_statistic_sources(columns, ref_source, sim_source):
    return tuple(
        ref_source
        if column.startswith("ref_")
        else sim_source
        if column.startswith("sim_")
        else f"{ref_source} / {sim_source}"
        for column in columns
    )


def _comparison_sim_groups(item, sim_sources, ref_source, sim_nml, ref_nml):
    """Group actual evaluated representations, not raw simulation data types."""
    ref_type = ref_nml[item][f"{ref_source}_data_type"]
    groups = {"stn": [], "grid": []}
    for source in sim_sources:
        sim_type = sim_nml[item][f"{source}_data_type"]
        groups["stn" if "stn" in (ref_type, sim_type) else "grid"].append(source)
    return {kind: sources for kind, sources in groups.items() if sources}


def _station_csv_column_mean(file_path: str, column: str, *, label: str) -> float:
    """Return a station CSV column mean, failing loudly when the column is absent."""
    df = pd.read_csv(file_path, sep=",", header=0)
    df = Convert_Type.convert_Frame(df)
    if column not in df.columns:
        raise KeyError(f"{label} '{column}' not found in station file: {file_path}")
    return df[column].mean(skipna=True)


def _station_pairwise_difference_by_id(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    value_column: str,
    *,
    left_label: str,
    right_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align station evaluation rows by ID before subtracting a metric/score."""
    frames = _station_frames_aligned_by_id({left_label: df1, right_label: df2}, value_column)
    left, right = frames[left_label], frames[right_label]
    return left, left[value_column] - right[value_column]


def _station_frames_aligned_by_id(
    frames: dict[str, pd.DataFrame], value_column: str | None = None
) -> dict[str, pd.DataFrame]:
    """Align the union of station IDs, retaining absent evaluations as NaN."""
    if not frames:
        return {}

    required = {"ID"} | ({value_column} if value_column is not None else set())
    for label, frame in frames.items():
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{label} station file is missing required columns: {sorted(missing)}")
        if frame["ID"].isna().any() or frame["ID"].duplicated().any():
            raise ValueError(f"{label} station file contains missing or duplicate station IDs")

    ids = pd.Index(pd.concat([frame["ID"] for frame in frames.values()]).drop_duplicates(), name="ID")
    aligned = {label: frame.set_index("ID").reindex(ids) for label, frame in frames.items()}
    # Coalesce only station metadata, never scores or missing-data status.
    for column in ("ref_lon", "ref_lat", "sim_lon", "sim_lat", "use_syear", "use_eyear"):
        available = [frame[column] for frame in aligned.values() if column in frame]
        if available:
            metadata = pd.concat(available, axis=1).bfill(axis=1).iloc[:, 0]
            for frame in aligned.values():
                frame[column] = frame[column].combine_first(metadata) if column in frame else metadata
    return {label: frame.reset_index() for label, frame in aligned.items()}


def _station_evaluation_frame(basedir, item, ref_source, sim_source, kind="metrics"):
    """Reload full station membership, including recorded evaluation data gaps."""
    root = Path(basedir)
    filename = f"{item}_stn_{ref_source}_{sim_source}_evaluations.csv"
    path = root / kind / filename
    if not path.exists():
        alternative = root / ("scores" if kind == "metrics" else "metrics") / filename
        if alternative.exists():
            path = alternative
    frame = pd.read_csv(path, dtype={"ID": str})
    status_path = root / "data" / f"stn_{ref_source}_{sim_source}" / f"{item}_evaluation_status.csv"
    if status_path.exists():
        status = pd.read_csv(status_path, dtype={"ID": str}).fillna({"reason": ""})
        values = frame.drop(columns=[col for col in status if col != "ID" and col in frame])
        frame = status.merge(values, on="ID", how="left", validate="one_to_one")
        # Never revive stale values from an earlier successful evaluation.
        metadata = {"ID", "sim_lat", "sim_lon", "ref_lat", "ref_lon", "use_syear", "use_eyear"}
        columns = [col for col in values if col not in metadata]
        frame.loc[frame["status"] == "unavailable", columns] = np.nan
    return Convert_Type.convert_Frame(frame)


def _load_station_pair(handler, basedir, item, ref_source, sim_source, row, ref_varname, sim_varname):
    """Load a station pair; only recorded or proven data gaps are recoverable."""
    if row.get("status") == "unavailable":
        raise StationDataUnavailable(str(row.get("reason") or "station data unavailable"))
    arrays = []
    for role, varname in (("sim", sim_varname), ("ref", ref_varname)):
        path = (
            Path(basedir)
            / "data"
            / f"stn_{ref_source}_{sim_source}"
            / (f"{item}_{role}_{row['ID']}_{int(row['use_syear'])}_{int(row['use_eyear'])}.nc")
        )
        if not path.exists() and path.with_suffix(".skip.txt").exists():
            raise StationDataUnavailable(path.with_suffix(".skip.txt").read_text(encoding="utf-8"))
        with xr.open_dataset(path) as ds:
            data = select_data_array(ds, varname).load()
        data = data.squeeze([dim for dim in data.dims if dim != "time" and data.sizes[dim] == 1])
        arrays.append(Convert_Type.convert_nc(data))
    s, o = align_station_times(*arrays, row["ID"], getattr(handler, "compare_tim_res", ""))
    s, o = _apply_pairwise_valid_mask(s, o)
    if not bool(np.isfinite(s).any()):
        raise StationDataUnavailable("no shared finite sim/ref pairs")
    return s, o


def _grid_score_mean(
    handler,
    casedir: str,
    evaluation_item: str,
    ref_source: str,
    sim_source: str,
    ref_varname: str,
    score: str,
):
    """Return a gridded score mean, failing loudly when required NC output is absent or malformed."""
    score_path = f"{casedir}/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
    with xr.open_dataset(score_path) as ds_file:
        ds = Convert_Type.convert_nc(ds_file.load())

    weight = getattr(handler, "weight", "none").lower()
    if weight == "area":
        weights = np.cos(np.deg2rad(ds.lat))
        return ds[score].weighted(weights).mean(skipna=True).values
    if weight == "mass":
        ref_path = handler._ref_data_path(casedir, evaluation_item, ref_source, ref_varname, sim_source)
        with xr.open_dataset(ref_path) as o_file:
            o = Convert_Type.convert_nc(select_data_array(o_file, ref_varname, evaluation_item).load())

        area_weights = np.cos(np.deg2rad(ds.lat))
        flux_weights = np.abs(o.mean("time"))
        combined_weights = area_weights * flux_weights
        normalized_weights = combined_weights / combined_weights.sum()
        return ds[score].weighted(normalized_weights.fillna(0)).mean(skipna=True).values
    return ds[score].mean(skipna=True).values


def _require_station_diagram_results(
    results: list[dict], *, diagram: str, item: str, ref_source: str, sim_source: str
) -> None:
    """Fail clearly when all station-level diagram inputs were skipped."""
    if any(result for result in results):
        return
    raise FileNotFoundError(
        f"{diagram}: no usable station data for {item}/{ref_source}/{sim_source}; "
        "all listed stations were skipped because required per-station input files or variables were missing"
    )


def _finite_distribution_values(data, *, plot: str, item: str, ref_source: str, sim_source: str, variable: str):
    """Return finite distribution values, failing before plotting empty series."""
    values = np.asarray(data)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{plot}: no finite data for {item}/{ref_source}/{sim_source}/{variable}")
    return values


def _finite_reduced_value(
    data,
    *,
    reducer: str,
    plot: str,
    item: str,
    ref_source: str,
    sim_source: str,
    variable: str,
) -> float:
    """Reduce finite values, failing before a comparison silently drops the requested variable."""
    values = _finite_distribution_values(
        data,
        plot=plot,
        item=item,
        ref_source=ref_source,
        sim_source=sim_source,
        variable=variable,
    )
    if reducer == "mean":
        return float(np.nanmean(values))
    if reducer == "median":
        return float(np.nanmedian(values))
    raise ValueError(f"Unsupported finite reducer: {reducer}")


def _apply_pairwise_valid_mask(s: xr.DataArray, o: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Mask sim/ref arrays without in-place NaN assignment into possibly integer arrays."""
    valid = np.isfinite(s) & np.isfinite(o)
    return s.where(valid), o.where(valid)


def _require_stat_method(handler, statistic: str):
    """Return a statistics method or fail with a clear configuration error."""
    method_name = f"stat_{str(statistic).lower()}"
    method = getattr(handler, method_name, None)
    if method is None or not callable(method):
        raise AttributeError(f"Statistics method {method_name!r} is not available")
    return method


def _write_csv_atomic(dataframe: pd.DataFrame, output_path: str, **kwargs) -> None:
    """Write a CSV via same-directory temp file to avoid exposing partial outputs."""
    _write_file_atomic(output_path, lambda tmp_path: dataframe.to_csv(tmp_path, **kwargs), suffix=".tmp.csv")


@contextmanager
def _atomic_text_writer(output_path: str, *, suffix: str = ".tmp.csv"):
    """Yield a text handle and atomically replace ``output_path`` when complete."""
    target = os.fspath(output_path)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{os.path.basename(target)}.", suffix=suffix, dir=parent)
    os.close(fd)
    try:
        with open(tmp_name, "w") as handle:
            yield handle
        os.replace(tmp_name, target)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            logging.debug("Could not remove temporary output file: %s", tmp_name)


__all__ = [
    "_station_csv_column_mean",
    "_station_pairwise_difference_by_id",
    "_station_frames_aligned_by_id",
    "_grid_score_mean",
    "_require_station_diagram_results",
    "_finite_distribution_values",
    "_finite_reduced_value",
    "_apply_pairwise_valid_mask",
    "_require_stat_method",
    "_write_csv_atomic",
    "_atomic_text_writer",
]
