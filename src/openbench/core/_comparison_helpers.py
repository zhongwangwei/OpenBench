# -*- coding: utf-8 -*-
"""Shared helpers for comparison processing.

Kept outside the large ``comparison.py`` orchestration class so filename,
station-alignment, and atomic-write behavior can be reviewed independently.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_file_atomic as _write_file_atomic


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
    required = {"ID", value_column}
    for label, frame in ((left_label, df1), (right_label, df2)):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{label} station file is missing required columns: {sorted(missing)}")
        duplicated = frame["ID"].duplicated()
        if duplicated.any():
            duplicate_ids = frame.loc[duplicated, "ID"].astype(str).unique().tolist()
            raise ValueError(f"{label} station file contains duplicate station IDs: {duplicate_ids}")

    left = df1.set_index("ID", drop=False)
    right = df2.set_index("ID", drop=False)
    common_ids = left.index.intersection(right.index)
    if common_ids.empty:
        raise ValueError(f"{left_label} and {right_label} have no station IDs in common")

    missing_right = left.index.difference(right.index)
    missing_left = right.index.difference(left.index)
    if not missing_right.empty or not missing_left.empty:
        raise ValueError(
            f"{left_label} and {right_label} station IDs differ; "
            f"missing from {right_label}: {list(missing_right.astype(str))}; "
            f"missing from {left_label}: {list(missing_left.astype(str))}"
        )

    aligned_left = left.loc[common_ids].reset_index(drop=True)
    aligned_right = right.loc[common_ids].reset_index(drop=True)
    diff = aligned_left[value_column].reset_index(drop=True) - aligned_right[value_column].reset_index(drop=True)
    return aligned_left, diff


def _station_frames_aligned_by_id(frames: dict[str, pd.DataFrame], value_column: str) -> dict[str, pd.DataFrame]:
    """Return station frames in first-frame ID order, failing on implicit row-order assumptions."""
    if not frames:
        return {}

    labels = list(frames)
    first_label = labels[0]
    first = frames[first_label]
    required = {"ID", value_column}
    for label, frame in frames.items():
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{label} station file is missing required columns: {sorted(missing)}")
        duplicated = frame["ID"].duplicated()
        if duplicated.any():
            duplicate_ids = frame.loc[duplicated, "ID"].astype(str).unique().tolist()
            raise ValueError(f"{label} station file contains duplicate station IDs: {duplicate_ids}")

    first_ids = pd.Index(first["ID"])
    first_set = set(first_ids)
    for label in labels[1:]:
        ids = pd.Index(frames[label]["ID"])
        missing_from_current = first_ids.difference(ids)
        missing_from_first = ids.difference(first_ids)
        if missing_from_current.size or missing_from_first.size:
            raise ValueError(
                f"{first_label} and {label} station IDs differ; "
                f"missing from {label}: {list(missing_from_current.astype(str))}; "
                f"missing from {first_label}: {list(missing_from_first.astype(str))}"
            )
        if set(ids) != first_set:
            raise ValueError(f"{first_label} and {label} station IDs differ")

    return {
        label: frame.set_index("ID", drop=False).loc[first_ids].reset_index(drop=True)
        for label, frame in frames.items()
    }


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
            o = Convert_Type.convert_nc(o_file[f"{ref_varname}"].load())

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
    valid = s.notnull() & o.notnull()
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
