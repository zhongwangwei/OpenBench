"""Metric/score reducers for seasonal portrait comparisons."""

from __future__ import annotations

import gc
import logging
import os

import numpy as np

from openbench.core._comparison_helpers import _finite_distribution_values, _finite_reduced_value
from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def process_portrait_metric(self, casedir, item, ref_source, sim_source, metric, s, o, vkey=None):
    try:
        pb = getattr(self, metric)(s, o)
        pb = pb.where(np.isfinite(pb), np.nan)
        _finite_distribution_values(
            pb,
            plot="Portrait Plot seasonal",
            item=item,
            ref_source=ref_source,
            sim_source=sim_source,
            variable=f"{metric}{vkey or ''}",
        )
        try:
            q_value = pb.quantile([0.05, 0.95], dim=["lat", "lon"], skipna=True)
            pb = pb.where((pb >= q_value[0]) & (pb <= q_value[1]), np.nan)
        except (ValueError, RuntimeError, AttributeError) as e:
            logging.debug(f"Quantile filtering failed for {metric}: {e}")

        _finite_distribution_values(
            pb,
            plot="Portrait Plot seasonal",
            item=item,
            ref_source=ref_source,
            sim_source=sim_source,
            variable=f"{metric}{vkey or ''}",
        )

        # Only save NetCDF for gridded data (with lat/lon dimensions)
        if hasattr(pb, "dims") and "lat" in pb.dims and "lon" in pb.dims:
            try:
                pb_da = Convert_Type.convert_nc(pb)
                pb_da.name = metric
                output_path = os.path.join(
                    casedir,
                    "comparisons",
                    "Portrait_Plot_seasonal",
                    f"{item}_ref_{ref_source}_sim_{sim_source}_{metric}{vkey}.nc",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                _write_netcdf_atomic(pb_da, output_path)
            except (OSError, IOError, PermissionError, ValueError, AttributeError) as e:
                logging.debug(f"Failed to save portrait plot data for {metric}: {e}")
        return _finite_reduced_value(
            pb,
            reducer="median",
            plot="Portrait Plot seasonal",
            item=item,
            ref_source=ref_source,
            sim_source=sim_source,
            variable=f"{metric}{vkey or ''}",
        )
    finally:
        gc.collect()  # Clean up memory after processing each metric


def process_portrait_score(self, casedir, item, ref_source, sim_source, score, s, o, vkey=None):
    try:
        pb = getattr(self, score)(s, o)
        if hasattr(pb, "where"):
            pb = pb.where(np.isfinite(pb), np.nan)
        _finite_distribution_values(
            pb,
            plot="Portrait Plot seasonal",
            item=item,
            ref_source=ref_source,
            sim_source=sim_source,
            variable=f"{score}{vkey or ''}",
        )

        # Only save NetCDF for gridded data (with lat/lon dimensions)
        if hasattr(pb, "dims") and "lat" in pb.dims and "lon" in pb.dims:
            try:
                pb_da = Convert_Type.convert_nc(pb)
                pb_da.name = score
                output_path = os.path.join(
                    casedir,
                    "comparisons",
                    "Portrait_Plot_seasonal",
                    f"{item}_ref_{ref_source}_sim_{sim_source}_{score}{vkey}.nc",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                _write_netcdf_atomic(pb_da, output_path)
            except (OSError, IOError, PermissionError, ValueError, AttributeError) as e:
                logging.debug(f"Failed to save portrait plot data for {score}: {e}")

        # Apply weighting only for gridded data
        if hasattr(pb, "dims") and "lat" in pb.dims and "lon" in pb.dims:
            if self.weight.lower() == "area":
                weights = np.cos(np.deg2rad(pb.coords["lat"]))
                pb = pb.where(np.isfinite(pb), np.nan).weighted(weights).mean(skipna=True)
            elif self.weight.lower() == "mass":
                area_weights = np.cos(np.deg2rad(pb.coords["lat"]))
                flux_weights = np.abs(o.mean("time"))
                combined_weights = area_weights * flux_weights
                normalized_weights = combined_weights / combined_weights.sum()
                pb = pb.where(np.isfinite(pb), np.nan).weighted(normalized_weights.fillna(0)).mean(skipna=True)
            else:
                pb = pb.mean(skipna=True)
        else:
            # For station data, just take the mean
            pb = pb.mean(skipna=True) if hasattr(pb, "mean") else pb
        return _finite_reduced_value(
            [Convert_Type.convert_nc(pb)],
            reducer="mean",
            plot="Portrait Plot seasonal",
            item=item,
            ref_source=ref_source,
            sim_source=sim_source,
            variable=f"{score}{vkey or ''}",
        )
    finally:
        gc.collect()  # Clean up memory after processing each score
