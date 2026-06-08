"""Common comparison-processing helpers split out of comparison.py."""

from __future__ import annotations

import gc
import logging
import os
import sys

import xarray as xr
from joblib import Parallel

from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def _comparison_attr(name: str, fallback=None):
    """Resolve monkeypatch-friendly attributes from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    return fallback


class CommonComparisonMixin:
    def _run_parallel_or_serial(self, jobs_iter):
        """Execute joblib `delayed(...)` jobs in parallel or eagerly serial.

        Falls back to a plain list comprehension when ``self.num_cores == 1``.
        This:
          * mirrors the n_jobs==1 short-circuit in core.evaluation, and
          * lets tests that monkey-patch `Parallel` to PermissionError
            assert the parallel path is skipped under sequential config.

        Pass a generator/iterable of `delayed(func)(*args, **kwargs)` calls
        — `joblib.delayed` returns a `(func, args, kwargs)` tuple that we
        unpack directly when running serially.
        """
        n_jobs = getattr(self, "num_cores", -1)
        jobs = list(jobs_iter)
        if n_jobs == 1:
            return [func(*args, **kwargs) for func, args, kwargs in jobs]
        return _comparison_attr("Parallel", Parallel)(n_jobs=n_jobs)(jobs)

    def _ref_data_path(self, basedir, evaluation_item, ref_source, ref_varname, sim_source=None):
        """Build path to the preprocessed ref data file.

        In per_pair mode, each sim-ref pair has its own masked ref copy.
        In intersection/strict mode, all sims share one ref file.
        """
        if self.time_alignment == "per_pair" and sim_source:
            pair_path = os.path.join(
                basedir,
                "data",
                f"{evaluation_item}_ref_{ref_source}_{sim_source}_{ref_varname}.nc",
            )
            if os.path.exists(pair_path):
                return pair_path
        # Default: shared ref
        return os.path.join(basedir, "data", f"{evaluation_item}_ref_{ref_source}_{ref_varname}.nc")

    def scenarios_IGBP_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """Delegate IGBP/PFT implementation details to the land-cover groupby engine."""
        from openbench.core.landcover_groupby import LC_groupby

        LC_groupby(self.main_nml, self.scores, self.metrics).scenarios_IGBP_groupby_comparison(
            casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
        )

    def scenarios_PFT_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """Delegate IGBP/PFT implementation details to the land-cover groupby engine."""
        from openbench.core.landcover_groupby import LC_groupby

        LC_groupby(self.main_nml, self.scores, self.metrics).scenarios_PFT_groupby_comparison(
            casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
        )

    def save_result(self, output_file, method_name, result):
        # Remove the existing output directory
        # logging.info(f"Saving {method_name} output to {output_file}")
        try:
            if isinstance(result, xr.DataArray) or isinstance(result, xr.Dataset):
                if isinstance(result, xr.DataArray):
                    result = result.to_dataset(name=f"{method_name}")
                result["lat"].attrs["standard_name"] = "latitude"
                result["lat"].attrs["long_name"] = "latitude"
                result["lat"].attrs["units"] = "degrees_north"
                result["lat"].attrs["axis"] = "Y"
                result["lon"].attrs["standard_name"] = "longitude"
                result["lon"].attrs["long_name"] = "longitude"
                result["lon"].attrs["units"] = "degrees_east"
                result["lon"].attrs["axis"] = "X"

                # Ensure the directory exists
                output_dir = os.path.dirname(output_file)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                _write_netcdf_atomic(result, output_file)
            else:
                logging.info(f"Result of {method_name}: {result}")
        finally:
            # Clean up memory
            gc.collect()
