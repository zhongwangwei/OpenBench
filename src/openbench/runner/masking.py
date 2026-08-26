"""Unified-mask preprocessing helpers for runner tasks."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Callable

logger = logging.getLogger(__name__)

NetcdfWriter = Callable[..., None]


def apply_unified_mask(
    info: dict,
    var_name: str,
    ref_source: str,
    sim_source: str,
    ref_override: str | None = None,
    *,
    write_netcdf_atomic_fn: NetcdfWriter,
    apply_spatial_mask: bool = True,
) -> None:
    """Apply unified mask: set ref to NaN wherever sim is NaN.

    This ensures evaluation metrics only cover grid cells where BOTH reference
    and simulation have valid data. Called for each sim source, so the mask
    accumulates — if any sim has NaN at a point, the ref gets NaN there too,
    ensuring consistent spatial coverage across all model comparisons.

    Only applies to grid data (station data is skipped).
    """
    import numpy as np

    casedir = info["casedir"]
    ref_varname = info.get("ref_varname", "")
    sim_varname = info.get("sim_varname", "")
    time_alignment = info.get("time_alignment", "intersection")

    if ref_override:
        ref_path = ref_override
    else:
        ref_path = os.path.join(casedir, "data", f"{var_name}_ref_{ref_source}_{ref_varname}.nc")
    sim_path = os.path.join(casedir, "data", f"{var_name}_sim_{sim_source}_{sim_varname}.nc")

    ref_path = os.path.abspath(ref_path)
    sim_path = os.path.abspath(sim_path)

    missing_paths = [path for path in (ref_path, sim_path) if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(f"Unified mask input file not found: {', '.join(missing_paths)}")

    ref_ds = None
    sim_ds = None
    staged_path = None
    try:
        import xarray as xr

        ref_ds = xr.open_dataset(ref_path, chunks="auto")
        sim_ds = xr.open_dataset(sim_path, chunks="auto")

        # The flat file stores the variable under the configured name OR the
        # relabelled evaluation item (when a fallback/convert derived it, e.g.
        # NEE from f_respc). Resolve robustly instead of hard-indexing the
        # possibly-stale config varname. (var_name is the evaluation item.)
        from openbench.util.names import select_data_array

        o = select_data_array(ref_ds, ref_varname, var_name)
        s = select_data_array(sim_ds, sim_varname, var_name)

        # Convert types if needed
        try:
            from openbench.util.converttype import Convert_Type

            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)
        except ImportError:
            pass

        # Align time dimension. In strict mode mismatches are errors. In the
        # default intersection mode, contribute the overlapping timestamps to
        # the shared mask instead of silently skipping this simulation; skipping
        # makes final masks depend on which sibling sims happened to align.
        same_length = len(s["time"]) == len(o["time"])
        same_values = same_length and np.array_equal(s["time"].values, o["time"].values)
        if not same_length:
            message = f"Unified mask: time length mismatch for {var_name} (ref={len(o['time'])}, sim={len(s['time'])})"
            if time_alignment == "strict":
                raise ValueError(message)
            logger.warning("%s, using overlapping timestamps", message)
        elif not same_values:
            message = f"Unified mask: time values mismatch for {var_name} (lengths equal but timestamps differ)"
            if time_alignment == "strict":
                raise ValueError(message)
            logger.warning("%s, using overlapping timestamps", message)

        if same_values:
            o_aligned, s_aligned = o, s
        else:
            o_aligned, s_aligned = xr.align(o, s, join="inner")
            if o_aligned.sizes.get("time", 0) == 0:
                raise ValueError(f"Unified mask: no overlapping timestamps for {var_name}")

        finite_pairs = np.isfinite(s_aligned) & np.isfinite(o_aligned)
        if apply_spatial_mask and time_alignment == "intersection" and "time" in finite_pairs.dims:
            non_time_dims = [dim for dim in finite_pairs.dims if dim != "time"]
            valid_times = finite_pairs.any(dim=non_time_dims) if non_time_dims else finite_pairs
            if hasattr(valid_times, "compute"):
                valid_times = valid_times.compute()
            if not bool(valid_times.any().item()):
                raise ValueError(f"Unified mask: no overlapping finite timestamps for {var_name}")
            o_aligned = o_aligned.where(valid_times, drop=True)
            s_aligned = s_aligned.where(valid_times, drop=True)
            finite_pairs = np.isfinite(s_aligned) & np.isfinite(o_aligned)

        # Keep the spatial mask lazy so chunked/dask-backed inputs are not
        # materialized twice before the NetCDF writer streams them.
        invalid_overlap = ~finite_pairs if apply_spatial_mask else None
        if time_alignment == "intersection":
            # Persist the exact shared time support. Repeating this for each
            # sibling simulation makes the reference time axis the global,
            # order-independent intersection used by every model.
            o_data = o_aligned.where(~invalid_overlap) if invalid_overlap is not None else o_aligned
        elif same_values:
            o_data = o.where(~invalid_overlap) if invalid_overlap is not None else o
        else:
            invalid_full = invalid_overlap.reindex_like(o, fill_value=False) if invalid_overlap is not None else None
            o_data = o.where(~invalid_full) if invalid_full is not None else o

        # Write to a sibling staging target while the lazy source datasets are
        # open, then close them before replacing the original. Windows refuses
        # to replace an open NetCDF file even when the replacement itself is atomic.
        fd, staged_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(ref_path)}.mask-",
            suffix=".nc",
            dir=os.path.dirname(ref_path),
        )
        os.close(fd)
        write_netcdf_atomic_fn(o_data, staged_path, compression=False)
        ref_ds.close()
        ref_ds = None
        sim_ds.close()
        sim_ds = None
        os.replace(staged_path, ref_path)
        staged_path = None
        logger.debug("Unified mask applied: %s (sim=%s)", var_name, sim_source)

        del o, s, o_aligned, s_aligned, invalid_overlap, o_data

    except Exception:
        logger.exception("Unified mask failed for %s (sim=%s)", var_name, sim_source)
        raise
    finally:
        if ref_ds is not None:
            try:
                ref_ds.close()
            except Exception:
                pass
        if sim_ds is not None:
            try:
                sim_ds.close()
            except Exception:
                pass
        if staged_path is not None:
            try:
                os.unlink(staged_path)
            except FileNotFoundError:
                pass
