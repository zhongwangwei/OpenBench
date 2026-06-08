"""Unified-mask preprocessing helpers for runner tasks."""

from __future__ import annotations

import logging
import os
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

    if not os.path.exists(ref_path) or not os.path.exists(sim_path):
        logger.debug("Unified mask skipped: ref or sim file not found")
        return

    ref_ds = None
    sim_ds = None
    try:
        import xarray as xr

        ref_ds = xr.open_dataset(ref_path)
        sim_ds = xr.open_dataset(sim_path)

        o = ref_ds[ref_varname]
        s = sim_ds[sim_varname]

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
                logger.warning("Unified mask: no overlapping timestamps for %s, skipping", var_name)
                return

        # Apply mask: NaN where either is NaN
        mask = np.isnan(s_aligned.values) | np.isnan(o_aligned.values)
        o_data = o.load()
        if same_values:
            o_data.values[mask] = np.nan
        else:
            masked_overlap = o_aligned.load()
            masked_overlap.values[mask] = np.nan
            o_data.loc[{"time": masked_overlap["time"]}] = masked_overlap

        # Close datasets before writing
        ref_ds.close()
        ref_ds = None
        sim_ds.close()
        sim_ds = None

        # Write masked ref back. Use an atomic replace so a failed NetCDF rewrite
        # does not leave the existing reference file truncated or otherwise unreadable.
        write_netcdf_atomic_fn(o_data, ref_path, compression=False)
        logger.debug("Unified mask applied: %s (sim=%s)", var_name, sim_source)

        del o, s, o_aligned, s_aligned, mask, o_data

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
