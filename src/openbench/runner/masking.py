"""Unified-mask preprocessing helpers for runner tasks."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Callable

logger = logging.getLogger(__name__)

NetcdfWriter = Callable[..., None]
SimMaskSource = str | tuple[str, str]


def _mask_sources(sim_source: SimMaskSource | Sequence[SimMaskSource], default_varname: str) -> list[tuple[str, str]]:
    if isinstance(sim_source, str):
        return [(sim_source, default_varname)]
    if isinstance(sim_source, tuple) and len(sim_source) == 2 and all(isinstance(value, str) for value in sim_source):
        return [sim_source]
    return [(source, default_varname) if isinstance(source, str) else source for source in sim_source]


def _check_time_alignment(var_name: str, reference, simulation, time_alignment: str) -> bool:
    import numpy as np

    same_length = len(simulation["time"]) == len(reference["time"])
    same_values = same_length and np.array_equal(simulation["time"].values, reference["time"].values)
    if not same_length:
        message = (
            f"Unified mask: time length mismatch for {var_name} "
            f"(ref={len(reference['time'])}, sim={len(simulation['time'])})"
        )
        if time_alignment == "strict":
            raise ValueError(message)
        logger.warning("%s, using overlapping timestamps", message)
    elif not same_values:
        message = f"Unified mask: time values mismatch for {var_name} (lengths equal but timestamps differ)"
        if time_alignment == "strict":
            raise ValueError(message)
        logger.warning("%s, using overlapping timestamps", message)
    return same_values


def apply_unified_mask(
    info: dict,
    var_name: str,
    ref_source: str,
    sim_source: SimMaskSource | Sequence[SimMaskSource],
    ref_override: str | None = None,
    *,
    write_netcdf_atomic_fn: NetcdfWriter,
    apply_spatial_mask: bool = True,
) -> None:
    """Apply unified mask: set ref to NaN wherever sims are NaN."""
    import numpy as np
    import xarray as xr

    from openbench.util.names import select_data_array

    casedir = info["casedir"]
    ref_varname = info.get("ref_varname", "")
    sim_varname = info.get("sim_varname", "")
    time_alignment = info.get("time_alignment", "intersection")
    sim_sources = _mask_sources(sim_source, sim_varname)

    if ref_override:
        ref_path = ref_override
    else:
        ref_path = os.path.join(casedir, "data", f"{var_name}_ref_{ref_source}_{ref_varname}.nc")
    ref_path = os.path.abspath(ref_path)
    sim_paths = [
        os.path.abspath(os.path.join(casedir, "data", f"{var_name}_sim_{source}_{varname}.nc"))
        for source, varname in sim_sources
    ]

    missing_paths = [path for path in (ref_path, *sim_paths) if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(f"Unified mask input file not found: {', '.join(missing_paths)}")

    staged_path = None
    try:
        with ExitStack() as stack:
            ref_ds = stack.enter_context(xr.open_dataset(ref_path, chunks="auto"))
            ref_data = select_data_array(ref_ds, ref_varname, var_name)

            try:
                from openbench.util.converttype import Convert_Type

                ref_data = Convert_Type.convert_nc(ref_data)
            except ImportError:
                pass

            if len(sim_sources) == 1:
                o = ref_data
                for (source, source_varname), sim_path in zip(sim_sources, sim_paths, strict=True):
                    sim_ds = stack.enter_context(xr.open_dataset(sim_path, chunks="auto"))
                    s = select_data_array(sim_ds, source_varname, var_name)
                    try:
                        from openbench.util.converttype import Convert_Type

                        s = Convert_Type.convert_nc(s)
                    except ImportError:
                        pass

                    same_values = _check_time_alignment(var_name, o, s, time_alignment)
                    if same_values:
                        o_aligned, s_aligned = o, s
                    else:
                        excluded_dims = set() if apply_spatial_mask else (set(o.dims) | set(s.dims)) - {"time"}
                        o_aligned, s_aligned = xr.align(o, s, join="inner", exclude=excluded_dims)
                        if o_aligned.sizes.get("time", 0) == 0:
                            raise ValueError(f"Unified mask: no overlapping timestamps for {var_name}")

                    finite_pairs = np.isfinite(s_aligned) & np.isfinite(o_aligned) if apply_spatial_mask else None
                    if finite_pairs is not None and time_alignment == "intersection" and "time" in finite_pairs.dims:
                        non_time_dims = [dim for dim in finite_pairs.dims if dim != "time"]
                        valid_times = finite_pairs.any(dim=non_time_dims) if non_time_dims else finite_pairs
                        if hasattr(valid_times, "compute"):
                            valid_times = valid_times.compute()
                        if not bool(valid_times.any().item()):
                            raise ValueError(f"Unified mask: no overlapping finite timestamps for {var_name}")
                        o_aligned = o_aligned.where(valid_times, drop=True)
                        s_aligned = s_aligned.where(valid_times, drop=True)
                        finite_pairs = np.isfinite(s_aligned) & np.isfinite(o_aligned)

                    invalid_overlap = ~finite_pairs if finite_pairs is not None else None
                    if time_alignment == "intersection":
                        o = o_aligned.where(~invalid_overlap) if invalid_overlap is not None else o_aligned
                    elif same_values:
                        o = o.where(~invalid_overlap) if invalid_overlap is not None else o
                    else:
                        invalid_full = (
                            invalid_overlap.reindex_like(o, fill_value=False) if invalid_overlap is not None else None
                        )
                        o = o.where(~invalid_full) if invalid_full is not None else o
            else:
                aligned_ref = ref_data
                global_finite = None
                # ponytail: one open handle per sibling avoids repeat reads/writes;
                # batch/reopen only if real model counts approach OS handle limits.
                for (source, source_varname), sim_path in zip(sim_sources, sim_paths, strict=True):
                    sim_ds = stack.enter_context(xr.open_dataset(sim_path, chunks="auto"))
                    sim_data = select_data_array(sim_ds, source_varname, var_name)
                    try:
                        from openbench.util.converttype import Convert_Type

                        sim_data = Convert_Type.convert_nc(sim_data)
                    except ImportError:
                        pass

                    same_values = _check_time_alignment(var_name, aligned_ref, sim_data, time_alignment)
                    if apply_spatial_mask or not same_values:
                        excluded_dims = (
                            set() if apply_spatial_mask else (set(aligned_ref.dims) | set(sim_data.dims)) - {"time"}
                        )
                        aligned_ref, sim_aligned = xr.align(
                            aligned_ref,
                            sim_data,
                            join="inner",
                            exclude=excluded_dims,
                        )
                        if aligned_ref.sizes.get("time", 0) == 0:
                            raise ValueError(f"Unified mask: no overlapping timestamps for {var_name}")
                        if global_finite is not None:
                            global_finite = global_finite.reindex_like(aligned_ref, fill_value=False)
                    else:
                        sim_aligned = sim_data
                    if apply_spatial_mask:
                        sim_finite = np.isfinite(sim_aligned)
                        global_finite = sim_finite if global_finite is None else (global_finite & sim_finite)

                if apply_spatial_mask:
                    assert global_finite is not None
                    global_finite = global_finite & np.isfinite(aligned_ref)
                    if time_alignment == "intersection" and "time" in global_finite.dims:
                        non_time_dims = [dim for dim in global_finite.dims if dim != "time"]
                        valid_times = global_finite.any(dim=non_time_dims) if non_time_dims else global_finite
                        if hasattr(valid_times, "compute"):
                            valid_times = valid_times.compute()
                        if not bool(valid_times.any().item()):
                            raise ValueError(f"Unified mask: no overlapping finite timestamps for {var_name}")
                        global_finite = global_finite.where(valid_times, drop=True)
                        aligned_ref = aligned_ref.where(valid_times, drop=True)
                    o = aligned_ref.where(global_finite)
                else:
                    o = aligned_ref

            fd, staged_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(ref_path)}.mask-",
                suffix=".nc",
                dir=os.path.dirname(ref_path),
            )
            os.close(fd)
            write_netcdf_atomic_fn(o, staged_path, compression=False)

        os.replace(staged_path, ref_path)
        staged_path = None
        logger.debug("Unified mask applied: %s (sims=%s)", var_name, [source for source, _ in sim_sources])

    except Exception:
        logger.exception("Unified mask failed for %s (sim=%s)", var_name, sim_source)
        raise
    finally:
        if staged_path is not None:
            try:
                os.unlink(staged_path)
            except FileNotFoundError:
                pass
