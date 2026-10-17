#!/usr/bin/env python3
"""Lightweight synthetic benchmark for OpenBench conservative regridding."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import xarray as xr

from openbench.data._processing_grid_regrid import GridRegridMixin
from openbench.data.regrid.methods import conservative
from openbench.data.regrid.utils import create_dot_dataarray


@dataclass(frozen=True)
class Case:
    name: str
    source_resolution: float
    target_resolution: float
    time_steps: int = 1
    with_nan: bool = False


CASES = (
    Case("A_same_0.5_to_0.5", 0.5, 0.5),
    Case("B_0.25_to_0.5", 0.25, 0.5),
    Case("C_0.1_to_0.25", 0.1, 0.25),
    Case("D_nan_0.25_to_0.5", 0.25, 0.5, with_nan=True),
    Case("E_multitime_0.25_to_0.5", 0.25, 0.5, time_steps=8),
)


def _centres(start: float, stop: float, resolution: float) -> np.ndarray:
    return np.arange(start + resolution / 2, stop, resolution)


def _measure(function):
    tracemalloc.start()
    start = time.perf_counter()
    result = function()
    if hasattr(result, "load"):
        result = result.load()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 1024**2


def _build_case(case: Case, *, global_domain: bool = False):
    # The default 40° × 80° domain is suitable for CI/workstations;
    # --global-domain exposes scaling at realistic global-grid sizes.
    min_lat, max_lat = (-90.0, 90.0) if global_domain else (-20.0, 20.0)
    min_lon, max_lon = (0.0, 360.0) if global_domain else (0.0, 80.0)
    lat = _centres(min_lat, max_lat, case.source_resolution)
    lon = _centres(min_lon, max_lon, case.source_resolution)
    target_lat = _centres(min_lat, max_lat, case.target_resolution)
    target_lon = _centres(min_lon, max_lon, case.target_resolution)
    time_coord = np.arange(case.time_steps)
    values = (
        np.sin(np.radians(lat))[None, :, None]
        + np.cos(np.radians(lon))[None, None, :]
        + time_coord[:, None, None] * 0.01
    ).astype("float64")
    if case.with_nan:
        values[:, values.shape[1] // 3 : values.shape[1] // 2, values.shape[2] // 3] = np.nan
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time_coord, "lat": lat, "lon": lon},
        name="synthetic",
    )
    target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})
    return data, target


def _construct_weights(data: xr.DataArray, target: xr.Dataset):
    conservative.clear_weight_cache()
    start = time.perf_counter()
    weights = {
        "lat": create_dot_dataarray(
            conservative.get_weights(data["lat"].values, target["lat"].values, spherical=True),
            "lat",
            target["lat"].values,
            data["lat"].values,
        ),
        "lon": create_dot_dataarray(
            conservative.get_weights(data["lon"].values, target["lon"].values),
            "lon",
            target["lon"].values,
            data["lon"].values,
        ),
    }
    density = sum(np.count_nonzero(weight.values) for weight in weights.values()) / sum(
        weight.size for weight in weights.values()
    )
    return weights, time.perf_counter() - start, density


def _same_grid_bypass(data: xr.DataArray, target: xr.Dataset):
    resolution = float(np.diff(target["lat"].values[:2]).item())

    class Processor(GridRegridMixin):
        min_lat = float(target["lat"].values[0] - resolution / 2)
        max_lat = float(target["lat"].values[-1] + resolution / 2)
        min_lon = float(target["lon"].values[0] - resolution / 2)
        max_lon = float(target["lon"].values[-1] + resolution / 2)
        compare_grid_res = resolution
        regrid_backend = "openbench_conservative"

    return Processor().remap_data(data.to_dataset())


def _maybe_xesmf(data: xr.DataArray, target: xr.Dataset):
    import xesmf as xe

    regridder = xe.Regridder(data.to_dataset(), target, "conservative", periodic=False)
    return regridder(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-xesmf", action="store_true", help="also run xESMF when ESMF is installed")
    parser.add_argument("--global-domain", action="store_true", help="use a synthetic global domain")
    args = parser.parse_args()

    print("case,implementation,weight_seconds,weight_density,regrid_seconds,peak_mib,max_abs_difference")
    for case in CASES:
        data, target = _build_case(case, global_domain=args.global_domain)
        weights, weight_seconds, density = _construct_weights(data, target)
        dense, dense_seconds, dense_peak = _measure(
            lambda: conservative.apply_weights(data, weights, True, 0.0, use_sparse=False)
        )
        optimized, sparse_seconds, sparse_peak = _measure(
            lambda: conservative.apply_weights(data, weights, True, 0.0, use_sparse=True)
        )
        difference = float(abs(optimized - dense).max(skipna=True).item())
        print(f"{case.name},dense,{weight_seconds:.6f},{density:.6f},{dense_seconds:.6f},{dense_peak:.3f},0")
        print(
            f"{case.name},sparse,{weight_seconds:.6f},{density:.6f},"
            f"{sparse_seconds:.6f},{sparse_peak:.3f},{difference:.3e}"
        )

        if case.source_resolution == case.target_resolution:
            _, bypass_seconds, bypass_peak = _measure(lambda: _same_grid_bypass(data, target))
            print(f"{case.name},same_grid_bypass,0,0,{bypass_seconds:.6f},{bypass_peak:.3f},0")

        if args.with_xesmf:
            try:
                xesmf_result, xesmf_seconds, xesmf_peak = _measure(lambda: _maybe_xesmf(data, target))
                xesmf_difference = float(abs(xesmf_result - dense).max(skipna=True).item())
                print(f"{case.name},xesmf,unknown,unknown,{xesmf_seconds:.6f},{xesmf_peak:.3f},{xesmf_difference:.3e}")
            except Exception as exc:
                print(f"{case.name},xesmf_unavailable,unknown,unknown,0,0,{type(exc).__name__}")


if __name__ == "__main__":
    main()
