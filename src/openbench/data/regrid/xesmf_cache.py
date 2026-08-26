"""Small persistent weight-cache wrapper for xESMF regridders."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from openbench.util.netcdf import write_file_atomic

XESMF_WEIGHT_SCHEMA_VERSION = 1


def cached_regridder(
    xe: Any,
    source: xr.Dataset | xr.DataArray,
    target: xr.Dataset,
    method: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    periodic: bool = False,
):
    """Return an xESMF Regridder, loading/storing weights when a cache dir is configured."""
    if cache_dir is None:
        return _new_regridder(xe, source, target, method, periodic=periodic)

    cache_path = _cache_path(source, target, method, cache_dir=cache_dir, periodic=periodic)
    if cache_path.exists():
        return _new_regridder(xe, source, target, method, periodic=periodic, weights=str(cache_path))

    from openbench.runner.cache import _file_lock

    with _file_lock(cache_path.with_suffix(cache_path.suffix + ".lock")):
        if cache_path.exists():
            return _new_regridder(xe, source, target, method, periodic=periodic, weights=str(cache_path))
        regridder = _new_regridder(xe, source, target, method, periodic=periodic)
        write_file_atomic(cache_path, lambda tmp: regridder.to_netcdf(str(tmp)), suffix=".nc")
        return regridder


def _new_regridder(
    xe: Any,
    source: xr.Dataset | xr.DataArray,
    target: xr.Dataset,
    method: str,
    *,
    periodic: bool,
    weights: str | None = None,
):
    kwargs: dict[str, object] = {"periodic": periodic}
    if weights is not None:
        kwargs["weights"] = weights
    return xe.Regridder(source, target, method, **kwargs)


def default_weight_cache_dir(owner: object | None = None) -> str | None:
    """Return configured xESMF weight-cache dir; mixin callers fall back to case scratch."""
    env_dir = os.environ.get("OPENBENCH_XESMF_WEIGHT_CACHE_DIR")
    if env_dir:
        return str(Path(env_dir).expanduser())
    casedir = getattr(owner, "casedir", None)
    if casedir:
        return os.path.join(str(casedir), "scratch", "xesmf_weights")
    return None


def _cache_path(
    source: xr.Dataset | xr.DataArray,
    target: xr.Dataset,
    method: str,
    *,
    cache_dir: str | os.PathLike[str],
    periodic: bool,
) -> Path:
    digest = hashlib.sha256()
    payload = {
        "schema": XESMF_WEIGHT_SCHEMA_VERSION,
        "method": method,
        "periodic": bool(periodic),
        "versions": _versions(),
        "source": _grid_token(source),
        "target": _grid_token(target),
    }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return Path(cache_dir).expanduser() / f"xesmf-weights-{digest.hexdigest()}.nc"


def _versions() -> dict[str, str | None]:
    return {name: _package_version(name) for name in ("xesmf", "ESMF", "esmpy")}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _grid_token(grid: xr.Dataset | xr.DataArray) -> list[dict[str, object]]:
    dataset = grid.to_dataset(name="__dataarray__") if isinstance(grid, xr.DataArray) else grid
    names = {
        str(name)
        for name, array in dataset.variables.items()
        if str(name).lower() in {"lon", "lat", "lon_b", "lat_b", "longitude", "latitude", "x", "y", "mask"}
        or str(array.attrs.get("standard_name", "")).lower() in {"longitude", "latitude"}
        or str(array.attrs.get("long_name", "")).lower() in {"longitude", "latitude"}
        or str(array.attrs.get("axis", "")).upper() in {"X", "Y"}
        or _is_latlon_unit(array.attrs.get("units"))
    }
    for name in tuple(names):
        bounds = dataset[name].attrs.get("bounds")
        if isinstance(bounds, str) and bounds in dataset.variables:
            names.add(bounds)
    return [_array_token(name, dataset[name]) for name in sorted(names)]


def _is_latlon_unit(value: object) -> bool:
    unit = str(value or "").lower().replace("_", "").replace(" ", "")
    return unit in {
        "degreee",
        "degreeeast",
        "degreen",
        "degreenorth",
        "degreese",
        "degreeseast",
        "degreesn",
        "degreesnorth",
    }


def _array_token(name: str, array: xr.DataArray) -> dict[str, object]:
    values = np.ascontiguousarray(array.to_numpy())
    digest = hashlib.sha256(values.view(np.uint8)).hexdigest()
    return {
        "name": name,
        "dims": tuple(str(dim) for dim in array.dims),
        "dtype": values.dtype.str,
        "shape": tuple(values.shape),
        "sha256": digest,
        "attrs": {
            key: str(array.attrs[key])
            for key in ("axis", "bounds", "long_name", "standard_name", "units")
            if key in array.attrs
        },
    }
