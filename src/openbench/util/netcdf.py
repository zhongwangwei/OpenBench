"""NetCDF IO helpers shared by runner, processing, and output modules."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


_TRUTHY = {"1", "true", "yes", "y", "on"}
_NETCDF_COMPRESSION_ENV = "OPENBENCH_NETCDF_COMPRESSION"
_NETCDF_COMPRESSION_LEVEL_ENV = "OPENBENCH_NETCDF_COMP_LEVEL"
_DEFAULT_COMPRESSION_LEVEL = 1


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _netcdf_compression_level() -> int:
    raw = os.environ.get(_NETCDF_COMPRESSION_LEVEL_ENV)
    if raw is None or raw.strip() == "":
        return _DEFAULT_COMPRESSION_LEVEL
    try:
        return max(0, min(9, int(raw)))
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using compression level %s",
            _NETCDF_COMPRESSION_LEVEL_ENV,
            raw,
            _DEFAULT_COMPRESSION_LEVEL,
        )
        return _DEFAULT_COMPRESSION_LEVEL


def _supports_netcdf_compression(kwargs: dict[str, Any]) -> bool:
    engine = str(kwargs.get("engine", "")).lower()
    if engine == "scipy":
        return False

    fmt = str(kwargs.get("format", "")).upper()
    if fmt.startswith("NETCDF3"):
        return False

    return True


def _compressible_variables(data: Any) -> list[str]:
    if isinstance(data, xr.DataArray):
        if data.name is None:
            return []
        return [str(data.name)] if _is_compressible_dtype(data.dtype) else []

    if isinstance(data, xr.Dataset):
        return [name for name, da in data.data_vars.items() if _is_compressible_dtype(da.dtype)]

    return []


def _is_compressible_dtype(dtype: Any) -> bool:
    resolved = np.dtype(dtype)
    # Keep object/string variables uncompressed; zlib can be backend-sensitive
    # for variable-length strings, while numeric grids are the high-ROI target.
    return resolved.kind in {"b", "i", "u", "f", "c"}


def _compression_enabled(compression: bool | None) -> bool:
    if compression is None:
        return _env_truthy(_NETCDF_COMPRESSION_ENV)
    return bool(compression)


def _compression_encoding(
    data: Any,
    kwargs: dict[str, Any],
    *,
    compression: bool | None = None,
) -> dict[str, dict[str, Any]]:
    if not _compression_enabled(compression) or not _supports_netcdf_compression(kwargs):
        return dict(kwargs.get("encoding") or {})

    encoding: dict[str, dict[str, Any]] = {
        str(name): dict(value) for name, value in (kwargs.get("encoding") or {}).items()
    }
    compression = {"zlib": True, "complevel": _netcdf_compression_level(), "shuffle": True}
    for name in _compressible_variables(data):
        variable_encoding = encoding.setdefault(name, {})
        for key, value in compression.items():
            variable_encoding.setdefault(key, value)
    return encoding


def _netcdf_kwargs_with_default_compression(
    data: Any,
    kwargs: dict[str, Any],
    *,
    compression: bool | None = None,
) -> dict[str, Any]:
    encoding = _compression_encoding(data, kwargs, compression=compression)
    if not encoding:
        return dict(kwargs)

    out = dict(kwargs)
    out["encoding"] = encoding
    return out


def _fsync_file(path: Path) -> None:
    """Best-effort fsync for a fully written temporary file."""
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        logger.debug("Could not fsync temporary output file: %s", path)


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for the directory entry updated by os.replace."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("Could not fsync output directory: %s", path)
    finally:
        os.close(fd)


def write_file_atomic(
    output_path: str | os.PathLike[str],
    writer: Any,
    *,
    suffix: str = ".tmp",
) -> None:
    """Run ``writer(temp_path)`` then atomically replace ``output_path``."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=suffix, dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        _fsync_file(tmp_path)
        os.replace(tmp_path, target)
        _fsync_directory(target.parent)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.debug("Could not remove temporary output file: %s", tmp_path)


def write_netcdf_atomic(
    data: Any,
    output_path: str | os.PathLike[str],
    *,
    compression: bool | None = None,
    **kwargs: Any,
) -> None:
    """Write a NetCDF file via same-directory temp file then atomic replace.

    Direct ``to_netcdf(target)`` can leave ``target`` truncated or otherwise
    unreadable if the write fails. Writing in the target directory first keeps
    ``os.replace`` atomic on the same filesystem and preserves any existing
    file until the new NetCDF is complete.

    When ``OPENBENCH_NETCDF_COMPRESSION=1`` is set, numeric data variables are
    written with zlib compression. ``OPENBENCH_NETCDF_COMP_LEVEL`` controls the
    zlib level and defaults to 1 to reduce IO without spending much extra CPU.
    Explicit per-variable ``encoding`` entries are preserved.

    ``compression=False`` is for scratch/intermediate files that are written
    only to be reopened by the same run. It bypasses the env default so a
    final-output compression run does not spend CPU compressing temporary flat
    files or mfdataset batch shards.
    """
    nc_kwargs = _netcdf_kwargs_with_default_compression(data, kwargs, compression=compression)
    write_file_atomic(output_path, lambda tmp_path: data.to_netcdf(tmp_path, **nc_kwargs), suffix=".tmp.nc")
