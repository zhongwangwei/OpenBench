"""Grid-data processing mixin and regridding helpers for OpenBench datasets."""

from __future__ import annotations

import glob
import logging
import os
import sys

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic

try:
    from openbench.util.dataset_loader import cached_glob, write_mfdataset_atomic as write_mfdataset_chunked_atomic
except ImportError:  # pragma: no cover - mirrors processing.py fallback
    cached_glob = lambda pattern, **kwargs: sorted(glob.glob(pattern))

    def write_mfdataset_chunked_atomic(paths, output_path, *, sortby=None, compression=None, **kwargs):
        with xr.open_mfdataset(paths, **kwargs) as ds:
            if sortby is not None:
                ds = ds.sortby(sortby)
            _write_netcdf_atomic(ds, output_path, compression=compression)


REGRID_ALGORITHM_VERSION = "2026-10-26.regrid-v4"
REGRID_BACKENDS = {
    "openbench_conservative",
    "cdo_remapcon",
    "xesmf_conservative",
    "basic_interpolation",
}

_EARTH_RADIUS_M = 6_371_000.0
_EXTENSIVE_GRID_CELL_ITEMS = {"inundation_area"}


def _axis_cell_factors(values: np.ndarray, *, spherical_latitude: bool) -> np.ndarray:
    """Return cell widths in radians, or spherical latitude area factors."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Grid-cell areas require at least two one-dimensional coordinates per axis")

    order = np.argsort(values)
    sorted_values = values[order]
    if np.any(np.diff(sorted_values) <= 0):
        raise ValueError("Grid-cell areas require unique coordinate centres")

    midpoints = (sorted_values[:-1] + sorted_values[1:]) / 2.0
    bounds = np.concatenate(
        ([2.0 * sorted_values[0] - midpoints[0]], midpoints, [2.0 * sorted_values[-1] - midpoints[-1]])
    )
    if spherical_latitude:
        bounds = np.clip(bounds, -90.0, 90.0)
        sorted_factors = np.sin(np.radians(bounds[1:])) - np.sin(np.radians(bounds[:-1]))
    else:
        sorted_factors = np.radians(bounds[1:] - bounds[:-1])

    factors = np.empty_like(sorted_factors)
    factors[order] = sorted_factors
    return factors


def _rectilinear_cell_area(grid: xr.Dataset) -> xr.DataArray:
    """Calculate spherical cell areas from one-dimensional lat/lon centres."""
    if "lat" not in grid.coords or "lon" not in grid.coords:
        raise ValueError("Grid-cell areas require 'lat' and 'lon' coordinates")
    if grid["lat"].ndim != 1 or grid["lon"].ndim != 1:
        raise ValueError("Grid-cell areas require one-dimensional 'lat' and 'lon' coordinates")

    lat_dim = grid["lat"].dims[0]
    lon_dim = grid["lon"].dims[0]
    lat_factors = _axis_cell_factors(grid["lat"].values, spherical_latitude=True)
    lon_factors = _axis_cell_factors(grid["lon"].values, spherical_latitude=False)
    coords = {}
    for dim in (lat_dim, lon_dim):
        if dim in grid.coords:
            coords[dim] = grid[dim]
    return xr.DataArray(
        _EARTH_RADIUS_M**2 * np.outer(lat_factors, lon_factors),
        dims=(lat_dim, lon_dim),
        coords=coords,
        name="cell_area",
        attrs={"units": "m2"},
    )


def _processing_attr(name, fallback):
    processing = sys.modules.get("openbench.data.processing")
    return getattr(processing, name, fallback) if processing is not None else fallback


def _parallel():
    return _processing_attr("Parallel", Parallel)


def _delayed():
    return _processing_attr("delayed", delayed)


class GridRegridMixin:
    """Split grid processing helpers."""

    _SAME_GRID_RTOL = 1e-7
    _SAME_GRID_ATOL = 1e-8

    @staticmethod
    def _normalized_longitudes(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        return ((values + 180.0) % 360.0) - 180.0

    @classmethod
    def _grids_match(cls, data: xr.Dataset, target: xr.Dataset) -> bool:
        """Return whether two ordered rectilinear grids have the same cell centres."""
        if "lat" not in data.coords or "lon" not in data.coords:
            return False
        if "lat" not in target.coords or "lon" not in target.coords:
            return False
        if data["lat"].ndim != 1 or data["lon"].ndim != 1:
            return False
        if target["lat"].ndim != 1 or target["lon"].ndim != 1:
            return False

        source_lat = np.asarray(data["lat"].values)
        target_lat = np.asarray(target["lat"].values)
        source_lon = cls._normalized_longitudes(data["lon"].values)
        target_lon = cls._normalized_longitudes(target["lon"].values)
        if source_lat.shape != target_lat.shape or source_lon.shape != target_lon.shape:
            return False
        return bool(
            np.allclose(source_lat, target_lat, rtol=cls._SAME_GRID_RTOL, atol=cls._SAME_GRID_ATOL)
            and np.allclose(source_lon, target_lon, rtol=cls._SAME_GRID_RTOL, atol=cls._SAME_GRID_ATOL)
        )

    @staticmethod
    def _assign_target_grid(data: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
        """Use exact target coordinates while retaining source coordinate metadata."""
        coords = {}
        for name in ("lat", "lon"):
            coords[name] = xr.DataArray(
                target[name].values,
                dims=data[name].dims,
                attrs=dict(data[name].attrs),
            )
        return data.assign_coords(coords)

    def _uses_extensive_grid_cell_regridding(self) -> bool:
        item = str(getattr(self, "item", "")).strip().lower().replace("-", "_").replace(" ", "_")
        return item in _EXTENSIVE_GRID_CELL_ITEMS

    @staticmethod
    def _scale_grid_cell_totals(data: xr.Dataset, cell_area: xr.DataArray, *, to_density: bool) -> xr.Dataset:
        """Convert spatial grid-cell totals to density, or density back to totals."""
        result = data.copy()
        area_dims = set(cell_area.dims)
        for name, array in data.data_vars.items():
            if not area_dims.issubset(array.dims):
                continue
            attrs = dict(array.attrs)
            result[name] = array / cell_area if to_density else array * cell_area
            result[name].attrs = attrs
        return result

    def remap_data(self, data: xr.Dataset) -> xr.Dataset:
        new_grid = self.create_target_grid()
        backend = str(getattr(self, "regrid_backend", "openbench_conservative") or "openbench_conservative").lower()
        backend_methods = {
            "openbench_conservative": self.remap_interpolate,
            "cdo_remapcon": self.remap_cdo,
            "xesmf_conservative": self.remap_xesmf,
            "basic_interpolation": self.remap_basic_interpolation,
        }
        if backend not in backend_methods:
            valid = ", ".join(sorted(backend_methods))
            raise ValueError(f"Unknown regrid_backend {backend!r}; expected one of: {valid}")

        if backend == "openbench_conservative" and self._grids_match(data, new_grid):
            logging.info("[REGRID] Skip regridding: source grid already matches target grid")
            return self._mark_regrid_backend(self._assign_target_grid(data, new_grid), backend)

        extensive_grid_cells = self._uses_extensive_grid_cell_regridding()
        if extensive_grid_cells:
            data = self._scale_grid_cell_totals(data, _rectilinear_cell_area(data), to_density=True)
            logging.info("[REGRID] Regridding %s as an extensive grid-cell total", self.item)

        try:
            data_regrid = backend_methods[backend](data, new_grid)
        except Exception as e:
            raise RuntimeError(f"Configured regrid backend {backend!r} failed: {e}") from e
        if extensive_grid_cells:
            data_regrid = self._scale_grid_cell_totals(
                data_regrid,
                _rectilinear_cell_area(new_grid),
                to_density=False,
            )
            data_regrid.attrs["openbench_regrid_quantity"] = "extensive_grid_cell_total"
        return self._mark_regrid_backend(data_regrid, backend)

    def create_target_grid(self) -> xr.Dataset:
        lon_new = np.arange(self.min_lon + self.compare_grid_res / 2, self.max_lon, self.compare_grid_res)
        lat_new = np.arange(self.min_lat + self.compare_grid_res / 2, self.max_lat, self.compare_grid_res)
        return xr.Dataset({"lon": lon_new, "lat": lat_new})

    def remap_interpolate(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        """OpenBench conservative regrid backend.

        Kept under the historical method name for compatibility; this is not
        linear interpolation.
        """
        from openbench.data.regrid import Grid

        grid = Grid(
            north=self.max_lat - self.compare_grid_res / 2,
            south=self.min_lat + self.compare_grid_res / 2,
            west=self.min_lon + self.compare_grid_res / 2,
            east=self.max_lon - self.compare_grid_res / 2,
            resolution_lat=self.compare_grid_res,
            resolution_lon=self.compare_grid_res,
        )
        target_dataset = grid.create_regridding_dataset(lat_name="lat", lon_name="lon")
        data_regrid = data.regrid.conservative(target_dataset, nan_threshold=0)

        return data_regrid

    @staticmethod
    def _mark_regrid_backend(data: xr.Dataset, backend: str) -> xr.Dataset:
        data = data.copy()
        data.attrs["openbench_regrid_backend"] = backend
        data.attrs["openbench_regrid_algorithm_version"] = REGRID_ALGORITHM_VERSION
        return data

    def remap_basic_interpolation(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        """Explicit non-conservative interpolation backend for opt-in use."""
        return data.interp(lat=new_grid["lat"], lon=new_grid["lon"], method="linear")

    def remap_xesmf(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        import xesmf as xe

        from openbench.data.regrid.xesmf_cache import cached_regridder, default_weight_cache_dir

        regridder = cached_regridder(
            xe,
            data,
            new_grid,
            "conservative",
            cache_dir=default_weight_cache_dir(self),
            periodic=False,
        )
        return regridder(data)

    def remap_cdo(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        import subprocess
        import tempfile

        data_prepared = data.copy()

        if "lon" in data_prepared.coords:
            if "standard_name" not in data_prepared["lon"].attrs:
                data_prepared["lon"].attrs["standard_name"] = "longitude"
            if "units" not in data_prepared["lon"].attrs:
                data_prepared["lon"].attrs["units"] = "degrees_east"

        if "lat" in data_prepared.coords:
            if "standard_name" not in data_prepared["lat"].attrs:
                data_prepared["lat"].attrs["standard_name"] = "latitude"
            if "units" not in data_prepared["lat"].attrs:
                data_prepared["lat"].attrs["units"] = "degrees_north"

        temp_input_name = None
        temp_output_name = None
        temp_grid_name = None

        try:
            temp_input = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
            temp_input_name = temp_input.name
            temp_input.close()

            temp_output = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
            temp_output_name = temp_output.name
            temp_output.close()

            temp_grid = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
            temp_grid_name = temp_grid.name
            temp_grid.close()

            # Save data to NetCDF with NETCDF4_CLASSIC format for CDO compatibility
            _write_netcdf_atomic(
                data_prepared,
                temp_input_name,
                compression=False,
                format="NETCDF4_CLASSIC",
            )

            self.create_target_grid_file(temp_grid_name, new_grid)

            # Use remapcon (conservative remapping) — CDO's standard conservative method.
            # List form (shell=False default) avoids shell injection on paths and
            # bypasses an unnecessary shell parse layer.
            cmd = ["cdo", "-s", f"remapcon,{temp_grid_name}", temp_input_name, temp_output_name]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Read result and load into memory before cleanup
            with xr.open_dataset(temp_output_name) as result_ds:
                result_data = result_ds.load()
            return Convert_Type.convert_nc(result_data)

        finally:
            # Clean up temporary files
            for f in [temp_input_name, temp_output_name, temp_grid_name]:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except (OSError, PermissionError) as e:
                        logging.debug(f"Could not delete temporary file {f}: {e}")

    def create_target_grid_file(self, filename: str, new_grid: xr.Dataset) -> None:
        with open(filename, "w") as f:
            f.write("gridtype = lonlat\n")
            f.write(f"xsize = {len(new_grid.lon)}\n")
            f.write(f"ysize = {len(new_grid.lat)}\n")
            f.write(f"xfirst = {self.min_lon + self.compare_grid_res / 2}\n")
            f.write(f"xinc = {self.compare_grid_res}\n")
            f.write(f"yfirst = {self.min_lat + self.compare_grid_res / 2}\n")
            f.write(f"yinc = {self.compare_grid_res}\n")

    def save_remapped_data(self, data: xr.Dataset, data_source: str, year: int) -> None:
        try:
            if data is None:
                logging.warning(f"No data to save for {data_source} year {year} - all regrid methods failed")
                return

            # Time resampling is now done before remap in _make_grid_parallel
            # (resample before remap is much more efficient for high-frequency data)
            data = data.sel(time=slice(f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59"))
            if "time" in data.coords:
                data["time"].attrs.pop("_openbench_temporal_resolution", None)

            out_file = self._remap_scratch_file(data_source, year)
            _write_netcdf_atomic(data, out_file, compression=False)
            logging.info(f"Saved remapped {data_source} data for year {year} to {out_file}")
        finally:
            if hasattr(data, "close"):
                data.close()
