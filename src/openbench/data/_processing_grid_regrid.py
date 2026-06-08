"""Grid-data processing mixin and regridding helpers for OpenBench datasets."""

from __future__ import annotations

import gc
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


REGRID_ALGORITHM_VERSION = "2026-05-27.regrid-v2"
REGRID_BACKENDS = {
    "openbench_conservative",
    "cdo_remapcon",
    "xesmf_conservative",
    "basic_interpolation",
}


def _processing_attr(name, fallback):
    processing = sys.modules.get("openbench.data.processing")
    return getattr(processing, name, fallback) if processing is not None else fallback


def _parallel():
    return _processing_attr("Parallel", Parallel)


def _delayed():
    return _processing_attr("delayed", delayed)


class GridRegridMixin:
    """Split grid processing helpers."""

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

        try:
            data_regrid = backend_methods[backend](data, new_grid)
        except Exception as e:
            raise RuntimeError(f"Configured regrid backend {backend!r} failed: {e}") from e
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
        # Convert sparse arrays to dense arrays
        data_regrid = data.regrid.conservative(target_dataset, nan_threshold=0)
        # data_regrid = data_regrid.compute()

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

        regridder = xe.Regridder(data, new_grid, "conservative")
        return regridder(data)

    def remap_cdo(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        import os
        import subprocess
        import tempfile

        # Prepare data - ensure proper coordinate attributes
        data_prepared = data.copy()

        # Add CF-compliant coordinate attributes if missing
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
            # Create temporary files
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

            # Create target grid file
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
            # Check if data is None (all regrid methods failed)
            if data is None:
                logging.warning(f"No data to save for {data_source} year {year} - all regrid methods failed")
                return

            # Time resampling is now done before remap in _make_grid_parallel
            # (resample before remap is much more efficient for high-frequency data)
            data = data.sel(time=slice(f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59"))

            varname = self.ref_varname[0] if data_source == "ref" else self.sim_varname[0]

            out_file = os.path.join(self.casedir, "scratch", f"{data_source}_{varname}_remap_{year}.nc")
            _write_netcdf_atomic(data, out_file, compression=False)
            logging.info(f"Saved remapped {data_source} data for year {year} to {out_file}")
        finally:
            if hasattr(data, "close"):
                data.close()
            gc.collect()
