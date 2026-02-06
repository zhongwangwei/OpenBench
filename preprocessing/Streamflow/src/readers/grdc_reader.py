"""GRDC reader for daily and monthly multi-station NetCDF in ZIPs."""

import logging
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from . import register_reader
from .base_reader import BaseReader
from ..models import StationTimeSeries, StationMetadata, StationDataset


@register_reader("grdc")
class GRDCReader(BaseReader):
    """Reader for GRDC dataset (daily + monthly, NetCDF in ZIP)."""

    source_name = "grdc"

    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read GRDC data. Returns up to 2 StationDatasets (daily, monthly)."""
        data_path = Path(config["source"]["path"])
        options = config.get("options", {})
        max_zips = options.get("max_zips", None)

        datasets = []

        for resolution in ["daily", "monthly"]:
            res_dir = data_path / resolution
            if not res_dir.exists():
                self.logger.warning(
                    "GRDC %s directory not found: %s", resolution, res_dir
                )
                continue

            nc_filename = (
                "GRDC-Daily.nc" if resolution == "daily" else "GRDC-Monthly.nc"
            )
            stations, metadata = self._read_resolution(
                res_dir, nc_filename, resolution, max_zips
            )

            if stations:
                ds = StationDataset(
                    source_name=self.source_name,
                    time_resolution=resolution,
                    timezone_type="fixed_offset",
                    timezone_utc_offset=0.0,
                    timezone_definition="Per-station UTC offset (see timezone variable)",
                    stations=stations,
                    metadata=metadata,
                )
                datasets.append(ds)
                self.logger.info("GRDC %s: %d stations", resolution, len(stations))

        return datasets

    def _read_resolution(
        self,
        res_dir: Path,
        nc_filename: str,
        resolution: str,
        max_zips: Optional[int],
    ) -> Tuple[List[StationTimeSeries], Dict[str, StationMetadata]]:
        """Read all ZIPs for one resolution."""
        stations: List[StationTimeSeries] = []
        metadata: Dict[str, StationMetadata] = {}

        zip_files = sorted(res_dir.glob("*.zip"))
        if max_zips:
            zip_files = zip_files[:max_zips]

        for source_info in self._iterate_sources(res_dir, zip_files, nc_filename):
            try:
                self._process_netcdf(
                    source_info, nc_filename, stations, metadata
                )
            except Exception as e:
                self.logger.warning("Error reading %s: %s", source_info, e)
                continue

        return stations, metadata

    def _iterate_sources(self, res_dir: Path, zip_files: List[Path], nc_filename: str):
        """Yield paths to NetCDF files from extracted directories first, then ZIPs.

        For extracted directories, yields the path string directly.
        For ZIP files, yields a tuple (zip_path, nc_filename) so the caller
        knows it needs extraction.
        """
        # Already-extracted directories
        for d in sorted(res_dir.iterdir()):
            if d.is_dir():
                nc_path = d / nc_filename
                if nc_path.exists():
                    yield str(nc_path)

        # ZIP files
        for zf_path in zip_files:
            yield ("zip", zf_path, nc_filename)

    def _process_netcdf(
        self,
        source_info,
        nc_filename: str,
        stations: List[StationTimeSeries],
        metadata: Dict[str, StationMetadata],
    ) -> None:
        """Open a NetCDF (from path or ZIP) and append stations + metadata."""
        if isinstance(source_info, str):
            # Already-extracted file on disk
            ds = xr.open_dataset(source_info)
            ds.load()  # Load all data into memory
            self._extract_stations(ds, nc_filename, stations, metadata)
            ds.close()
        else:
            # ZIP extraction required
            _, zf_path, nc_name = source_info
            try:
                with zipfile.ZipFile(zf_path, "r") as zf:
                    # Find the NetCDF file inside the ZIP (may be nested)
                    nc_members = [
                        n for n in zf.namelist() if n.endswith(nc_name)
                    ]
                    if not nc_members:
                        self.logger.warning(
                            "No %s found in %s", nc_name, zf_path
                        )
                        return

                    with tempfile.TemporaryDirectory() as tmpdir:
                        for member in nc_members:
                            zf.extract(member, tmpdir)
                        extracted_path = Path(tmpdir) / nc_members[0]
                        ds = xr.open_dataset(str(extracted_path))
                        ds.load()  # Load ALL data before tmpdir is removed
                        self._extract_stations(ds, nc_name, stations, metadata)
                        ds.close()
            except zipfile.BadZipFile:
                self.logger.warning("Bad ZIP file: %s", zf_path)

    def _extract_stations(
        self,
        ds: xr.Dataset,
        nc_filename: str,
        stations: List[StationTimeSeries],
        metadata: Dict[str, StationMetadata],
    ) -> None:
        """Extract all stations from an opened (and loaded) xarray Dataset."""
        ids = ds["id"].values
        runoff = ds["runoff_mean"].values  # shape (time, id)
        time_vals = ds["time"].values
        geo_x = ds["geo_x"].values
        geo_y = ds["geo_y"].values
        areas = ds["area"].values

        # Optional variables
        station_names = self._get_string_var(ds, "station_name", len(ids))
        river_names = self._get_string_var(ds, "river_name", len(ids))
        countries = self._get_string_var(ds, "country", len(ids))

        geo_z = ds["geo_z"].values if "geo_z" in ds else np.full(len(ids), np.nan)

        for i, grdc_id in enumerate(ids):
            station_id = f"GRDC_{int(grdc_id)}"

            # Extract discharge, replace fill values with NaN
            discharge = runoff[:, i].astype(np.float64)
            discharge[discharge == -999.0] = np.nan
            discharge[discharge < -900.0] = np.nan

            lon = float(geo_x[i])
            lat = float(geo_y[i])
            area = float(areas[i])
            if area == -999.0 or area < 0:
                area = np.nan

            station = StationTimeSeries(
                station_id=station_id,
                discharge=discharge,
                time=time_vals,
                latitude=lat,
                longitude=lon,
                upstream_area=area,
            )

            meta = StationMetadata(
                name=station_names[i],
                river=river_names[i],
                country=countries[i],
                elevation=float(geo_z[i]),
                source_file=nc_filename,
                source_variable="runoff_mean",
                source_unit="m3/s",
                source_crs="WGS84",
            )

            stations.append(station)
            metadata[station_id] = meta

    @staticmethod
    def _get_string_var(ds: xr.Dataset, var_name: str, n: int) -> List[str]:
        """Safely get a string variable from an xarray Dataset."""
        if var_name in ds:
            vals = ds[var_name].values
            return [str(v) if v is not None else "" for v in vals]
        return [""] * n
