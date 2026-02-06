"""GSHA reader for monthly consolidated NetCDF (~21k stations)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from . import register_reader
from .base_reader import BaseReader
from ..models import StationTimeSeries, StationMetadata, StationDataset

# Fill value used in GSHA NetCDF for both data and coordinate variables
_FILL_VALUE = -9999.0


@register_reader("gsha")
class GSHAReader(BaseReader):
    """Reader for GSHA dataset (monthly, single consolidated NetCDF).

    Data lives in a single file ``GSHA_monthly.nc`` with dimensions
    ``(station, time)`` -- note the dimension order is *station-major*.
    The primary discharge variable is ``mean`` (monthly mean, m3/s).
    """

    source_name = "gsha"

    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read GSHA data.  Returns one StationDataset (monthly).

        Config keys used:
            source.path   -- directory containing GSHA_monthly.nc
            options.max_stations -- optional cap on number of stations
        """
        data_path = Path(config["source"]["path"])
        options = config.get("options", {})
        max_stations: Optional[int] = options.get("max_stations", None)

        nc_file = data_path / "GSHA_monthly.nc"
        if not nc_file.exists():
            self.logger.error("GSHA file not found: %s", nc_file)
            return []

        self.logger.info("Opening GSHA file: %s", nc_file)
        ds = xr.open_dataset(nc_file)
        ds.load()  # load everything into memory so we can close the file
        ds.close()

        stations, metadata = self._extract_stations(ds, max_stations)

        if not stations:
            self.logger.warning("No stations extracted from GSHA data")
            return []

        dataset = StationDataset(
            source_name=self.source_name,
            time_resolution="monthly",
            timezone_type="utc",
            timezone_utc_offset=0.0,
            timezone_definition="UTC (monthly means)",
            stations=stations,
            metadata=metadata,
        )

        self.logger.info("GSHA monthly: %d stations loaded", len(stations))
        return [dataset]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_stations(
        self,
        ds: xr.Dataset,
        max_stations: Optional[int],
    ) -> tuple:
        """Extract station time-series and metadata from a loaded Dataset.

        Returns:
            (stations_list, metadata_dict)
        """
        # Coordinate / data arrays  -----------------------------------
        station_ids = ds["station_id"].values            # (station,)
        mean_discharge = ds["mean"].values               # (station, time)
        time_vals = ds["time"].values                    # (time,)
        latitudes = ds["latitude"].values                # (station,)
        longitudes = ds["longitude"].values              # (station,)
        areas = ds["area"].values                        # (station,)

        # Optional metadata arrays
        agencies = self._get_string_var(ds, "agency", len(station_ids))
        verifications = self._get_string_var(ds, "verification", len(station_ids))
        comids = ds["COMID"].values if "COMID" in ds else np.zeros(len(station_ids), dtype=np.int32)

        n_stations = len(station_ids)
        if max_stations is not None:
            n_stations = min(n_stations, max_stations)

        stations: List[StationTimeSeries] = []
        metadata: Dict[str, StationMetadata] = {}

        for i in range(n_stations):
            sid = str(station_ids[i])

            # Discharge: shape is (station, time) so row i is this station
            discharge = mean_discharge[i, :].astype(np.float64)
            discharge[discharge == _FILL_VALUE] = np.nan

            lat = float(latitudes[i])
            lon = float(longitudes[i])
            area = float(areas[i])

            # Replace fill-valued coordinates with NaN
            if lat == _FILL_VALUE:
                lat = np.nan
            if lon == _FILL_VALUE:
                lon = np.nan
            if area == _FILL_VALUE or area < 0:
                area = np.nan

            station = StationTimeSeries(
                station_id=sid,
                discharge=discharge,
                time=time_vals,
                latitude=lat,
                longitude=lon,
                upstream_area=area,
            )

            meta = StationMetadata(
                name=sid,
                source_file="GSHA_monthly.nc",
                source_variable="mean",
                source_unit="m3/s",
                source_crs="WGS84",
            )

            # Stash extra info into the generic fields where appropriate
            if agencies[i]:
                meta.country = agencies[i]  # closest available field
            if verifications[i]:
                meta.river = verifications[i]  # reuse river field for verification tag

            stations.append(station)
            metadata[sid] = meta

        return stations, metadata

    @staticmethod
    def _get_string_var(ds: xr.Dataset, var_name: str, n: int) -> List[str]:
        """Safely get a string variable from a Dataset."""
        if var_name in ds:
            vals = ds[var_name].values
            return [str(v) if v is not None else "" for v in vals]
        return [""] * n
