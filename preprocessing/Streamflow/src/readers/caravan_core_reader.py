"""Caravan-core reader for 5 sub-datasets."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import register_reader
from .base_reader import BaseReader
from ..models import StationTimeSeries, StationMetadata, StationDataset
from ..utils.unit_converter import convert_mmd_to_m3s

# Only these 5 sub-datasets from Caravan-core
CARAVAN_SUBDATASETS = ["camels", "camelsaus", "camelscl", "hysets", "lamah"]


@register_reader("caravan_core")
class CaravanCoreReader(BaseReader):
    """Reader for Caravan-core dataset (5 sub-datasets, ~14758 stations)."""

    source_name = "caravan_core"

    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read all Caravan-core data. Returns one StationDataset (daily)."""
        data_path = Path(config["source"]["path"])
        options = config.get("options", {})
        subdatasets = options.get("subdatasets", CARAVAN_SUBDATASETS)
        max_stations = options.get("max_stations", None)

        all_stations: List[StationTimeSeries] = []
        all_metadata: Dict[str, StationMetadata] = {}

        for subdataset in subdatasets:
            if subdataset not in CARAVAN_SUBDATASETS:
                self.logger.warning(f"Skipping unknown subdataset: {subdataset}")
                continue

            self.logger.info(f"Reading Caravan-core subdataset: {subdataset}")

            # Read attributes
            attrs_path = (
                data_path / "attributes" / subdataset
                / f"attributes_other_{subdataset}.csv"
            )
            if not attrs_path.exists():
                self.logger.warning(f"Attributes not found: {attrs_path}")
                continue
            attrs_df = pd.read_csv(attrs_path)

            # Read per-station NetCDFs
            nc_dir = data_path / "timeseries" / "netcdf" / subdataset
            if not nc_dir.exists():
                self.logger.warning(f"NetCDF directory not found: {nc_dir}")
                continue

            count = 0
            for _, row in attrs_df.iterrows():
                gauge_id = str(row["gauge_id"])
                nc_file = nc_dir / f"{gauge_id}.nc"

                if not nc_file.exists():
                    self.logger.debug(f"NetCDF not found for {gauge_id}")
                    continue

                try:
                    ds = xr.open_dataset(nc_file)

                    # Extract streamflow in mm/d
                    streamflow_mmd = ds["streamflow"].values.astype(np.float64)
                    time_vals = ds["date"].values

                    # Get timezone from global attributes
                    tz_str = ds.attrs.get("Timezone", "UTC")

                    ds.close()

                    # Get station attributes
                    lat = float(row["gauge_lat"])
                    lon = float(row["gauge_lon"])
                    area_km2 = float(row["area"])

                    # Convert mm/d to m3/s
                    discharge_m3s = convert_mmd_to_m3s(streamflow_mmd, area_km2)

                    station = StationTimeSeries(
                        station_id=gauge_id,
                        discharge=discharge_m3s,
                        time=time_vals,
                        latitude=lat,
                        longitude=lon,
                        upstream_area=area_km2,
                    )

                    metadata = StationMetadata(
                        name=str(row.get("gauge_name", "")),
                        country=str(row.get("country", "")),
                        source_file=str(nc_file.name),
                        source_variable="streamflow",
                        source_unit="mm/d",
                        source_crs="WGS84",
                    )

                    all_stations.append(station)
                    all_metadata[gauge_id] = metadata
                    count += 1

                    if max_stations and count >= max_stations:
                        break

                except Exception as e:
                    self.logger.warning(f"Error reading {gauge_id}: {e}")
                    continue

            self.logger.info(f"  {subdataset}: {count} stations loaded")

            if max_stations and len(all_stations) >= max_stations:
                break

        self.logger.info(f"Total Caravan-core stations: {len(all_stations)}")

        dataset = StationDataset(
            source_name=self.source_name,
            time_resolution="daily",
            timezone_type="local",
            timezone_utc_offset=0.0,
            timezone_definition="Per-station local timezone (see :Timezone attribute)",
            stations=all_stations,
            metadata=all_metadata,
        )

        return [dataset]
