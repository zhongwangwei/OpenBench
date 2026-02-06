"""CAMELS-DK reader for Danish catchment data (304 gauged stations)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import register_reader
from .base_reader import BaseReader
from ..models import StationTimeSeries, StationMetadata, StationDataset
from ..utils.unit_converter import convert_mmd_to_m3s
from ..utils.crs_converter import reproject_to_wgs84

# Source CRS for CAMELS-DK coordinates (UTM zone 32N)
SOURCE_CRS = "EPSG:25832"


@register_reader("camels_dk")
class CamelsDKReader(BaseReader):
    """Reader for CAMELS-DK dataset (304 gauged catchments, Denmark)."""

    source_name = "camels_dk"

    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read all CAMELS-DK gauged data. Returns one StationDataset (daily)."""
        data_path = Path(config["source"]["path"])
        options = config.get("options", {})
        max_stations = options.get("max_stations", None)

        # Read attributes from topography CSV (note: folder has typo "Attibutes")
        topo_path = data_path / "Attibutes" / "CAMELS_DK_topography.csv"
        if not topo_path.exists():
            self.logger.error("Topography file not found: %s", topo_path)
            return []

        topo_df = pd.read_csv(topo_path)
        self.logger.info(
            "Loaded topography attributes: %d rows", len(topo_df)
        )

        # Build lookup: catch_id -> (easting, northing, area_m2)
        attrs = {}
        for _, row in topo_df.iterrows():
            catch_id = str(int(row["catch_id"]))
            attrs[catch_id] = {
                "easting": float(row["catch_outlet_lon"]),
                "northing": float(row["catch_outlet_lat"]),
                "area_m2": float(row["catch_area"]),
            }

        # Read gauged timeseries from Dynamics/Gauged_catchments/
        gauged_dir = data_path / "Dynamics" / "Gauged_catchments"
        if not gauged_dir.exists():
            self.logger.error(
                "Gauged catchments directory not found: %s", gauged_dir
            )
            return []

        csv_files = sorted(gauged_dir.glob("CAMELS_DK_obs_based_*.csv"))
        self.logger.info(
            "Found %d gauged timeseries files", len(csv_files)
        )

        all_stations: List[StationTimeSeries] = []
        all_metadata: Dict[str, StationMetadata] = {}
        count = 0

        for csv_file in csv_files:
            # Extract station ID from filename: CAMELS_DK_obs_based_{STAID}.csv
            station_id = csv_file.stem.replace("CAMELS_DK_obs_based_", "")

            if station_id not in attrs:
                self.logger.warning(
                    "No attributes found for station %s, skipping",
                    station_id,
                )
                continue

            try:
                station, metadata = self._read_station(
                    csv_file, station_id, attrs[station_id]
                )
                all_stations.append(station)
                all_metadata[station_id] = metadata
                count += 1

                if max_stations and count >= max_stations:
                    break

            except Exception as e:
                self.logger.warning(
                    "Error reading station %s: %s", station_id, e
                )
                continue

        self.logger.info("Total CAMELS-DK stations loaded: %d", count)

        dataset = StationDataset(
            source_name=self.source_name,
            time_resolution="daily",
            timezone_type="fixed_offset",
            timezone_utc_offset=1.0,
            timezone_definition="CET (UTC+1)",
            stations=all_stations,
            metadata=all_metadata,
        )

        return [dataset]

    def _read_station(
        self,
        csv_file: Path,
        station_id: str,
        station_attrs: Dict,
    ) -> tuple:
        """Read a single gauged station CSV and return (StationTimeSeries, StationMetadata)."""
        # Read timeseries CSV
        df = pd.read_csv(csv_file)

        # Parse time column
        time_vals = pd.to_datetime(df["time"]).values

        # Extract discharge (Qobs) in mm/day; empty strings become NaN
        discharge_mmd = df["Qobs"].values.astype(np.float64)

        # Convert area from m2 to km2
        area_km2 = station_attrs["area_m2"] / 1e6

        # Convert discharge from mm/d to m3/s
        discharge_m3s = convert_mmd_to_m3s(discharge_mmd, area_km2)

        # Reproject coordinates from EPSG:25832 to WGS84
        lon, lat = reproject_to_wgs84(
            station_attrs["easting"],
            station_attrs["northing"],
            SOURCE_CRS,
        )

        station = StationTimeSeries(
            station_id=station_id,
            discharge=discharge_m3s,
            time=time_vals,
            latitude=float(lat),
            longitude=float(lon),
            upstream_area=area_km2,
        )

        metadata = StationMetadata(
            name="",
            country="DK",
            source_file=csv_file.name,
            source_variable="Qobs",
            source_unit="mm/d",
            source_crs=SOURCE_CRS,
        )

        return station, metadata
