"""CAMELS-ES reader for daily streamflow (Caravan-style NetCDF, mm/d -> m3/s)."""

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

# Sub-dataset prefix used in file/directory naming
CAMELSES_PREFIX = "camelses"


@register_reader("camels_es")
class CAMELSESReader(BaseReader):
    """Reader for CAMELS-ES dataset (269 stations in Spain, daily)."""

    source_name = "camels_es"

    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read all CAMELS-ES data. Returns one StationDataset (daily).

        Tries NetCDF first (timeseries/netcdf/camelses/); falls back to CSV
        if NetCDF directory is missing.
        """
        data_path = Path(config["source"]["path"])
        options = config.get("options", {})
        max_stations = options.get("max_stations", None)

        # Load station attributes
        attrs_df = self._load_attributes(data_path)
        if attrs_df is None:
            return []

        # Decide source: prefer NetCDF, fall back to CSV
        nc_dir = data_path / "timeseries" / "netcdf" / CAMELSES_PREFIX
        csv_dir = data_path / "timeseries" / "csv" / CAMELSES_PREFIX
        use_netcdf = nc_dir.exists() and any(nc_dir.glob("*.nc"))

        if use_netcdf:
            self.logger.info("Reading CAMELS-ES from NetCDF: %s", nc_dir)
        elif csv_dir.exists():
            self.logger.info(
                "NetCDF not available; falling back to CSV: %s", csv_dir
            )
        else:
            self.logger.error(
                "No timeseries directory found (checked %s and %s)",
                nc_dir, csv_dir,
            )
            return []

        stations: List[StationTimeSeries] = []
        metadata: Dict[str, StationMetadata] = {}

        count = 0
        for _, row in attrs_df.iterrows():
            gauge_id = str(row["gauge_id"])

            try:
                lat = float(row["gauge_lat"])
                lon = float(row["gauge_lon"])
                area_km2 = float(row["area"])
            except (ValueError, KeyError) as exc:
                self.logger.warning(
                    "Bad attributes for %s: %s", gauge_id, exc
                )
                continue

            stn: Optional[StationTimeSeries] = None
            source_file = ""

            if use_netcdf:
                stn, source_file = self._read_netcdf_station(
                    nc_dir, gauge_id, lat, lon, area_km2
                )
            else:
                stn, source_file = self._read_csv_station(
                    csv_dir, gauge_id, lat, lon, area_km2
                )

            if stn is None:
                continue

            meta = StationMetadata(
                name=str(row.get("gauge_name", "")),
                country=str(row.get("country", "")),
                source_file=source_file,
                source_variable="streamflow",
                source_unit="mm/d",
                source_crs="WGS84",
            )

            stations.append(stn)
            metadata[gauge_id] = meta
            count += 1

            if max_stations and count >= max_stations:
                break

        self.logger.info("CAMELS-ES: %d stations loaded", count)

        dataset = StationDataset(
            source_name=self.source_name,
            time_resolution="daily",
            timezone_type="fixed_offset",
            timezone_utc_offset=1.0,
            timezone_definition="CET (UTC+1, Spain)",
            stations=stations,
            metadata=metadata,
        )

        return [dataset]

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    def _load_attributes(self, data_path: Path) -> Optional[pd.DataFrame]:
        """Load station attributes from attributes_other_camelses.csv."""
        attrs_csv = (
            data_path / "attributes" / CAMELSES_PREFIX
            / f"attributes_other_{CAMELSES_PREFIX}.csv"
        )
        if not attrs_csv.exists():
            self.logger.error("Attributes CSV not found: %s", attrs_csv)
            return None

        attrs_df = pd.read_csv(attrs_csv)
        self.logger.info("Loaded %d station attributes", len(attrs_df))
        return attrs_df

    # ------------------------------------------------------------------
    # NetCDF reader (preferred)
    # ------------------------------------------------------------------

    def _read_netcdf_station(
        self,
        nc_dir: Path,
        gauge_id: str,
        lat: float,
        lon: float,
        area_km2: float,
    ) -> tuple:
        """Read a single station from NetCDF. Returns (StationTimeSeries, filename) or (None, "")."""
        nc_file = nc_dir / f"{gauge_id}.nc"
        if not nc_file.exists():
            self.logger.debug("NetCDF not found for %s", gauge_id)
            return None, ""

        try:
            ds = xr.open_dataset(nc_file)
            streamflow_mmd = ds["streamflow"].values.astype(np.float64)
            time_vals = ds["date"].values
            ds.close()
        except Exception as exc:
            self.logger.warning("Error reading NetCDF for %s: %s", gauge_id, exc)
            return None, ""

        # Convert mm/d to m3/s
        discharge_m3s = convert_mmd_to_m3s(streamflow_mmd, area_km2)

        stn = StationTimeSeries(
            station_id=gauge_id,
            discharge=discharge_m3s,
            time=time_vals,
            latitude=lat,
            longitude=lon,
            upstream_area=area_km2,
        )
        return stn, nc_file.name

    # ------------------------------------------------------------------
    # CSV reader (fallback)
    # ------------------------------------------------------------------

    def _read_csv_station(
        self,
        csv_dir: Path,
        gauge_id: str,
        lat: float,
        lon: float,
        area_km2: float,
    ) -> tuple:
        """Read a single station from CSV. Returns (StationTimeSeries, filename) or (None, "")."""
        csv_file = csv_dir / f"{gauge_id}.csv"
        if not csv_file.exists():
            self.logger.debug("CSV not found for %s", gauge_id)
            return None, ""

        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            self.logger.warning("Error reading CSV for %s: %s", gauge_id, exc)
            return None, ""

        if "date" not in df.columns or "streamflow" not in df.columns:
            self.logger.warning(
                "Missing expected columns in %s: %s", csv_file, list(df.columns)
            )
            return None, ""

        dates = pd.to_datetime(df["date"], format="%Y-%m-%d")
        streamflow_mmd = df["streamflow"].values.astype(np.float64)

        valid_count = int(np.count_nonzero(~np.isnan(streamflow_mmd)))
        if valid_count == 0:
            self.logger.warning(
                "Station %s: streamflow column is entirely NaN in CSV", gauge_id
            )

        discharge_m3s = convert_mmd_to_m3s(streamflow_mmd, area_km2)

        stn = StationTimeSeries(
            station_id=gauge_id,
            discharge=discharge_m3s,
            time=dates.values,
            latitude=lat,
            longitude=lon,
            upstream_area=area_km2,
        )
        return stn, csv_file.name
