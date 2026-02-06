"""CAMELS_FR reader for daily and monthly streamflow (semicolon CSV, L/s)."""

import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import register_reader
from .base_reader import BaseReader
from ..models import StationTimeSeries, StationMetadata, StationDataset
from ..constants import DISCHARGE_CONVERSIONS
from ..utils.unit_converter import convert_mmd_to_m3s


@register_reader("camels_fr")
class CAMELSFRReader(BaseReader):
    """Reader for CAMELS_FR dataset (654 stations, daily + monthly CSV)."""

    source_name = "camels_fr"

    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read CAMELS_FR data. Returns up to 2 StationDatasets (daily, monthly)."""
        data_path = Path(config["source"]["path"])
        options = config.get("options", {})
        max_stations = options.get("max_stations", None)
        resolutions = options.get("resolutions", ["daily", "monthly"])

        # Ensure attributes are extracted
        attrs_df = self._load_attributes(data_path)
        if attrs_df is None:
            return []

        datasets: List[StationDataset] = []

        for resolution in resolutions:
            if resolution not in ("daily", "monthly"):
                self.logger.warning("Skipping unsupported resolution: %s", resolution)
                continue

            stations, metadata = self._read_resolution(
                data_path, attrs_df, resolution, max_stations
            )

            if stations:
                ds = StationDataset(
                    source_name=self.source_name,
                    time_resolution=resolution,
                    timezone_type="hydrological_day",
                    timezone_utc_offset=1.0,
                    timezone_definition="Hydrological day (UTC+1, France)",
                    stations=stations,
                    metadata=metadata,
                )
                datasets.append(ds)
                self.logger.info(
                    "CAMELS_FR %s: %d stations", resolution, len(stations)
                )

        return datasets

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    def _load_attributes(self, data_path: Path) -> Optional[pd.DataFrame]:
        """Load station attributes, extracting the ZIP if necessary.

        The attributes directory structure (from the ZIP or pre-extracted) is:
            CAMELS_FR_attributes/static_attributes/CAMELS_FR_station_general_attributes.csv
        """
        static_dir = data_path / "CAMELS_FR_attributes" / "static_attributes"

        if not static_dir.exists():
            zip_path = data_path / "CAMELS_FR_attributes.zip"
            if not zip_path.exists():
                self.logger.error(
                    "Neither CAMELS_FR_attributes/ nor CAMELS_FR_attributes.zip "
                    "found in %s",
                    data_path,
                )
                return None
            self.logger.info("Extracting %s ...", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_path)

        attrs_csv = static_dir / "CAMELS_FR_station_general_attributes.csv"
        if not attrs_csv.exists():
            self.logger.error("Attributes CSV not found: %s", attrs_csv)
            return None

        attrs_df = pd.read_csv(attrs_csv, sep=";")
        self.logger.info("Loaded %d station attributes", len(attrs_df))
        return attrs_df

    # ------------------------------------------------------------------
    # Per-resolution reading
    # ------------------------------------------------------------------

    def _read_resolution(
        self,
        data_path: Path,
        attrs_df: pd.DataFrame,
        resolution: str,
        max_stations: Optional[int],
    ):
        """Read all stations for one resolution (daily or monthly)."""
        stations: List[StationTimeSeries] = []
        metadata: Dict[str, StationMetadata] = {}

        ts_dir = data_path / "CAMELS_FR_time_series" / resolution
        if not ts_dir.exists():
            self.logger.warning("Time-series directory not found: %s", ts_dir)
            return stations, metadata

        # Build lookup from station code to attributes row
        attrs_lookup = {
            str(row["sta_code_h3"]): row for _, row in attrs_df.iterrows()
        }

        prefix = "tsd" if resolution == "daily" else "tsm"
        csv_files = sorted(ts_dir.glob(f"CAMELS_FR_{prefix}_*.csv"))

        count = 0
        for csv_file in csv_files:
            # Extract station ID from filename: CAMELS_FR_tsd_{STAID}.csv
            station_id = csv_file.stem.replace(f"CAMELS_FR_{prefix}_", "")

            attrs_row = attrs_lookup.get(station_id)
            if attrs_row is None:
                self.logger.debug(
                    "No attributes for station %s, skipping", station_id
                )
                continue

            try:
                lat = float(attrs_row["sta_y_w84"])
                lon = float(attrs_row["sta_x_w84"])
                area_km2 = float(attrs_row["sta_area_snap"])
                station_name = str(attrs_row.get("sta_label", ""))
                country = str(attrs_row.get("sta_territory", ""))
            except (ValueError, KeyError) as e:
                self.logger.warning(
                    "Bad attributes for station %s: %s", station_id, e
                )
                continue

            try:
                if resolution == "daily":
                    stn = self._read_daily_csv(
                        csv_file, station_id, lat, lon, area_km2
                    )
                    source_variable = "tsd_q_l"
                    source_unit = "L/s"
                else:
                    stn = self._read_monthly_csv(
                        csv_file, station_id, lat, lon, area_km2
                    )
                    source_variable = "tsm_q_mm"
                    source_unit = "mm"

                if stn is None:
                    continue

                meta = StationMetadata(
                    name=station_name,
                    country=country,
                    source_file=csv_file.name,
                    source_variable=source_variable,
                    source_unit=source_unit,
                    source_crs="WGS84",
                )

                stations.append(stn)
                metadata[station_id] = meta
                count += 1

                if max_stations and count >= max_stations:
                    break

            except Exception as e:
                self.logger.warning(
                    "Error reading %s for station %s: %s",
                    resolution, station_id, e,
                )
                continue

        return stations, metadata

    # ------------------------------------------------------------------
    # Daily CSV
    # ------------------------------------------------------------------

    def _read_daily_csv(
        self,
        csv_file: Path,
        station_id: str,
        lat: float,
        lon: float,
        area_km2: float,
    ) -> Optional[StationTimeSeries]:
        """Read a daily time-series CSV. Discharge in L/s -> m3/s."""
        df = pd.read_csv(
            csv_file,
            sep=";",
            skiprows=7,
            na_values=["NA"],
        )

        if "tsd_date" not in df.columns or "tsd_q_l" not in df.columns:
            self.logger.warning(
                "Missing expected columns in %s: %s", csv_file, list(df.columns)
            )
            return None

        dates = pd.to_datetime(df["tsd_date"], format="%Y%m%d")
        discharge_ls = df["tsd_q_l"].values.astype(np.float64)

        # Convert L/s to m3/s
        discharge_m3s = discharge_ls * DISCHARGE_CONVERSIONS["L/s"]

        return StationTimeSeries(
            station_id=station_id,
            discharge=discharge_m3s,
            time=dates.values,
            latitude=lat,
            longitude=lon,
            upstream_area=area_km2,
        )

    # ------------------------------------------------------------------
    # Monthly CSV
    # ------------------------------------------------------------------

    def _read_monthly_csv(
        self,
        csv_file: Path,
        station_id: str,
        lat: float,
        lon: float,
        area_km2: float,
    ) -> Optional[StationTimeSeries]:
        """Read a monthly time-series CSV. Discharge in mm (monthly total) -> m3/s."""
        df = pd.read_csv(
            csv_file,
            sep=";",
            skiprows=7,
            na_values=["NA"],
        )

        if "tsm_date" not in df.columns or "tsm_q_mm" not in df.columns:
            self.logger.warning(
                "Missing expected columns in %s: %s", csv_file, list(df.columns)
            )
            return None

        # Parse YYYYMM to datetime (first day of month)
        dates = pd.to_datetime(df["tsm_date"].astype(str), format="%Y%m")
        discharge_mm = df["tsm_q_mm"].values.astype(np.float64)

        # Convert monthly total mm to mm/d by dividing by days in month
        days_in_month = dates.dt.days_in_month.values.astype(np.float64)
        discharge_mmd = discharge_mm / days_in_month

        # Convert mm/d to m3/s
        discharge_m3s = convert_mmd_to_m3s(discharge_mmd, area_km2)

        return StationTimeSeries(
            station_id=station_id,
            discharge=discharge_m3s,
            time=dates.values,
            latitude=lat,
            longitude=lon,
            upstream_area=area_km2,
        )
