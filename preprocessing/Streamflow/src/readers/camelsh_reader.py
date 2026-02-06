"""CAMELSH reader for hourly per-station NetCDFs inside a ZIP archive."""

import csv
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from . import register_reader
from .base_reader import BaseReader
from ..models import StationTimeSeries, StationMetadata, StationDataset


@register_reader("camelsh")
class CAMELSHReader(BaseReader):
    """Reader for CAMELSH dataset (hourly streamflow, per-station NC in ZIP).

    Data layout::

        CAMELSH/
        +-- attributes_gageii_BasinID.csv   (9,008 stations)
        +-- Hourly2.zip                     (per-station NetCDFs)

    Each NetCDF inside the ZIP is named ``Hourly2/{STAID}_hourly.nc`` and
    contains an hourly ``streamflow`` variable already in m3/s.
    """

    source_name = "camelsh"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read CAMELSH data.  Returns one StationDataset (hourly)."""
        data_path = Path(config["source"]["path"])
        options = config.get("options", {})
        max_stations: Optional[int] = options.get("max_stations", None)

        # 1. Read station attributes CSV
        attrs_file = data_path / "attributes_gageii_BasinID.csv"
        if not attrs_file.exists():
            self.logger.error("Attributes CSV not found: %s", attrs_file)
            return []
        attrs = self._read_attributes(attrs_file)
        self.logger.info("CAMELSH attributes: %d stations", len(attrs))

        # 2. Open the ZIP and iterate per-station NetCDFs
        zip_path = data_path / "Hourly2.zip"
        if not zip_path.exists():
            self.logger.error("ZIP archive not found: %s", zip_path)
            return []

        all_stations, all_metadata = self._read_stations_from_zip(
            zip_path, attrs, max_stations
        )

        if not all_stations:
            self.logger.warning("No stations loaded from CAMELSH")
            return []

        dataset = StationDataset(
            source_name=self.source_name,
            time_resolution="hourly",
            timezone_type="local",
            timezone_utc_offset=0.0,
            timezone_definition="US local time zones (varies by station)",
            stations=all_stations,
            metadata=all_metadata,
        )

        self.logger.info("CAMELSH: %d stations loaded", len(all_stations))
        return [dataset]

    # ------------------------------------------------------------------
    # Attributes CSV
    # ------------------------------------------------------------------

    @staticmethod
    def _read_attributes(attrs_file: Path) -> Dict[str, Dict]:
        """Parse the attributes CSV into a dict keyed by zero-padded STAID.

        The STANAME field may contain commas and is double-quoted, so we use
        the ``csv`` module with ``quotechar`` handling.
        """
        attrs: Dict[str, Dict] = {}
        with open(attrs_file, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, quotechar='"')
            for row in reader:
                staid = str(row["STAID"]).strip().zfill(8)
                attrs[staid] = {
                    "name": row.get("STANAME", "").strip(),
                    "lat": float(row["LAT_GAGE"]),
                    "lon": float(row["LNG_GAGE"]),
                    "area_km2": float(row["DRAIN_SQKM"]),
                }
        return attrs

    # ------------------------------------------------------------------
    # ZIP / NetCDF reading
    # ------------------------------------------------------------------

    def _read_stations_from_zip(
        self,
        zip_path: Path,
        attrs: Dict[str, Dict],
        max_stations: Optional[int],
    ):
        """Extract and read per-station NetCDFs one at a time from the ZIP."""
        all_stations: List[StationTimeSeries] = []
        all_metadata: Dict[str, StationMetadata] = {}

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                nc_members = [
                    name for name in zf.namelist()
                    if name.endswith("_hourly.nc")
                ]
                self.logger.info(
                    "CAMELSH ZIP contains %d NetCDF files", len(nc_members)
                )

                count = 0
                for member in nc_members:
                    # Derive STAID from filename, e.g. "Hourly2/01011000_hourly.nc"
                    basename = member.rsplit("/", 1)[-1]  # "01011000_hourly.nc"
                    staid = basename.replace("_hourly.nc", "").zfill(8)

                    if staid not in attrs:
                        self.logger.debug(
                            "Station %s not in attributes CSV, skipping", staid
                        )
                        continue

                    try:
                        stn, meta = self._read_one_station(
                            zf, member, staid, attrs[staid]
                        )
                        all_stations.append(stn)
                        all_metadata[staid] = meta
                        count += 1

                        if count % 500 == 0:
                            self.logger.info("  ... %d stations read", count)

                        if max_stations and count >= max_stations:
                            self.logger.info(
                                "Reached max_stations=%d, stopping", max_stations
                            )
                            break
                    except Exception as e:
                        self.logger.warning(
                            "Error reading station %s: %s", staid, e
                        )
                        continue

        except zipfile.BadZipFile:
            self.logger.error("Bad ZIP file: %s", zip_path)

        return all_stations, all_metadata

    def _read_one_station(
        self,
        zf: zipfile.ZipFile,
        member: str,
        staid: str,
        attr: Dict,
    ):
        """Extract a single NetCDF from the ZIP, read it, and return data.

        Uses a temporary directory so the extracted file is cleaned up
        immediately after reading.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract(member, tmpdir)
            extracted_path = Path(tmpdir) / member

            ds = xr.open_dataset(str(extracted_path))
            ds.load()  # Load into memory before tmpdir is removed

            # Time: "hours since 1980-01-01 00:00:00"
            time_vals = ds["time"].values

            # Streamflow already in m3/s
            discharge = ds["streamflow"].values.astype(np.float64)

            ds.close()

        # Replace NaN / fill values
        discharge = np.where(np.isfinite(discharge), discharge, np.nan)

        station = StationTimeSeries(
            station_id=staid,
            discharge=discharge,
            time=time_vals,
            latitude=attr["lat"],
            longitude=attr["lon"],
            upstream_area=attr["area_km2"],
        )

        metadata = StationMetadata(
            name=attr["name"],
            source_file=f"{staid}_hourly.nc",
            source_variable="streamflow",
            source_unit="m3/s",
            source_crs="WGS84",
        )

        return station, metadata
