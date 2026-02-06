"""Data model classes for Streamflow Pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class StationTimeSeries:
    """Single station's time series data, standardized."""
    station_id: str
    discharge: np.ndarray       # m3/s
    time: np.ndarray            # datetime64
    latitude: float             # WGS84
    longitude: float            # WGS84
    upstream_area: float        # km2

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.discharge)))


@dataclass
class StationMetadata:
    """Provenance and metadata for a single station."""
    name: str = ""
    river: str = ""
    country: str = ""
    elevation: float = np.nan
    source_file: str = ""
    source_variable: str = ""
    source_unit: str = ""
    source_crs: str = "WGS84"


@dataclass
class StationDataset:
    """One dataset's complete output from a reader."""
    source_name: str
    time_resolution: str         # "daily", "monthly", "hourly"
    timezone_type: str           # "utc", "fixed_offset", "hydrological_day", "local"
    timezone_utc_offset: float   # hours
    timezone_definition: str     # human-readable
    stations: List[StationTimeSeries] = field(default_factory=list)
    metadata: Dict[str, StationMetadata] = field(default_factory=dict)

    @property
    def station_count(self) -> int:
        return len(self.stations)
