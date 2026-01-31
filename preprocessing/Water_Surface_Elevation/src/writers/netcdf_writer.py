# src/writers/netcdf_writer.py
"""NetCDF writer for WSE station data with time series."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import numpy as np

from ..core.station import Station, StationList
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NetCDFWriter:
    """Write station data to CF-1.8 compliant NetCDF file."""

    SOURCE_CODES = {
        'hydroweb': 1,
        'cgls': 2,
        'icesat': 3,
        'icesat2': 4,
        'hydrosat': 5,
    }

    RESOLUTIONS = ['01min', '03min', '05min', '06min', '15min']

    def __init__(self, config: dict):
        """
        Initialize NetCDF writer.

        Args:
            config: Output configuration dict with keys:
                - netcdf_file: Output file path
                - time_reference: Time reference date (default: 1800-01-01)
                - min_uparea: Minimum upstream area filter (default: 100.0 km²)
                - chunk_size: Batch size for processing (default: 1000)
        """
        self.output_path = Path(config.get('netcdf_file', 'OpenBench_WSE.nc'))
        self.time_ref = config.get('time_reference', '1800-01-01')
        self.min_uparea = config.get('min_uparea', 100.0)
        self.chunk_size = config.get('chunk_size', 1000)

        # Parse time reference
        self.time_ref_date = datetime.strptime(self.time_ref, '%Y-%m-%d').date()

        # Time range (will be set from config or auto-detected)
        self.time_start = self._parse_date(config.get('time_start', '1995-01-01'))
        self.time_end = self._parse_date(config.get('time_end', '2024-12-31'))

        # Readers will be initialized lazily
        self._readers = {}

    def _filter_stations(self, stations: StationList) -> List[Station]:
        """
        Filter stations by upstream area and valid CaMa allocation.

        A station passes if ANY resolution has:
        - flag > 0 (valid allocation)
        - uparea > min_uparea

        Args:
            stations: Input station list

        Returns:
            List of stations passing the filter
        """
        filtered = []

        for station in stations:
            if self._station_passes_filter(station):
                filtered.append(station)

        logger.info(f"Filtered stations: {len(filtered)}/{len(stations)} "
                   f"(min_uparea={self.min_uparea} km2)")

        return filtered

    def _station_passes_filter(self, station: Station) -> bool:
        """Check if station passes upstream area filter."""
        for res in self.RESOLUTIONS:
            cama = station.cama_results.get(f'glb_{res}', {})
            flag = cama.get('flag', 0)
            uparea = cama.get('uparea', 0)

            if flag > 0 and uparea > self.min_uparea:
                return True

        return False

    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object."""
        return datetime.strptime(date_str, '%Y-%m-%d').date()

    def _date_to_days(self, d: date) -> int:
        """Convert date to days since time reference."""
        delta = d - self.time_ref_date
        return delta.days

    def _build_time_axis(self) -> tuple:
        """
        Build unified time axis.

        Returns:
            Tuple of (dates_list, days_values_array)
        """
        from datetime import timedelta

        dates = []
        current = self.time_start
        while current <= self.time_end:
            dates.append(current)
            current += timedelta(days=1)

        days_values = np.array([self._date_to_days(d) for d in dates], dtype=np.int64)

        return dates, days_values

    def write(self, stations: StationList, data_paths: dict) -> Path:
        """
        Write stations to NetCDF file.

        Args:
            stations: Processed station list with CaMa results
            data_paths: Dict mapping source names to data directories

        Returns:
            Path to output file
        """
        raise NotImplementedError("NetCDFWriter.write() not yet implemented")
