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

    def _create_netcdf(self, stations: List[Station], time_axis: List[date],
                       time_values: np.ndarray) -> None:
        """Create NetCDF file with dimensions and variables."""
        import netCDF4 as nc

        n_stations = len(stations)
        n_times = len(time_axis)

        logger.info(f"创建 NetCDF: {n_stations} 站点, {n_times} 时间点")

        ds = nc.Dataset(str(self.output_path), 'w', format='NETCDF4')

        try:
            # Global attributes
            ds.title = "OpenBench Water Surface Elevation Dataset"
            ds.institution = "OpenBench"
            ds.source = "HydroWeb, CGLS, ICESat, ICESat-2, HydroSat"
            ds.history = f"Created {datetime.now().isoformat()}"
            ds.Conventions = "CF-1.8"
            ds.references = "https://github.com/zhongwangwei/OpenBench"

            # Dimensions
            ds.createDimension('station', n_stations)
            ds.createDimension('time', n_times)

            # Time coordinate
            time_var = ds.createVariable('time', 'i8', ('time',))
            time_var.long_name = "Time"
            time_var.standard_name = "time"
            time_var.units = f"days since {self.time_ref}"
            time_var.calendar = "gregorian"
            time_var[:] = time_values

            # Station index
            station_idx = ds.createVariable('station', 'i8', ('station',))
            station_idx.long_name = "Station index"
            station_idx[:] = np.arange(n_stations)

            # Time series data
            wse = ds.createVariable('wse', 'f4', ('station', 'time'),
                                   fill_value=np.nan, zlib=True, complevel=4)
            wse.long_name = "Water surface elevation"
            wse.units = "m"
            wse.standard_name = "water_surface_height_above_reference_datum"

            data_source = ds.createVariable('data_source', 'i1', ('station', 'time'),
                                           fill_value=-1, zlib=True, complevel=4)
            data_source.long_name = "Data source code"
            data_source.flag_values = np.array([1, 2, 3, 4, 5], dtype=np.int8)
            data_source.flag_meanings = "HydroWeb CGLS ICESat ICESat-2 HydroSat"

            # Station metadata variables
            lat = ds.createVariable('lat', 'f8', ('station',), fill_value=np.nan)
            lat.long_name = "Latitude"
            lat.units = "degrees_north"
            lat.standard_name = "latitude"

            lon = ds.createVariable('lon', 'f8', ('station',), fill_value=np.nan)
            lon.long_name = "Longitude"
            lon.units = "degrees_east"
            lon.standard_name = "longitude"

            station_id = ds.createVariable('station_id', str, ('station',))
            station_id.long_name = "Station identifier"

            station_name = ds.createVariable('station_name', str, ('station',))
            station_name.long_name = "Station name"

            elevation = ds.createVariable('elevation', 'f4', ('station',), fill_value=np.nan)
            elevation.long_name = "Mean water surface elevation"
            elevation.units = "m"

            num_obs = ds.createVariable('num_observations', 'i4', ('station',))
            num_obs.long_name = "Number of observations"

            source_var = ds.createVariable('source', str, ('station',))
            source_var.long_name = "Data source name"

            egm08 = ds.createVariable('EGM08', 'f4', ('station',), fill_value=np.nan)
            egm08.long_name = "EGM2008 geoid height"
            egm08.units = "m"

            egm96 = ds.createVariable('EGM96', 'f4', ('station',), fill_value=np.nan)
            egm96.long_name = "EGM96 geoid height"
            egm96.units = "m"

            # CaMa allocation results for each resolution
            for res in self.RESOLUTIONS:
                self._create_cama_variables(ds, res)

            # Write station metadata
            for i, station in enumerate(stations):
                lat[i] = station.lat
                lon[i] = station.lon
                station_id[i] = station.id
                station_name[i] = station.name
                elevation[i] = station.elevation
                num_obs[i] = station.num_observations
                source_var[i] = station.source
                egm08[i] = station.egm08 if station.egm08 is not None else np.nan
                egm96[i] = station.egm96 if station.egm96 is not None else np.nan

                # Write CaMa results
                self._write_cama_results(ds, i, station)

        finally:
            ds.close()

        logger.info(f"NetCDF 结构创建完成: {self.output_path}")

    def _create_cama_variables(self, ds, res: str) -> None:
        """Create CaMa-related variables for a resolution."""
        cama_lat = ds.createVariable(f'cama_lat_{res}', 'f4', ('station',), fill_value=np.nan)
        cama_lat.long_name = f"CaMa-Flood grid cell latitude ({res} resolution)"
        cama_lat.units = "degrees_north"

        cama_lon = ds.createVariable(f'cama_lon_{res}', 'f4', ('station',), fill_value=np.nan)
        cama_lon.long_name = f"CaMa-Flood grid cell longitude ({res} resolution)"
        cama_lon.units = "degrees_east"

        cama_flag = ds.createVariable(f'cama_flag_{res}', 'i1', ('station',), fill_value=-1)
        cama_flag.long_name = f"CaMa allocation flag ({res} resolution)"

        cama_uparea = ds.createVariable(f'cama_uparea_{res}', 'f4', ('station',), fill_value=np.nan)
        cama_uparea.long_name = f"CaMa-Flood upstream area ({res} resolution)"
        cama_uparea.units = "km2"

    def _write_cama_results(self, ds, idx: int, station: Station) -> None:
        """Write CaMa results for a station."""
        for res in self.RESOLUTIONS:
            cama = station.cama_results.get(f'glb_{res}', {})

            ds.variables[f'cama_lat_{res}'][idx] = cama.get('lat_cama', np.nan)
            ds.variables[f'cama_lon_{res}'][idx] = cama.get('lon_cama', np.nan)
            ds.variables[f'cama_flag_{res}'][idx] = cama.get('flag', -1)
            ds.variables[f'cama_uparea_{res}'][idx] = cama.get('uparea', np.nan)

    def _write_station_timeseries(self, station_idx: int, station: Station,
                                  timeseries: List[Dict], time_index: Dict[date, int]) -> None:
        """
        Write time series data for a single station.

        Args:
            station_idx: Station index in NetCDF
            station: Station object
            timeseries: List of {datetime, elevation, ...} dicts
            time_index: Dict mapping date to time axis index
        """
        import netCDF4 as nc

        source_code = self.SOURCE_CODES.get(station.source, -1)

        with nc.Dataset(str(self.output_path), 'a') as ds:
            wse_var = ds.variables['wse']
            source_var = ds.variables['data_source']

            for obs in timeseries:
                dt = obs.get('datetime')
                if dt is None:
                    continue

                # Convert datetime to date if needed
                if isinstance(dt, datetime):
                    dt = dt.date()

                if dt not in time_index:
                    continue  # Outside time range

                t_idx = time_index[dt]
                elev = obs.get('elevation')

                if elev is not None:
                    wse_var[station_idx, t_idx] = elev
                    source_var[station_idx, t_idx] = source_code

    def _get_reader(self, source: str):
        """Get or create reader for a data source."""
        if source not in self._readers:
            if source == 'hydroweb':
                from ..readers.hydroweb_reader import HydroWebReader
                self._readers[source] = HydroWebReader()
            elif source == 'cgls':
                from ..readers.cgls_reader import CGLSReader
                self._readers[source] = CGLSReader()
            elif source in ('icesat', 'icesat2'):
                from ..readers.icesat_reader import ICESatReader
                self._readers[source] = ICESatReader()
            elif source == 'hydrosat':
                from ..readers.hydrosat_reader import HydroSatReader
                self._readers[source] = HydroSatReader()
            else:
                return None

        return self._readers.get(source)

    def _read_station_timeseries(self, station: Station, data_paths: dict) -> List[Dict]:
        """
        Read time series for a station from source file.

        Args:
            station: Station object (must have filepath in metadata)
            data_paths: Dict mapping source names to data directories

        Returns:
            List of time series observations
        """
        reader = self._get_reader(station.source)
        if reader is None:
            logger.warning(f"No reader for source: {station.source}")
            return []

        # Get filepath from metadata or reconstruct
        filepath = station.metadata.get('filepath')
        if not filepath:
            # Try to reconstruct filepath
            data_dir = data_paths.get(station.source)
            if not data_dir:
                logger.warning(f"No data path for source: {station.source}")
                return []
            filepath = self._find_station_file(station, data_dir)

        if not filepath or not Path(filepath).exists():
            logger.warning(f"File not found for station {station.id}")
            return []

        return reader.read_timeseries(filepath)

    def _find_station_file(self, station: Station, data_dir: str) -> Optional[str]:
        """Try to find the source file for a station."""
        data_path = Path(data_dir)

        # Different naming patterns for each source
        source = station.source

        if source == 'hydroweb':
            # HydroWeb: hydroweb_river/R_*.txt
            pattern = f"**/*{station.id}*.txt"
        elif source == 'cgls':
            # CGLS: *.geojson
            pattern = f"**/*{station.id}*.geojson"
        elif source in ('icesat', 'icesat2'):
            # ICESat: station ID is usually in filename
            pattern = f"**/*{station.id}*"
        elif source == 'hydrosat':
            # HydroSat: WL_hydrosat/*.txt
            pattern = f"**/*{station.id}*.txt"
        else:
            return None

        matches = list(data_path.glob(pattern))
        if matches:
            return str(matches[0])

        return None

    def write(self, stations: StationList, data_paths: dict) -> Path:
        """
        Write stations to NetCDF file.

        Args:
            stations: Processed station list with CaMa results
            data_paths: Dict mapping source names to data directories

        Returns:
            Path to output file
        """
        logger.info(f"[NetCDF Export] 开始导出到 {self.output_path}")

        # 1. Filter stations
        filtered = self._filter_stations(stations)
        if not filtered:
            logger.warning("没有站点通过过滤条件")
            return self.output_path

        # 2. Build time axis
        time_axis, time_values = self._build_time_axis()
        time_index = {d: i for i, d in enumerate(time_axis)}

        logger.info(f"时间轴: {self.time_start} 至 {self.time_end} ({len(time_axis)} 天)")

        # 3. Create NetCDF structure
        self._create_netcdf(filtered, time_axis, time_values)

        # 4. Write time series data in batches
        total = len(filtered)
        for i, station in enumerate(filtered):
            # Read time series
            timeseries = self._read_station_timeseries(station, data_paths)

            # Write to NetCDF
            if timeseries:
                self._write_station_timeseries(i, station, timeseries, time_index)

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                logger.info(f"进度: {i + 1}/{total} ({pct:.0f}%)")

        logger.info(f"[NetCDF Export] 完成: {self.output_path}")
        return self.output_path
