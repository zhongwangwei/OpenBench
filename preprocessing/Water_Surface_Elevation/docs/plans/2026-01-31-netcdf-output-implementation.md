# NetCDF Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NetCDF output capability to Step 4, generating CF-1.8 compliant files with station time series data.

**Architecture:** Create a `NetCDFWriter` class in `src/writers/` that reads processed station metadata from `StationList`, retrieves time series by re-reading source files via existing Readers, and writes to NetCDF with unified time axis. Filter stations by upstream area > 100 km².

**Tech Stack:** netCDF4, numpy, existing Readers (HydroWebReader, CGLSReader, ICESatReader, HydroSatReader)

---

## Task 1: Add read_timeseries to HydroSatReader

**Files:**
- Modify: `src/readers/hydrosat_reader.py`
- Test: `tests/test_hydrosat_reader.py`

HydroSatReader is missing `read_timeseries()` method. Other readers have it.

**Step 1: Write the failing test**

Create test file if not exists, add test:

```python
# tests/test_hydrosat_reader.py
import pytest
from pathlib import Path
from src.readers.hydrosat_reader import HydroSatReader


class TestHydroSatReaderTimeseries:
    """Test HydroSatReader.read_timeseries()"""

    def test_read_timeseries_returns_list(self, tmp_path):
        """read_timeseries should return a list of dicts with datetime/elevation"""
        # Create test file
        test_file = tmp_path / "test_station.txt"
        test_file.write_text("""# hydrosat_no.: 12345
# object: Amazon
# latitude: -3.5
# longitude: -60.0
# DATA
2020,1,15,100.5,0.1
2020,2,20,101.2,0.2
2020,3,25,99.8,0.15
""")
        reader = HydroSatReader()
        ts = reader.read_timeseries(str(test_file))

        assert isinstance(ts, list)
        assert len(ts) == 3
        assert 'datetime' in ts[0]
        assert 'elevation' in ts[0]
        assert ts[0]['elevation'] == 100.5

    def test_read_timeseries_file_not_found(self, tmp_path):
        """read_timeseries should return empty list for missing file"""
        reader = HydroSatReader()
        ts = reader.read_timeseries(str(tmp_path / "nonexistent.txt"))
        assert ts == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_hydrosat_reader.py::TestHydroSatReaderTimeseries -v
```

Expected: FAIL with `AttributeError: 'HydroSatReader' object has no attribute 'read_timeseries'`

**Step 3: Implement read_timeseries**

Add to `src/readers/hydrosat_reader.py` after `_parse_satellite` method (around line 288):

```python
    def read_timeseries(self, filepath: str) -> List[Dict[str, Any]]:
        """
        读取站点时间序列数据

        Args:
            filepath: 文件路径

        Returns:
            时间序列数据列表，每项包含 datetime, elevation, uncertainty
        """
        from typing import List, Dict, Any

        timeseries = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except FileNotFoundError:
            self.log('warning', f"File not found: {filepath}")
            return timeseries
        except PermissionError:
            self.log('error', f"Permission denied: {filepath}")
            return timeseries

        # 复用 _parse_data 方法
        data = self._parse_data(lines)

        for item in data:
            if item.get('value') is not None:
                timeseries.append({
                    'datetime': item.get('date'),
                    'elevation': item['value'],
                    'uncertainty': item.get('error'),
                })

        return timeseries
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_hydrosat_reader.py::TestHydroSatReaderTimeseries -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/readers/hydrosat_reader.py tests/test_hydrosat_reader.py
git commit -m "feat(hydrosat): add read_timeseries method for NetCDF export"
```

---

## Task 2: Create writers module structure

**Files:**
- Create: `src/writers/__init__.py`
- Create: `src/writers/netcdf_writer.py` (stub)

**Step 1: Create __init__.py**

```python
# src/writers/__init__.py
"""Writers module for output file generation."""

from .netcdf_writer import NetCDFWriter

__all__ = ['NetCDFWriter']
```

**Step 2: Create netcdf_writer.py stub**

```python
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

        # Readers will be initialized lazily
        self._readers = {}

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
```

**Step 3: Verify import works**

```bash
python -c "from src.writers import NetCDFWriter; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add src/writers/__init__.py src/writers/netcdf_writer.py
git commit -m "feat(writers): create writers module with NetCDFWriter stub"
```

---

## Task 3: Implement station filtering

**Files:**
- Modify: `src/writers/netcdf_writer.py`
- Test: `tests/test_netcdf_writer.py`

**Step 1: Write the failing test**

```python
# tests/test_netcdf_writer.py
import pytest
from src.writers.netcdf_writer import NetCDFWriter
from src.core.station import Station, StationList


class TestNetCDFWriterFilter:
    """Test station filtering logic."""

    def test_filter_by_uparea(self):
        """Should filter stations with uparea > min_uparea"""
        writer = NetCDFWriter({'min_uparea': 100.0})

        stations = StationList()

        # Station with uparea > 100 (should pass)
        s1 = Station(id='1', name='S1', lon=0, lat=0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}
        stations.add(s1)

        # Station with uparea < 100 (should fail)
        s2 = Station(id='2', name='S2', lon=1, lat=1, source='cgls')
        s2.cama_results = {'glb_03min': {'flag': 20, 'uparea': 50.0}}
        stations.add(s2)

        # Station with no CaMa result (should fail)
        s3 = Station(id='3', name='S3', lon=2, lat=2, source='icesat')
        s3.cama_results = {}
        stations.add(s3)

        # Station with flag=0 (should fail)
        s4 = Station(id='4', name='S4', lon=3, lat=3, source='hydrosat')
        s4.cama_results = {'glb_03min': {'flag': 0, 'uparea': 200.0}}
        stations.add(s4)

        filtered = writer._filter_stations(stations)

        assert len(filtered) == 1
        assert filtered[0].id == '1'

    def test_filter_any_resolution_passes(self):
        """Station passes if ANY resolution has valid uparea"""
        writer = NetCDFWriter({'min_uparea': 100.0})

        stations = StationList()
        s1 = Station(id='1', name='S1', lon=0, lat=0, source='hydroweb')
        s1.cama_results = {
            'glb_01min': {'flag': 0, 'uparea': 50.0},   # fail
            'glb_03min': {'flag': 20, 'uparea': 150.0}, # pass
        }
        stations.add(s1)

        filtered = writer._filter_stations(stations)
        assert len(filtered) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterFilter -v
```

Expected: FAIL with `AttributeError: 'NetCDFWriter' object has no attribute '_filter_stations'`

**Step 3: Implement _filter_stations**

Add to `src/writers/netcdf_writer.py`:

```python
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

        logger.info(f"过滤后站点数: {len(filtered)}/{len(stations)} "
                   f"(min_uparea={self.min_uparea} km²)")

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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterFilter -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/writers/netcdf_writer.py tests/test_netcdf_writer.py
git commit -m "feat(netcdf): implement station filtering by upstream area"
```

---

## Task 4: Implement time axis builder

**Files:**
- Modify: `src/writers/netcdf_writer.py`
- Modify: `tests/test_netcdf_writer.py`

**Step 1: Write the failing test**

Add to `tests/test_netcdf_writer.py`:

```python
from datetime import date


class TestNetCDFWriterTimeAxis:
    """Test time axis building."""

    def test_build_time_axis_daily(self):
        """Should create daily time axis from start to end date"""
        writer = NetCDFWriter({
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
            'time_reference': '1800-01-01',
        })

        time_axis, time_values = writer._build_time_axis()

        # 5 days: Jan 1-5
        assert len(time_axis) == 5
        assert time_axis[0] == date(2020, 1, 1)
        assert time_axis[-1] == date(2020, 1, 5)

        # Days since 1800-01-01
        # 2020-01-01 is 80353 days after 1800-01-01
        assert time_values[0] == 80353

    def test_date_to_days_since_ref(self):
        """Should convert date to days since reference"""
        writer = NetCDFWriter({'time_reference': '1800-01-01'})

        # 1800-01-01 -> 0
        assert writer._date_to_days(date(1800, 1, 1)) == 0

        # 1800-01-02 -> 1
        assert writer._date_to_days(date(1800, 1, 2)) == 1

        # 2000-01-01 -> 73049
        assert writer._date_to_days(date(2000, 1, 1)) == 73049
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterTimeAxis -v
```

Expected: FAIL

**Step 3: Implement time axis methods**

Add to `src/writers/netcdf_writer.py` in `__init__`:

```python
        # Time range (will be set from config or auto-detected)
        self.time_start = self._parse_date(config.get('time_start', '1995-01-01'))
        self.time_end = self._parse_date(config.get('time_end', '2024-12-31'))
```

Add methods:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterTimeAxis -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/writers/netcdf_writer.py tests/test_netcdf_writer.py
git commit -m "feat(netcdf): implement time axis builder"
```

---

## Task 5: Implement NetCDF file creation

**Files:**
- Modify: `src/writers/netcdf_writer.py`
- Modify: `tests/test_netcdf_writer.py`

**Step 1: Write the failing test**

Add to `tests/test_netcdf_writer.py`:

```python
import netCDF4 as nc


class TestNetCDFWriterCreate:
    """Test NetCDF file creation."""

    def test_create_netcdf_structure(self, tmp_path):
        """Should create NetCDF with correct dimensions and variables"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-10',
        })

        # Create mock station list
        stations = StationList()
        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb',
                    elevation=100.0, num_observations=50)
        s1.egm08 = 30.0
        s1.egm96 = 29.5
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0,
                          'lon_cama': 10.1, 'lat_cama': 20.1}}
        stations.add(s1)

        filtered = [s1]
        time_axis, time_values = writer._build_time_axis()

        writer._create_netcdf(filtered, time_axis, time_values)

        # Verify file structure
        with nc.Dataset(output_file, 'r') as ds:
            # Check dimensions
            assert 'station' in ds.dimensions
            assert 'time' in ds.dimensions
            assert len(ds.dimensions['station']) == 1
            assert len(ds.dimensions['time']) == 10

            # Check variables exist
            assert 'wse' in ds.variables
            assert 'lat' in ds.variables
            assert 'lon' in ds.variables
            assert 'station_id' in ds.variables
            assert 'time' in ds.variables
            assert 'EGM08' in ds.variables

            # Check global attributes
            assert 'CF-1.8' in ds.Conventions
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterCreate -v
```

Expected: FAIL

**Step 3: Implement _create_netcdf**

Add to `src/writers/netcdf_writer.py`:

```python
    def _create_netcdf(self, stations: List[Station], time_axis: List[date],
                       time_values: np.ndarray) -> None:
        """
        Create NetCDF file with dimensions and variables.

        Args:
            stations: Filtered station list
            time_axis: List of dates
            time_values: Array of days since reference
        """
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

            # Station metadata
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
        import netCDF4 as nc

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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterCreate -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/writers/netcdf_writer.py tests/test_netcdf_writer.py
git commit -m "feat(netcdf): implement NetCDF file structure creation"
```

---

## Task 6: Implement time series reading and writing

**Files:**
- Modify: `src/writers/netcdf_writer.py`
- Modify: `tests/test_netcdf_writer.py`

**Step 1: Write the failing test**

Add to `tests/test_netcdf_writer.py`:

```python
class TestNetCDFWriterTimeseries:
    """Test time series writing."""

    def test_write_station_timeseries(self, tmp_path):
        """Should write time series data to correct positions"""
        output_file = tmp_path / "test.nc"
        writer = NetCDFWriter({
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-05',
        })

        # Setup
        stations = StationList()
        s1 = Station(id='1', name='S1', lon=10.0, lat=20.0, source='hydroweb')
        s1.cama_results = {'glb_03min': {'flag': 20, 'uparea': 150.0}}
        s1.metadata = {'filepath': 'dummy.txt'}
        stations.add(s1)

        filtered = [s1]
        time_axis, time_values = writer._build_time_axis()
        writer._create_netcdf(filtered, time_axis, time_values)

        # Mock timeseries
        timeseries = [
            {'datetime': date(2020, 1, 2), 'elevation': 100.5},
            {'datetime': date(2020, 1, 4), 'elevation': 101.2},
        ]

        # Build time index
        time_index = {d: i for i, d in enumerate(time_axis)}

        writer._write_station_timeseries(0, s1, timeseries, time_index)

        # Verify
        with nc.Dataset(output_file, 'r') as ds:
            wse = ds.variables['wse'][0, :]
            # Index 0 (Jan 1) should be NaN
            assert np.isnan(wse[0])
            # Index 1 (Jan 2) should be 100.5
            assert wse[1] == pytest.approx(100.5)
            # Index 3 (Jan 4) should be 101.2
            assert wse[3] == pytest.approx(101.2)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterTimeseries -v
```

Expected: FAIL

**Step 3: Implement _write_station_timeseries**

Add to `src/writers/netcdf_writer.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_netcdf_writer.py::TestNetCDFWriterTimeseries -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/writers/netcdf_writer.py tests/test_netcdf_writer.py
git commit -m "feat(netcdf): implement time series writing"
```

---

## Task 7: Implement Reader integration

**Files:**
- Modify: `src/writers/netcdf_writer.py`

**Step 1: Implement _get_reader and _read_station_timeseries**

Add to `src/writers/netcdf_writer.py`:

```python
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
```

**Step 2: Verify it compiles**

```bash
python -c "from src.writers.netcdf_writer import NetCDFWriter; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add src/writers/netcdf_writer.py
git commit -m "feat(netcdf): implement reader integration for time series"
```

---

## Task 8: Implement main write() method

**Files:**
- Modify: `src/writers/netcdf_writer.py`

**Step 1: Implement write() method**

Replace the stub `write()` method:

```python
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
```

**Step 2: Verify it compiles**

```bash
python -c "from src.writers.netcdf_writer import NetCDFWriter; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add src/writers/netcdf_writer.py
git commit -m "feat(netcdf): implement main write() method with batch processing"
```

---

## Task 9: Integrate with Step4Merge

**Files:**
- Modify: `src/steps/step4_merge.py`
- Modify: `config/global.yaml`

**Step 1: Update Step4Merge.run()**

Modify `src/steps/step4_merge.py`:

```python
"""Step 4: Output merge and file generation."""
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from ..core.station import Station, StationList
from ..utils.logger import get_logger
from ..constants import RESOLUTIONS

logger = get_logger(__name__)


class Step4Merge:
    """Step 4: Generate output files."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './output'))
        self.resolutions = config.get('resolutions', RESOLUTIONS)

    def run(self, stations: StationList, merge: bool = False) -> List[str]:
        """Generate output files.

        Args:
            stations: StationList with all processed stations
            merge: If True, merge all sources into single file

        Returns:
            List of output file paths
        """
        logger.info("[Step 4] 生成输出文件...")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check output format
        output_config = self.config.get('output', {})
        output_format = output_config.get('format', 'txt')

        if output_format == 'netcdf':
            return self._write_netcdf(stations)
        elif merge:
            return self._write_merged(stations)
        else:
            return self._write_separate(stations)

    def _write_netcdf(self, stations: StationList) -> List[str]:
        """Write to NetCDF format."""
        from ..writers.netcdf_writer import NetCDFWriter

        output_config = self.config.get('output', {})

        # Set default output path if not specified
        if 'netcdf_file' not in output_config:
            output_config['netcdf_file'] = str(self.output_dir / 'OpenBench_WSE.nc')

        writer = NetCDFWriter(output_config)

        # Get data paths from config
        data_paths = self.config.get('data_sources', {})

        output_path = writer.write(stations, data_paths)
        return [str(output_path)]

    # ... rest of existing methods unchanged ...
```

**Step 2: Update config/global.yaml**

Add to output section:

```yaml
# ============================================================
# 输出配置
# ============================================================

output:
  # 输出格式: txt 或 netcdf
  format: txt

  # NetCDF 配置 (当 format: netcdf 时生效)
  netcdf_file: OpenBench_WSE.nc
  time_reference: "1800-01-01"
  time_start: "1995-01-01"
  time_end: "2024-12-31"
  min_uparea: 100.0        # km², 最小上游面积过滤
  chunk_size: 1000         # 批处理大小

  # 站点列表文件名模板 (当 format: txt 时生效)
  station_list: altimetry_{source}_{date}.txt
```

**Step 3: Verify integration**

```bash
python -c "
from src.steps.step4_merge import Step4Merge
from src.core.station import StationList
config = {'output': {'format': 'txt'}}
step = Step4Merge(config)
print('OK')
"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add src/steps/step4_merge.py config/global.yaml
git commit -m "feat(step4): integrate NetCDF writer with output format selection"
```

---

## Task 10: Add integration test

**Files:**
- Create: `tests/test_netcdf_integration.py`

**Step 1: Write integration test**

```python
# tests/test_netcdf_integration.py
"""Integration test for NetCDF output."""
import pytest
import tempfile
from pathlib import Path
import netCDF4 as nc
import numpy as np

from src.core.station import Station, StationList
from src.writers.netcdf_writer import NetCDFWriter


class TestNetCDFIntegration:
    """End-to-end NetCDF output test."""

    def test_full_write_workflow(self, tmp_path):
        """Test complete workflow: filter -> create -> write"""
        # Setup
        output_file = tmp_path / "test_wse.nc"

        config = {
            'netcdf_file': str(output_file),
            'time_start': '2020-01-01',
            'time_end': '2020-01-10',
            'min_uparea': 100.0,
        }

        writer = NetCDFWriter(config)

        # Create test stations
        stations = StationList()

        # Station 1: passes filter
        s1 = Station(id='HW001', name='Amazon_Station', lon=-60.0, lat=-3.5,
                    source='hydroweb', elevation=50.0, num_observations=100)
        s1.egm08 = 20.0
        s1.egm96 = 19.5
        s1.cama_results = {
            'glb_03min': {'flag': 20, 'uparea': 500000.0, 'lon_cama': -60.1, 'lat_cama': -3.6},
            'glb_15min': {'flag': 20, 'uparea': 490000.0, 'lon_cama': -60.0, 'lat_cama': -3.5},
        }
        s1.metadata = {'filepath': '/mock/path.txt'}
        stations.add(s1)

        # Station 2: fails filter (uparea < 100)
        s2 = Station(id='HW002', name='Small_Stream', lon=-61.0, lat=-4.0,
                    source='hydroweb', elevation=100.0, num_observations=50)
        s2.cama_results = {'glb_03min': {'flag': 20, 'uparea': 50.0}}
        stations.add(s2)

        # Mock data_paths (won't be used since filepath doesn't exist)
        data_paths = {'hydroweb': '/mock/data'}

        # Run
        result = writer.write(stations, data_paths)

        # Verify
        assert result == output_file
        assert output_file.exists()

        with nc.Dataset(output_file, 'r') as ds:
            # Only 1 station should pass
            assert len(ds.dimensions['station']) == 1

            # Check metadata
            assert ds.variables['station_id'][0] == 'HW001'
            assert ds.variables['lat'][0] == pytest.approx(-3.5)
            assert ds.variables['lon'][0] == pytest.approx(-60.0)
            assert ds.variables['EGM08'][0] == pytest.approx(20.0)

            # Check CaMa results
            assert ds.variables['cama_uparea_03min'][0] == pytest.approx(500000.0)

            # Check time dimension
            assert len(ds.dimensions['time']) == 10

            # Check global attributes
            assert 'CF-1.8' in ds.Conventions
```

**Step 2: Run integration test**

```bash
pytest tests/test_netcdf_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_netcdf_integration.py
git commit -m "test(netcdf): add integration test for full workflow"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add read_timeseries to HydroSatReader | hydrosat_reader.py |
| 2 | Create writers module structure | writers/__init__.py, netcdf_writer.py |
| 3 | Implement station filtering | netcdf_writer.py |
| 4 | Implement time axis builder | netcdf_writer.py |
| 5 | Implement NetCDF file creation | netcdf_writer.py |
| 6 | Implement time series writing | netcdf_writer.py |
| 7 | Implement Reader integration | netcdf_writer.py |
| 8 | Implement main write() method | netcdf_writer.py |
| 9 | Integrate with Step4Merge | step4_merge.py, global.yaml |
| 10 | Add integration test | test_netcdf_integration.py |

**Dependencies:** netCDF4>=1.6.0
