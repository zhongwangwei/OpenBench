import glob
import importlib
import logging
import os
import re
import shutil
import gc
import time
import functools
import psutil
from typing import List, Dict, Any, Tuple, Callable, Union

# Module-level set to track custom filter warnings
_MODULE_CUSTOM_FILTER_WARNINGS = set()

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed

# Check pandas version for frequency alias compatibility
try:
    pd_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
    USE_NEW_FREQ_ALIASES = pd_version >= (2, 2)  # New aliases introduced in pandas 2.2
except:
    USE_NEW_FREQ_ALIASES = False

from .Lib_Unit import UnitProcessing
from openbench.util.Mod_Converttype import Convert_Type

# Import interfaces
try:
    from openbench.util.Mod_Interfaces import IDataProcessor, IDataLoader, BaseProcessor, ProcessingPipeline
    _HAS_INTERFACES = True
except ImportError:
    _HAS_INTERFACES = False
    IDataProcessor = object
    IDataLoader = object
    BaseProcessor = object
    ProcessingPipeline = object

# Import data pipeline
try:
    from openbench.data.Mod_DataPipeline import (
        create_standard_pipeline, 
        process_dataset, 
        DataPipelineBuilder,
        DataValidationProcessor,
        CoordinateProcessor
    )
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False
    def create_standard_pipeline(*args, **kwargs):
        return None
    def process_dataset(data, *args, **kwargs):
        return data

# Import unified exception handling
try:
    from openbench.util.Mod_Exceptions import (
        DataProcessingError, 
        FileSystemError, 
        error_handler, 
        safe_execute,
        validate_file_exists,
        log_performance_warning
    )
    _HAS_EXCEPTIONS = True
except ImportError:
    _HAS_EXCEPTIONS = False
    # Fallback error handling
    DataProcessingError = Exception
    FileSystemError = Exception
    def error_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def safe_execute(func, *args, **kwargs):
        return func(*args, **kwargs)
    def validate_file_exists(path):
        pass
    def log_performance_warning(*args, **kwargs):
        pass


# Import enhanced logging if available
try:
    from openbench.util.Mod_LoggingSystem import get_logging_manager, performance_logged
    _HAS_ENHANCED_LOGGING = True
except ImportError:
    _HAS_ENHANCED_LOGGING = False
    def performance_logged(operation=None):
        def decorator(func):
            return func
        return decorator

# Import caching system (required for data processing)
try:
    from openbench.data.Mod_CacheSystem import get_cache_manager, cached, DataCache
    _HAS_CACHE = True
except ImportError:
    raise ImportError(
        "CacheSystem is required for data processing modules. "
        "Please ensure openbench.data.Mod_CacheSystem is available. "
        "This module provides essential caching functionality for data processing performance."
    )

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger("xarray").setLevel(logging.WARNING)

def performance_monitor(func: Callable) -> Callable:
    """Enhanced decorator to monitor function performance with error handling."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory usage
        try:
            process = psutil.Process(os.getpid())
            start_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
            start_time = time.time()
            start_cpu = process.cpu_percent()
        except Exception as e:
            logging.warning(f"Failed to get system resources: {e}")
            start_mem = 0
            start_time = time.time()
            start_cpu = 0

        try:
            # Execute the function
            result = func(*args, **kwargs)

            # Calculate execution time and memory usage
            end_time = time.time()
            execution_time = end_time - start_time
            
            try:
                end_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
                end_cpu = process.cpu_percent()
                memory_used = end_mem - start_mem
                cpu_used = end_cpu - start_cpu
            except:
                memory_used = 0
                cpu_used = 0

            # Log performance 
            logging.info(f"Performance  for {func.__name__}:")
            logging.info(f"  Execution time: {execution_time:.2f} seconds")
            logging.info(f"  Memory usage: {memory_used:.3f} GB")
            logging.info(f"  CPU usage: {cpu_used:.1f}%")

            # Use enhanced performance warning if available
            if _HAS_EXCEPTIONS:
                log_performance_warning(func.__name__, execution_time)
            
            # Log warning if memory usage is high
            try:
                total_memory = psutil.virtual_memory().total / (1024 ** 3)
                if memory_used > 0.8 * total_memory:  # 80% of total memory
                    logging.warning(f"High memory usage detected in {func.__name__}: {memory_used:.3f} GB")
            except:
                pass  # Ignore errors in memory check

            return result

        except Exception as e:
            # Log error with performance context
            end_time = time.time()
            execution_time = end_time - start_time
            
            try:
                end_mem = process.memory_info().rss / 1024 / 1024 / 1024
                memory_used = end_mem - start_mem
            except:
                memory_used = 0

            logging.error(f"Error in {func.__name__} after {execution_time:.2f}s and using {memory_used:.3f} GB:")
            logging.error(str(e))
            raise

    return wrapper


def get_system_resources():
    """
    Get system resources information with cross-platform compatibility.
    
    Returns:
        dict: Dictionary containing system resource information
    """
    import platform
    
    # Initialize default values
    result = {
        'total_memory_gb': 8,  # Default values
        'available_memory_gb': 4,
        'cpu_count': 4,
        'cpu_freq_mhz': 0
    }
    
    try:
        # Get memory information - works on all platforms
        memory_info = psutil.virtual_memory()
        result['total_memory_gb'] = memory_info.total / (1024 ** 3)
        result['available_memory_gb'] = memory_info.available / (1024 ** 3)
    except Exception as e:
        logging.warning(f"Failed to get memory info: {e}")
    
    try:
        # Get CPU count - works on all platforms
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is not None:
            result['cpu_count'] = cpu_count
        else:
            # Fallback to logical CPU count
            result['cpu_count'] = psutil.cpu_count(logical=True) or 4
    except Exception as e:
        logging.warning(f"Failed to get CPU count: {e}")
    
    # Get CPU frequency with platform-specific handling
    cpu_freq_from_psutil = False
    try:
        cpu_freq_info = psutil.cpu_freq()
        if cpu_freq_info is not None and hasattr(cpu_freq_info, 'max') and cpu_freq_info.max:
            result['cpu_freq_mhz'] = cpu_freq_info.max
            cpu_freq_from_psutil = True
        elif cpu_freq_info is not None and hasattr(cpu_freq_info, 'current') and cpu_freq_info.current:
            result['cpu_freq_mhz'] = cpu_freq_info.current
            cpu_freq_from_psutil = True
    except Exception as e:
        logging.debug(f"psutil.cpu_freq() failed: {e}")
    
    # If psutil didn't work, try platform-specific fallbacks
    if not cpu_freq_from_psutil:
        try:
            system = platform.system().lower()
            if system == 'darwin':  # macOS
                result['cpu_freq_mhz'] = _get_macos_cpu_freq()
            elif system == 'linux':
                result['cpu_freq_mhz'] = _get_linux_cpu_freq()
            elif system == 'windows':
                result['cpu_freq_mhz'] = _get_windows_cpu_freq()
        except Exception as e:
            logging.debug(f"Platform-specific CPU frequency detection failed: {e}")
            # CPU frequency is optional, so we continue with 0
    
    return result


def _get_macos_cpu_freq():
    """Get CPU frequency on macOS."""
    try:
        import subprocess
        
        # For Apple Silicon Macs, try sysctl to get CPU frequency
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.cpufrequency_max'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip().isdigit():
                # Convert Hz to MHz
                return float(result.stdout.strip()) / 1000000
        except Exception:
            pass
        
        # Try alternative sysctl commands for Apple Silicon
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.cpufrequency'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return float(result.stdout.strip()) / 1000000
        except Exception:
            pass
        
        # For Intel Macs or fallback, try system_profiler
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            import re
            for line in result.stdout.split('\n'):
                if 'Processor Speed' in line:
                    # Extract frequency (e.g., "2.3 GHz" -> 2300)
                    match = re.search(r'(\d+\.?\d*)\s*GHz', line)
                    if match:
                        return float(match.group(1)) * 1000
                elif 'Chip:' in line and 'Apple' in line:
                    # For Apple Silicon, provide estimated frequencies based on chip model
                    if 'M1' in line:
                        return 3200  # M1 estimated max frequency
                    elif 'M2' in line:
                        return 3500  # M2 estimated max frequency
                    elif 'M3' in line:
                        return 4000  # M3 estimated max frequency
                    elif 'M4' in line:
                        return 4400  # M4 estimated max frequency
        
        return 0
    except Exception:
        return 0


def _get_linux_cpu_freq():
    """Get CPU frequency on Linux."""
    try:
        # Try reading from /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('cpu MHz'):
                    return float(line.split(':')[1].strip())
        return 0
    except Exception:
        return 0


def _get_windows_cpu_freq():
    """Get CPU frequency on Windows."""
    try:
        import subprocess
        # Try wmic command
        result = subprocess.run(['wmic', 'cpu', 'get', 'MaxClockSpeed', '/format:value'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'MaxClockSpeed=' in line:
                    freq = line.split('=')[1].strip()
                    if freq.isdigit():
                        return float(freq)  # Already in MHz
        return 0
    except Exception:
        return 0


def calculate_optimal_chunk_size(dataset_size_gb: float, available_memory_gb: float) -> Dict[str, str]:
    """
    Calculate optimal chunk size based on dataset size and available memory.
    Using 'auto' for all dimensions to let xarray handle chunking automatically.
    
    Args:
        dataset_size_gb (float): Size of the dataset in GB
        available_memory_gb (float): Available memory in GB
    
    Returns:
        dict: Dictionary containing chunk sizes for different dimensions
    """
    # Return 'auto' for all dimensions
    return {
        'time': 'auto',
        'lat': 'auto',
        'lon': 'auto'
    }


def calculate_optimal_cores(cpu_count: int, available_memory_gb: float, dataset_size_gb: float) -> int:
    """
    Calculate optimal number of cores based on system resources and dataset size.
    
    Args:
        cpu_count (int): Number of CPU cores
        available_memory_gb (float): Available memory in GB
        dataset_size_gb (float): Size of the dataset in GB
    
    Returns:
        int: Optimal number of cores to use
    """
    # Calculate memory per core needed
    memory_per_core = dataset_size_gb / cpu_count

    # If memory per core is too high, reduce number of cores
    if memory_per_core > available_memory_gb * 0.8:
        optimal_cores = max(1, int(available_memory_gb * 0.8 / memory_per_core))
    else:
        # Leave one core free for system processes
        optimal_cores = max(1, cpu_count - 1)

    return optimal_cores


class BaseDatasetProcessing(BaseProcessor if _HAS_INTERFACES else object):
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize base processor if available
        if _HAS_INTERFACES:
            super().__init__(name=config.get('name', 'BaseDatasetProcessing'))
        
        self.initialize_attributes(config)
        self.setup_output_directories()
        self.initialize_resource_parameters()
        self.setup_data_pipeline(config)

    def initialize_resource_parameters(self):
        """Initialize resource-related parameters based on system capabilities."""
        # Get system resources
        resources = get_system_resources()

        # Set default num_cores if not specified
        if not hasattr(self, 'num_cores') or self.num_cores <= 0:
            self.num_cores = resources['cpu_count']

        # Store resource information
        self.system_resources = resources

        # Default chunk size (will be adjusted per operation)
        self.default_chunks = {
            'time': 'auto',
            'lat': 'auto',
            'lon': 'auto'
        }
    
    def setup_data_pipeline(self, config: Dict[str, Any]):
        """Setup data processing pipeline with configuration."""
        if not _HAS_PIPELINE:
            self.pipeline = None
            return
        
        # Create pipeline configuration
        pipeline_config = {
            'enable_validation': config.get('enable_validation', True),
            'enable_coordinate_processing': config.get('enable_coordinate_processing', True),
            'enable_unit_conversion': config.get('enable_unit_conversion', False),
            'enable_quality_control': config.get('enable_quality_control', False),
            'validation_rules': {
                'required_dims': ['lat', 'lon'],
                'check_missing': True,
                'check_infinite': True
            },
            'coordinate_map': {
                'latitude': 'lat',
                'longitude': 'lon',
                'time': 'time'
            }
        }
        
        # Update with user configuration
        pipeline_config.update(config.get('pipeline_config', {}))
        
        # Create pipeline
        try:
            self.pipeline = create_standard_pipeline(pipeline_config)
            logging.debug(f"Data pipeline created with {getattr(self.pipeline, 'get_processor_count', lambda: 0)()} processors")
        except Exception as e:
            logging.warning(f"Failed to create data pipeline: {e}")
            self.pipeline = None

    def get_optimal_chunks(self, dataset_size_gb: float) -> Dict[str, str]:
        """
        Get optimal chunk size for a dataset.
        
        Args:
            dataset_size_gb (float): Size of the dataset in GB
        
        Returns:
            dict: Optimal chunk sizes
        """
        return calculate_optimal_chunk_size(
            dataset_size_gb,
            self.system_resources['available_memory_gb']
        )

    def get_optimal_cores(self, dataset_size_gb: float) -> int:
        """
        Get optimal number of cores for processing.
        
        Args:
            dataset_size_gb (float): Size of the dataset in GB
        
        Returns:
            int: Optimal number of cores
        """
        return calculate_optimal_cores(
            self.system_resources['cpu_count'],
            self.system_resources['available_memory_gb'],
            dataset_size_gb
        )

    def _convert_legacy_freq_alias(self, freq: str) -> str:
        """
        Convert legacy pandas frequency aliases to new ones if needed.
        
        Args:
            freq (str): Frequency string that might contain legacy aliases
            
        Returns:
            str: Updated frequency string with new aliases if applicable
        """
        if not USE_NEW_FREQ_ALIASES:
            return freq
            
        # Map of legacy to new frequency aliases
        legacy_to_new = {
            'M': 'ME',    # Month end
            'Y': 'YE',    # Year end
            'Q': 'QE',    # Quarter end
            'H': 'h',     # Hour
            'T': 'min',   # Minute
            'S': 's',     # Second
            'L': 'ms',    # Millisecond
            'U': 'us',    # Microsecond
            'N': 'ns',    # Nanosecond
        }
        
        # Handle compound frequencies like '3M' -> '3ME'
        import re
        pattern = r'(\d*)([A-Z])'
        
        def replacer(match):
            number = match.group(1)
            letter = match.group(2)
            new_letter = legacy_to_new.get(letter, letter)
            return number + new_letter
        
        return re.sub(pattern, replacer, freq)
    
    def _normalize_frequency(self, freq: str) -> str:
        """
        Convert human-readable frequency strings to pandas-compatible codes.
        
        Args:
            freq (str): Input frequency string (e.g., 'month', 'day', 'hour')
            
        Returns:
            str: Pandas-compatible frequency code (e.g., 'M', 'D', 'H')
        """
        # Use appropriate frequency aliases based on pandas version
        if USE_NEW_FREQ_ALIASES:
            freq_map = {
                'month': 'ME',    # Month End (new alias)
                'mon': 'ME',
                'monthly': 'ME',
                'day': 'D',
                'daily': 'D',
                'hour': 'h',      # Hour (lowercase in new pandas)
                'Hour': 'h',
                'hr': 'h',
                'Hr': 'h',
                'h': 'h',
                'hourly': 'h',
                'year': 'YE',     # Year End (new alias)
                'yr': 'YE',
                'yearly': 'YE',
                'week': 'W',
                'wk': 'W',
                'weekly': 'W',
            }
        else:
            freq_map = {
                'month': 'M',     # Month (old alias)
                'mon': 'M',
                'monthly': 'M',
                'day': 'D',
                'daily': 'D',
                'hour': 'H',      # Hour (uppercase in old pandas)
                'Hour': 'H',
                'hr': 'H',
                'Hr': 'H',
                'h': 'H',
                'hourly': 'H',
                'year': 'Y',      # Year (old alias)
                'yr': 'Y',
                'yearly': 'Y',
                'week': 'W',
                'wk': 'W',
                'weekly': 'W',
            }
        
        # Convert to lowercase for case-insensitive matching
        normalized_freq = freq.lower().strip()
        
        # Get mapped frequency or use original if no mapping found
        result_freq = freq_map.get(normalized_freq, freq)
        
        # Don't convert if we already got a mapped frequency from freq_map
        # Only convert if we're returning the original frequency
        if result_freq == freq:
            result_freq = self._convert_legacy_freq_alias(result_freq)
        
        return result_freq

    def initialize_attributes(self, config: Dict[str, Any]) -> None:
        self.__dict__.update(config)  # Changed this line
        self.sim_varname = [self.sim_varname] if isinstance(self.sim_varname, str) else self.sim_varname
        self.ref_varname = [self.ref_varname] if isinstance(self.ref_varname, str) else self.ref_varname
        # Handle both single values and Series for use_syear and use_eyear
        if hasattr(self.use_syear, 'iloc'):
            self.minyear = int(self.use_syear.min())
            self.maxyear = int(self.use_eyear.max())
        else:
            self.minyear = int(self.use_syear)
            self.maxyear = int(self.use_eyear)

        essential_attrs = ['sim_tim_res', 'ref_tim_res', 'compare_tim_res']
        for attr in essential_attrs:
            if not hasattr(self, attr):
                setattr(self, attr, config.get(attr, 'M'))
                if self.debug_mode:
                    logging.warning(
                        f"Warning: '{attr}' was not provided in the config. Using value from 'tim_res': {getattr(self, attr)}")
        
        # Apply frequency normalization to timing resolution attributes
        if hasattr(self, 'compare_tim_res'):
            original_freq = self.compare_tim_res
            self.compare_tim_res = self._normalize_frequency(self.compare_tim_res)
            if self.compare_tim_res != original_freq:
                logging.debug(f"Normalized frequency: {original_freq} -> {self.compare_tim_res}")
        
        # Also normalize other timing resolution attributes
        for attr in ['sim_tim_res', 'ref_tim_res']:
            if hasattr(self, attr):
                original_freq = getattr(self, attr)
                normalized_freq = self._normalize_frequency(original_freq)
                setattr(self, attr, normalized_freq)
                if normalized_freq != original_freq:
                    logging.debug(f"Normalized {attr}: {original_freq} -> {normalized_freq}")

    def setup_output_directories(self) -> None:
        if self.ref_data_type == 'stn' or self.sim_data_type == 'stn':
            self.station_list = Convert_Type.convert_Frame(pd.read_csv(os.path.join(self.casedir, "stn_list.txt"), header=0))
            output_dir = os.path.join(self.casedir, 'output', 'data', f'stn_{self.ref_source}_{self.sim_source}')
            # shutil.rmtree(output_dir, ignore_errors=True)
            # print(f"Re-creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

    def get_data_params(self, datasource: str) -> Dict[str, Any]:
        return {
            'data_dir': getattr(self, f"{datasource}_dir"),
            'data_groupby': getattr(self, f"{datasource}_data_groupby").lower(),
            'varname': getattr(self, f"{datasource}_varname"),
            'tim_res': getattr(self, f"{datasource}_tim_res"),
            'varunit': getattr(self, f"{datasource}_varunit"),
            'prefix': getattr(self, f"{datasource}_prefix"),
            'suffix': getattr(self, f"{datasource}_suffix"),
            'datasource': datasource,  # This should be 'ref' or 'sim'
            'data_type': getattr(self, f"{datasource}_data_type"),
            'syear': getattr(self, f"{datasource}_syear"),
            'eyear': getattr(self, f"{datasource}_eyear"),
        }

    @performance_monitor
    def process(self, data: Union[str, xr.Dataset] = None, **kwargs) -> xr.Dataset:
        """
        Process data according to interface or legacy mode.
        
        Args:
            data: Either datasource string (legacy) or xr.Dataset (interface mode)
            **kwargs: Additional parameters
            
        Returns:
            Processed dataset (interface mode) or None (legacy mode)
        """
        # Legacy mode - data is datasource string
        if isinstance(data, str):
            datasource = data
            logging.info(f"Processing {datasource} data")
            self._preprocess(datasource)
            logging.info(f"{datasource.capitalize()} data prepared!")
            return None
        
        # Interface mode - data is xr.Dataset
        elif isinstance(data, xr.Dataset):
            # Use enhanced pipeline if available
            if _HAS_PIPELINE and self.pipeline:
                try:
                    processed_data = self.pipeline.process(data, **kwargs)
                    logging.debug("Data processed using enhanced pipeline")
                    return processed_data
                except Exception as e:
                    logging.warning(f"Pipeline processing failed, falling back to basic processing: {e}")
            
            # Fallback to basic processing
            if _HAS_INTERFACES and hasattr(super(), 'validate_input'):
                if not self.validate_input(data):
                    raise ValueError("Input dataset validation failed")
            
            # Apply basic processing steps
            processed_data = self.check_dataset(data)
            processed_data = self.check_coordinate(processed_data)
            
            return processed_data
        
        # Backward compatibility - no arguments provided
        else:
            raise ValueError("Either datasource string or xr.Dataset must be provided")
    
    def process_legacy(self, datasource: str) -> None:
        """Legacy processing method for backward compatibility."""
        return self.process(datasource)

    @performance_monitor
    def _preprocess(self, datasource: str) -> None:
        data_params = self.get_data_params(datasource)

        if data_params['data_type'] != 'stn':
            logging.info(f"Processing {data_params['data_type']} data")
            self.process_grid_data(data_params)
        else:
            self.process_station_data(data_params)

    def check_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        if not isinstance(ds, xr.Dataset):
            logging.error("Input data must be a xarray dataset.")
            raise ValueError("Input data must be a xarray dataset.")
        return ds

    def check_units(self, input_units: str, target_units: str) -> bool:
        input_units_list = sorted(input_units.split())
        target_units_list = sorted(target_units.split())
        return input_units_list == target_units_list

    def check_coordinate(self, ds: xr.Dataset) -> xr.Dataset:
        for coord in ds.coords:
            if coord in self.coordinate_map:
                ds = ds.rename({coord: self.coordinate_map[coord]})
        # if the longitude is not between -180 and 180, convert it to the equivalent value between -180 and 180
        if 'lon' in ds.coords:
            ds['lon'] = (ds['lon'] + 180) % 360 - 180
            # Reindex the dataset
            ds = ds.reindex(lon=sorted(ds.lon.values))
        return ds

    def check_time(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str) -> xr.Dataset:
        if 'time' not in ds.coords:
            print("The dataset does not contain a 'time' coordinate.")
            # Based on the syear and eyear, create a time index
            lon = ds.lon.values
            lat = ds.lat.values
            data = ds.values
            time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
            # Check data dimension, if it is 2-dimensional, reshape to 3-dimensional
            if data.ndim == 2:
                data = data.reshape((1, data.shape[0], data.shape[1]))
            try:
                ds1 = xr.Dataset({f'{ds.name}': (['time', 'lat', 'lon'], data)},
                                 coords={'time': time_index, 'lat': lat, 'lon': lon})
            except:
                try:
                    ds1 = xr.Dataset({f'{ds.name}': (['time', 'lon', 'lat'], data)},
                                     coords={'time': time_index, 'lon': lon, 'lat': lat})
                except:
                    ds1 = xr.Dataset({f'{ds.name}': (['lat', 'lon', 'time'], data)},
                                     coords={'lat': lat, 'lon': lon, 'time': time_index})
            ds1 = ds1.transpose('time', 'lat', 'lon')
            return ds1[f'{ds.name}']

        if not hasattr(ds['time'], 'dt'):
            try:
                ds['time'] = pd.to_datetime(ds['time'])
            except:
                lon = ds.lon.values
                lat = ds.lat.values
                data = ds.values
                time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
                try:
                    ds1 = xr.Dataset({f'{ds.name}': (['time', 'lat', 'lon'], data)},
                                     coords={'time': time_index, 'lat': lat, 'lon': lon})
                except:
                    try:
                        ds1 = xr.Dataset({f'{ds.name}': (['time', 'lon', 'lat'], data)},
                                         coords={'time': time_index, 'lon': lon, 'lat': lat})
                    except:
                        ds1 = xr.Dataset({f'{ds.name}': (['lat', 'lon', 'time'], data)},
                                         coords={'lat': lat, 'lon': lon, 'time': time_index})
                    ds1 = ds1.transpose('time', 'lat', 'lon')
                return ds1[f'{ds.name}']

        # Check for duplicate time values
        if ds['time'].to_index().has_duplicates:
            logging.warning("Warning: Duplicate time values found. Removing duplicates...")
            # Remove duplicates by keeping the first occurrence
            _, index = np.unique(ds['time'], return_index=True)
            ds = ds.isel(time=index)

        # Ensure time is sorted
        ds = ds.sortby('time')
        try:
            return ds.transpose('time', 'lat', 'lon')[f'{ds.name}']
        except:
            try:
                return ds.transpose('time', 'lat', 'lon')
            except:
                try:
                    return ds.transpose('time', 'lon', 'lat')
                except:
                    return ds.squeeze()

    @performance_monitor
    @cached(key_prefix="time_integrity", ttl=1800)  # Cache for 30 minutes
    def check_dataset_time_integrity(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str, datasource: str) -> xr.Dataset:
        """Checks and fills missing time values in an xarray Dataset with specified comparison scales."""
        # Ensure the dataset has a proper time index
        ds = self.check_time(ds, syear, eyear, tim_res)
        # Apply model-specific time adjustments
        if datasource == 'stat':
            ds['time'] = pd.DatetimeIndex(ds['time'].values)
        else:
            if self.sim_data_type != 'stn':
                ds = self.apply_model_specific_time_adjustment(ds, datasource, syear, eyear, tim_res)
        ds = self.make_time_integrity(ds, syear, eyear, tim_res, datasource)
        return ds

    @performance_monitor
    @cached(key_prefix="make_time_integrity", ttl=1800)  # Cache for 30 minutes
    def make_time_integrity(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str, datasource: str) -> xr.Dataset:
        match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
        if match:
            num_value, time_unit = match.groups()
            num_value = int(num_value) if num_value else 1
            time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
            if time_unit.lower() in ['m', 'month', 'mon']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-15T00:00:00'))
                try:
                    ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-15T00:00:00'))
                except:
                    ds['time'] = time_index
            elif time_unit.lower() in ['d', 'day', '1d', '1day']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-%dT12:00:00'))
                try:
                    ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT12:00:00'))
                except:
                    ds['time'] = time_index
            elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-%dT%H:30:00'))
                try:
                    ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))
                except:
                    ds['time'] = time_index
            elif time_unit.lower() in ['y', 'year', '1y', '1year']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-01-01T00:00:00'))
                try:
                    ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-01-01T00:00:00'))
                except:
                    ds['time'] = time_index
            time_var = ds.time
            # Safely set calendar attribute using encoding
            try:
                time_var.encoding['calendar'] = 'proleptic_gregorian'
            except Exception:
                try:
                    # If encoding fails, create new time coordinate with calendar attribute
                    new_time = xr.DataArray(
                        time_var.values,
                        dims=['time'],
                        attrs={'calendar': 'proleptic_gregorian'}
                    )
                    ds = ds.assign_coords(time=new_time)
                    time_var = ds.time
                except Exception:
                    # If all else fails, just continue without setting calendar
                    logging.debug("Could not set calendar attribute, proceeding without it")
                    pass
            time_values = time_var
            
            # Create a complete time series based on the specified time frequency and range 
            # Compare the actual time with the complete time series to find the missing time
            # print('Checking time series completeness...')
            missing_times = time_index[~np.isin(time_index, time_values)]
            if len(missing_times) > 0 and len(missing_times) < len(time_var):
                logging.warning("Time series is not complete. Missing time values found: ")
                print(missing_times)
                logging.info('Filling missing time values with np.nan')
                # Fill missing time values with np.nan
                ds = ds.reindex(time=time_index)
                ds = ds.where(ds.time.isin(time_values), np.nan)
            else:
                # print('Time series is complete.')
                pass
        return ds

    @performance_monitor
    def apply_model_specific_time_adjustment(self, ds: xr.Dataset, datasource: str, syear: int, eyear: int,
                                             tim_res: str) -> xr.Dataset:
        model = self.sim_source if datasource == 'sim' else self.ref_source
        try:
            custom_module = importlib.import_module(f"openbench.data.custom.{model}_filter")
            custom_time_adjustment = getattr(custom_module, f"adjust_time_{model}")
            ds = custom_time_adjustment(self, ds, syear, eyear, tim_res)
        except (ImportError, AttributeError):
            # This is expected behavior - not all models need custom time adjustments
            # Log at debug level to avoid noise
            logging.debug(f"No custom time adjustment found for {model}. Using original time values.")
            pass
        return ds

    def select_var(self, syear: int, eyear: int, tim_res: str, VarFile: str, varname: List[str], datasource: str) -> xr.Dataset:
        try:
            try:
                ds = xr.open_dataset(VarFile)  # .squeeze()
            except:
                ds = xr.open_dataset(VarFile, decode_times=False)  # .squeeze()
        except Exception as e:
            logging.error(f"Failed to open dataset: {VarFile}")
            logging.error(f"Error: {str(e)}")
            raise
        try:
            ds = self.apply_custom_filter(datasource, ds, varname)
            ds = Convert_Type.convert_nc(ds)
        except:
            ds = Convert_Type.convert_nc(ds[varname[0]])
        return ds

    def apply_custom_filter(self, datasource: str, ds: xr.Dataset, varname: List) -> xr.Dataset:
        if datasource == 'stat':
            return ds[varname[0]]
        else:
            model = self.sim_source if datasource == 'sim' else self.ref_source
            try:
                logging.info(f"Loading custom variable filter for {model}")
                custom_module = importlib.import_module(f"openbench.data.custom.{model}_filter")
                custom_filter = getattr(custom_module, f"filter_{model}")
                self, ds = custom_filter(self, ds)
            except AttributeError:
                # Only show warning once per model using module-level tracking
                if model not in _MODULE_CUSTOM_FILTER_WARNINGS:
                    logging.warning(f"Custom filter function for {model} not found.")
                    _MODULE_CUSTOM_FILTER_WARNINGS.add(model)
                raise
        return ds

    @performance_monitor
    def select_timerange(self, ds: xr.Dataset, syear: int, eyear: int) -> xr.Dataset:
        if (eyear < syear) or (ds.sel(time=slice(f'{syear}-01-01T00:00:00', f'{eyear}-12-31T23:59:59')) is None):
            logging.error(f"Error: Attempting checking the data time range.")
            exit()
        else:
            return ds.sel(time=slice(f'{syear}-01-01T00:00:00', f'{eyear}-12-31T23:59:59'))

    @performance_monitor
    @cached(key_prefix="resample_data", ttl=1800)  # Cache for 30 minutes
    def resample_data(self, dfx1: xr.Dataset, tim_res: str, startx: int, endx: int) -> xr.Dataset:
        match = re.match(r'(\d+)\s*([a-zA-Z]+)', tim_res)
        if not match:
            logging.error("Invalid time resolution format. Use '3month', '6hr', etc.")
            raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")

        value, unit = match.groups()
        value = int(value)
        
        # Get frequency map based on pandas version
        if USE_NEW_FREQ_ALIASES:
            freq_map = {'month': 'ME', 'day': 'D', 'hour': 'h', 'year': 'YE', 'week': 'W'}
        else:
            freq_map = {'month': 'M', 'day': 'D', 'hour': 'H', 'year': 'Y', 'week': 'W'}
            
        freq = freq_map.get(unit.lower())
        if not freq:
            logging.error(f"Unsupported time unit: {unit}")
            raise ValueError(f"Unsupported time unit: {unit}")

        # Build frequency string 
        freq_str = f'{value}{freq}'
        time_index = pd.date_range(start=f'{startx}-01-01T00:00:00', end=f'{endx}-12-31T59:59:59', freq=freq_str)
        ds = xr.Dataset({'data': ('time', np.nan * np.ones(len(time_index)))}, coords={'time': time_index})
        orig_ds_reindexed = dfx1.reindex(time=ds.time)
        return xr.merge([ds, orig_ds_reindexed]).drop_vars('data')

    @performance_monitor
    def split_year(self, ds: xr.Dataset, casedir: str, suffix: str, prefix: str, use_syear: int, use_eyear: int,
                   datasource: str) -> None:
        def save_year(casedir: str, suffix: str, prefix: str, ds: xr.Dataset, year: int) -> None:
            ds_year = None
            try:
                ds_year = ds.sel(time=slice(f'{year}-01-01T00:00:00', f'{year}-12-31T23:59:59'))
                ds_year.attrs = {}
                output_file = os.path.join(casedir, 'scratch', f'{datasource}_{prefix}{year}{suffix}.nc')
                ds_year.to_netcdf(output_file)
                logging.info(f"Saved {output_file}")
            finally:
                # Clean up memory
                if ds_year is not None and hasattr(ds_year, 'close'):
                    ds_year.close()
                gc.collect()

        try:
            # Calculate dataset size in GB
            dataset_size_gb = ds.nbytes / (1024 ** 3)

            # Update number of cores based on dataset size
            optimal_cores = min(self.get_optimal_cores(dataset_size_gb), self.num_cores)
            logging.info(f"Using {optimal_cores} cores for splitting years")

            years = range(use_syear, use_eyear + 1)
            Parallel(n_jobs=optimal_cores)(
                delayed(save_year)(casedir, suffix, prefix, ds, year) for year in years
            )
        finally:
            # Ensure main dataset is closed
            if hasattr(ds, 'close'):
                ds.close()
            gc.collect()

    @performance_monitor
    def combine_year(self, year: int, casedir: str, dirx: str, suffix: str, prefix: str, varname: List[str], datasource: str,
                     tim_res: str) -> xr.Dataset:
        try:
            var_files = glob.glob(os.path.join(dirx, f'{prefix}{year}*{suffix}.nc'))
        except:
            var_files = glob.glob(os.path.join(dirx, str(year), f'{prefix}{year}*{suffix}.nc'))

        datasets = []
        try:
            for file in var_files:
                ds = self.select_var(year, year, tim_res, file, varname, datasource)
                datasets.append(ds)
            data0 = xr.concat(datasets, dim="time").sortby('time')
            return data0
        finally:
            # Clean up memory
            for ds in datasets:
                if hasattr(ds, 'close'):
                    ds.close()
            gc.collect()

    def check_file_exist(self, file: str) -> str:
        if not os.path.exists(file):
            logging.error(f"File '{file}' not found.")
            raise FileNotFoundError(f"File '{file}' not found.")
        return file

    @performance_monitor
    def check_all(self, dirx: str, syear: int, eyear: int, tim_res: str, varunit: str, varname: List[str],
                  groupby: str, casedir: str, suffix: str, prefix: str, datasource: str) -> None:
        if groupby == 'single':
            self.preprocess_single_file(dirx, syear, eyear, tim_res, varunit, varname, casedir, suffix, prefix, datasource)
        elif groupby != 'year':
            self.preprocess_non_yearly_files(dirx, syear, eyear, tim_res, varunit, varname, casedir, suffix, prefix, datasource)
        else:
            self.preprocess_yearly_files(dirx, syear, eyear, tim_res, varunit, varname, casedir, suffix, prefix, datasource)

    @performance_monitor
    def preprocess_single_file(self, dirx: str, syear: int, eyear: int, tim_res: str, varunit: str, varname: List[str],
                               casedir: str, suffix: str, prefix: str, datasource: str) -> None:
        logging.info('The dataset groupby is Single --> split it to Year')
        varfile = self.check_file_exist(os.path.join(dirx, f'{prefix}{suffix}.nc'))
        ds = self.select_var(syear, eyear, tim_res, varfile, varname, datasource)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        ds = self.select_timerange(ds, self.minyear, self.maxyear)
        ds, varunit = self.process_units(ds, varunit)
        self.split_year(ds, casedir, suffix, prefix, self.minyear, self.maxyear, datasource)

    @performance_monitor
    def preprocess_non_yearly_files(self, dirx: str, syear: int, eyear: int, tim_res: str, varunit: str, varname: List[str],
                                    casedir: str, suffix: str, prefix: str, datasource: str) -> None:
        logging.info('The dataset groupby is not Year --> combine it to Year')
        ds = self.combine_year(syear, casedir, dirx, suffix, prefix, varname, datasource, tim_res)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        ds, varunit = self.process_units(ds, varunit)
        ds = self.select_timerange(ds, syear, eyear)
        ds.to_netcdf(os.path.join(casedir, 'scratch', f'{datasource}_{prefix}{syear}{suffix}.nc'))

    @performance_monitor
    def preprocess_yearly_files(self, dirx: str, syear: int, eyear: int, tim_res: str, varunit: str, varname: List[str],
                                casedir: str, suffix: str, prefix: str, datasource: str) -> None:
        varfiles = self.check_file_exist(os.path.join(dirx, f'{prefix}{syear}{suffix}.nc'))
        ds = self.select_var(syear, eyear, tim_res, varfiles, varname, datasource)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        ds, varunit = self.process_units(ds, varunit)
        ds = self.select_timerange(ds, syear, eyear)
        ds.to_netcdf(os.path.join(casedir, 'scratch', f'{datasource}_{prefix}{syear}{suffix}.nc'))

    @performance_monitor
    @cached(key_prefix="process_units", ttl=3600)  # Cache for 1 hour
    def process_units(self, ds: xr.Dataset, varunit: str) -> Tuple[xr.Dataset, str]:
        try:
            # 确保我们获取实际的数据数组而不是方法
            if isinstance(ds, xr.Dataset):
                # 如果是数据集，获取第一个变量的数据
                var_name = list(ds.data_vars)[0]
                data_array = ds[var_name].values
            elif isinstance(ds, xr.DataArray):
                # 如果是数据数组，直接获取值
                data_array = ds.values
            else:
                # 如果已经是numpy数组，直接使用
                data_array = ds

            # 进行单位转换
            converted_data, new_unit = UnitProcessing.convert_unit(data_array, varunit.lower())
            # 创建新的数据集或更新现有数据集
            if isinstance(ds, xr.Dataset):
                # 更新数据集中的数据
                ds[var_name].values = converted_data
            elif isinstance(ds, xr.DataArray):
                # 创建新的DataArray
                ds.values = converted_data

            # 更新单位属性
            ds.attrs['units'] = new_unit
            logging.info(f"Converted unit from {varunit} to {new_unit}")

            return ds, new_unit

        except ValueError as e:
            logging.warning(f"Warning: {str(e)}. Attempting specific conversion.")
            # 不要直接退出，而是返回原始数据
            return ds, varunit
        except Exception as e:
            logging.error(f"Error in unit conversion: {str(e)}")
            # 返回原始数据
            return ds, varunit

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class StationDatasetProcessing(BaseDatasetProcessing):
    def process_station_data(self, data_params: Dict[str, Any]) -> None:
        try:
            logging.info("Processing station data")
            Parallel(n_jobs=-1)(
                delayed(self._make_stn_parallel)(
                    self.station_list, data_params['datasource'], i
                ) for i in range(len(self.station_list['ID']))
            )
        finally:
            gc.collect()

    def process_single_station_data(self, stn_data: xr.Dataset, start_year: int, end_year: int, datasource: str) -> xr.Dataset:
        varname = self.ref_varname if datasource == 'ref' else self.sim_varname
        # Check if the variable exists in the dataset
        if varname[0] not in stn_data:
            logging.error(f"Variable '{varname[0]}' not found in the station data.")
            raise ValueError(f"Variable '{varname[0]}' not found in the station data.")

        ds = stn_data[varname[0]]

        # Check the time dimension
        if 'time' not in ds.dims:
            logging.error("Time dimension not found in the station data.")
            raise ValueError("Time dimension not found in the station data.")

        # Ensure the time coordinate is datetime
        if not np.issubdtype(ds.time.dtype, np.datetime64):
            ds['time'] = pd.to_datetime(ds.time.values)

        # Select the time range before resampling
        ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

        # Resample only if there's data in the selected time range
        if len(ds.time) > 0:
            ds = ds.resample(time=self.compare_tim_res).mean()
        else:
            logging.warning(f"No data found for the specified time range {start_year}-{end_year}")
            return None  # or return an empty dataset with the correct structure

        # ds = ds.resample(time=self.compare_tim_res).mean()
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, start_year, end_year, self.compare_tim_res, datasource)

        # Apply unit conversion for station data
        current_varunit = getattr(self, f"{datasource}_varunit")
        if current_varunit:
            try:
                ds, converted_unit = self.process_units(ds, current_varunit)
                logging.info(f"Applied unit conversion for {datasource} station data: {current_varunit} -> {converted_unit}")
            except Exception as e:
                logging.warning(f"Unit conversion failed for {datasource} station data: {e}")

        ds = self.select_timerange(ds, start_year, end_year)
        return ds  # .where((ds > -1e20) & (ds < 1e20), np.nan)

    @performance_monitor
    def _make_stn_parallel(self, station_list: pd.DataFrame, datasource: str, index: int) -> None:
        try:
            station = station_list.iloc[index]
            start_year = int(station['use_syear'])
            end_year = int(station['use_eyear'])
            file_path = station["sim_dir"] if datasource == 'sim' else station["ref_dir"]
            with xr.open_dataset(file_path) as stn_data:
                stn_data = Convert_Type.convert_nc(stn_data)
                processed_data = self.process_single_station_data(stn_data, start_year, end_year, datasource)
                self.save_station_data(processed_data, station, datasource)
        finally:
            gc.collect()


    def save_station_data(self, data: xr.Dataset, station: pd.Series, datasource: str) -> None:
        try:
            station = Convert_Type.convert_Frame(station)
            output_file = os.path.join(self.casedir, 'output', 'data',
                                       f'stn_{self.ref_source}_{self.sim_source}',
                                       f'{self.item}_{datasource}_{station["ID"]}_{station["use_syear"]}_{station["use_eyear"]}.nc')
            # Handle calendar attribute safely
            data.to_netcdf(output_file)
            logging.info(f"Saved station data to {output_file}")
        finally:
            if hasattr(data, 'close'):
                data.close()
            gc.collect()

    @performance_monitor
    def _extract_stn_parallel(self, datasource: str, dataset: xr.Dataset, station_list: pd.DataFrame, index: int) -> None:
        try:
            station = station_list.iloc[index]
            start_year = int(station['use_syear'])
            end_year = int(station['use_eyear'])

            station_data = self.extract_single_station_data(dataset, station, datasource)
            processed_data = self.process_extracted_data(station_data, start_year, end_year)
            self.save_extracted_data(processed_data, station, datasource)
        finally:
            gc.collect()

    @performance_monitor
    def extract_single_station_data(self, dataset: xr.Dataset, station: pd.Series, datasource: str) -> xr.Dataset:
        if datasource == 'ref':
            return dataset.sel(lat=[station['sim_lat']], lon=[station['sim_lon']], method="nearest")
        elif datasource == 'sim':
            return dataset.sel(lat=[station['ref_lat']], lon=[station['ref_lon']], method="nearest")
        else:
            logging.error(f"Invalid datasource: {datasource}")
            raise ValueError(f"Invalid datasource: {datasource}")

    @performance_monitor
    def process_extracted_data(self, data: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
        data = data.sel(time=slice(f'{start_year}-01-01T00:00:00', f'{end_year}-12-31T23:59:59'))
        data = data  # .where((data > -1e20) & (data < 1e20), np.nan)
        return data.resample(time=self.compare_tim_res).mean()

    def save_extracted_data(self, data: xr.Dataset, station: pd.Series, datasource: str) -> None:
        try:
            output_file = os.path.join(self.casedir, 'output', 'data',
                                       f'stn_{self.ref_source}_{self.sim_source}',
                                       f'{self.item}_{datasource}_{station["ID"]}_{station["use_syear"]}_{station["use_eyear"]}.nc')

            data.to_netcdf(output_file)
            logging.info(f"Saved extracted station data to {output_file}")
        finally:
            if hasattr(data, 'close'):
                data.close()
            gc.collect()


class GridDatasetProcessing(BaseDatasetProcessing):
    @performance_monitor
    def process_grid_data(self, data_params: Dict[str, Any]) -> None:
        try:
            self.prepare_grid_data(data_params)
            self.remap_and_combine_data(data_params)
            self.extract_station_data_if_needed(data_params)
        finally:
            gc.collect()

    @performance_monitor
    def prepare_grid_data(self, data_params: Dict[str, Any]) -> None:
        if data_params['data_groupby'] == 'single':
            self.process_single_file(data_params)
        elif data_params['data_groupby'] != 'year':
            self.process_non_yearly_files(data_params)
        else:
            self.process_yearly_files(data_params)

    @performance_monitor
    def process_single_file(self, data_params: Dict[str, Any]) -> None:
        self.check_all(data_params['data_dir'], data_params['syear'], data_params['eyear'],
                       data_params['tim_res'], data_params['varunit'],
                       data_params['varname'], 'single', self.casedir,
                       data_params['suffix'], data_params['prefix'], data_params['datasource'])
        setattr(self, f"{data_params['datasource']}_data_groupby", 'year')

    @performance_monitor
    def process_non_yearly_files(self, data_params: Dict[str, Any]) -> None:
        logging.info(f"Combining data to yearly files...")
        years = range(self.minyear, self.maxyear + 1)
        Parallel(n_jobs=self.num_cores)(
            delayed(self.check_all)(data_params['data_dir'], year, year,
                                    data_params['tim_res'], data_params['varunit'],
                                    data_params['varname'], data_params['data_groupby'],
                                    self.casedir, data_params['suffix'], data_params['prefix'],
                                    data_params['datasource'])
            for year in years
        )

    @performance_monitor
    def process_yearly_files(self, data_params: Dict[str, Any]) -> None:
        years = range(self.minyear, self.maxyear + 1)
        Parallel(n_jobs=self.num_cores)(
            delayed(self.check_all)(
                data_params['data_dir'],
                year,
                year,
                data_params['tim_res'],
                data_params['varunit'],
                data_params['varname'],
                data_params['data_groupby'],
                self.casedir,
                data_params['suffix'],
                data_params['prefix'],
                data_params['datasource']
            )
            for year in years
        )

    @performance_monitor
    def remap_and_combine_data(self, data_params: Dict[str, Any]) -> None:
        data_dir = os.path.join(self.casedir, 'scratch')
        years = range(self.minyear, self.maxyear + 1)

        data_source = data_params['datasource']
        if data_source not in ['ref', 'sim']:
            logging.error(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")
            raise ValueError(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")

        if self.ref_data_type != 'stn' and self.sim_data_type != 'stn':
            Parallel(n_jobs=self.num_cores)(
                delayed(self._make_grid_parallel)(data_source,
                                                  data_params['suffix'],
                                                  data_params['prefix'],
                                                  data_dir, year)
                for year in years
            )
            var_files = glob.glob(os.path.join(self.casedir, 'tmp', f'{data_source}_{data_params["varname"][0]}_remap_*.nc'))
        else:
            var_files = glob.glob(os.path.join(data_dir, f'{data_source}_{data_params["prefix"]}*{data_params["suffix"]}.nc'))

        self.combine_and_save_data(var_files, data_params)

    @performance_monitor
    def combine_and_save_data(self, var_files: List[str], data_params: Dict[str, Any]) -> None:
        with xr.open_mfdataset(var_files, combine='by_coords') as ds:
            ds = ds.sortby('time')
            output_file = self.get_output_filename(data_params)
            with ProgressBar():
                ds.to_netcdf(output_file)
            gc.collect()  # Add garbage collection after saving combined data

        # Only cleanup temp files if we created them (i.e., when processing grid data)
        if self.ref_data_type != 'stn' and self.sim_data_type != 'stn':
            self.cleanup_temp_files(data_params)

    def get_output_filename(self, data_params: Dict[str, Any]) -> str:
        if data_params['datasource'] == 'ref':
            return os.path.join(self.casedir, 'output', 'data', f'{self.item}_{data_params["datasource"]}_{self.ref_source}_{data_params["varname"][0]}.nc')
        else:
            return os.path.join(self.casedir, 'output', 'data', f'{self.item}_{data_params["datasource"]}_{self.sim_source}_{data_params["varname"][0]}.nc')

    def cleanup_temp_files(self, data_params: Dict[str, Any]) -> None:
        """Clean up temporary files, silently skipping non-existent files."""
        failed_removals = []
        for year in range(self.minyear, self.maxyear + 1):
            temp_file = os.path.join(self.casedir, 'tmp', f'{data_params["datasource"]}_{data_params["varname"][0]}_remap_{year}.nc')
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logging.debug(f"Removed temporary file: {temp_file}")
                except OSError as e:
                    failed_removals.append((temp_file, str(e)))
        
        # Only warn if we actually failed to remove existing files
        if failed_removals:
            logging.warning(f"Failed to remove {len(failed_removals)} temporary file(s)")
            for file_path, error in failed_removals:
                logging.debug(f"  Failed to remove {file_path}: {error}")

    def extract_station_data_if_needed(self, data_params: Dict[str, Any]) -> None:
        if self.ref_data_type == 'stn' or self.sim_data_type == 'stn':
            logging.info(f"Extracting station data for {data_params['datasource']} data")
            self.extract_station_data(data_params)

    def extract_station_data(self, data_params: Dict[str, Any]) -> None:
        output_file = self.get_output_filename(data_params)
        with xr.open_dataset(output_file) as ds:
            ds = Convert_Type.convert_nc(ds)
            Parallel(n_jobs=-1)(
                delayed(self._extract_stn_parallel)(
                    data_params['datasource'], ds, self.station_list, i
                ) for i in range(len(self.station_list['ID']))
            )
            gc.collect()  # Add garbage collection after extracting station data
        os.remove(output_file)

    @performance_monitor
    def _make_grid_parallel(self, data_source: str, suffix: str, prefix: str, dirx: str, year: int) -> None:
        try:
            if data_source not in ['ref', 'sim']:
                logging.error(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")
                raise ValueError(f"Invalid data_source: {data_source}. Expected 'ref' or 'sim'.")

            var_file = os.path.join(dirx, f'{data_source}_{prefix}{year}{suffix}.nc')
            if self.debug_mode:
                logging.info(f"Processing {var_file} for year {year}")
                logging.info(f"Processing {data_source} data for year {year}")

            with xr.open_dataset(var_file) as data:
                data = Convert_Type.convert_nc(data)
                data = self.preprocess_grid_data(data)
                remapped_data = self.remap_data(data)
                self.save_remapped_data(remapped_data, data_source, year)
        finally:
            gc.collect()

    @performance_monitor
    @cached(key_prefix="preprocess_grid_data", ttl=3600)  # Cache for 1 hour
    def preprocess_grid_data(self, data: xr.Dataset) -> xr.Dataset:
        # Check if lon and lat are 2D
        data = self.check_coordinate(data)
        if data['lon'].ndim == 2 and data['lat'].ndim == 2:
            try:
                from regrid.regrid_wgs84 import convert_to_wgs84_xesmf
                data = convert_to_wgs84_xesmf(data, self.compare_grid_res)
            except:
                from regrid.regrid_wgs84 import convert_to_wgs84_scipy
                data = convert_to_wgs84_scipy(data, self.compare_grid_res)

        # Convert longitude values
        lon = data['lon'].values
        lon_adjusted = np.where(lon > 180, lon - 360, lon)

        # Create a new DataArray with adjusted longitude values
        new_lon = xr.DataArray(lon_adjusted, dims='lon', attrs=data['lon'].attrs)

        # Assign the new longitude to the dataset
        data = data.assign_coords(lon=new_lon)

        # If needed, sort the dataset by the new longitude values
        data = data.sortby('lon')

        return data

    @performance_monitor
    def remap_data(self, data: xr.Dataset) -> xr.Dataset:
        new_grid = self.create_target_grid()

        remapping_methods = [
            self.remap_interpolate,
            self.remap_xesmf,
            self.remap_cdo
        ]

        for method in remapping_methods:
            try:
                return method(data, new_grid)
            except Exception as e:
                logging.warning(f"{method.__name__} failed: {e}")

        # If all remapping methods fail, try basic interpolation as fallback
        logging.warning("All remapping methods failed, trying basic interpolation")
        try:
            return self.remap_basic_interpolation(data, new_grid)
        except Exception as e:
            logging.warning(f"Basic interpolation also failed: {e}")
            logging.warning("Returning original data without remapping")
            return data

    @performance_monitor
    @cached(key_prefix="create_target_grid", ttl=7200)  # Cache for 2 hours (grid rarely changes)
    def create_target_grid(self) -> xr.Dataset:
        lon_new = np.arange(self.min_lon + self.compare_grid_res / 2, self.max_lon, self.compare_grid_res)
        lat_new = np.arange(self.min_lat + self.compare_grid_res / 2, self.max_lat, self.compare_grid_res)
        return xr.Dataset({'lon': lon_new, 'lat': lat_new})

    @performance_monitor
    def remap_interpolate(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        from openbench.data.regrid import Grid, Regridder
        grid = Grid(
            north=self.max_lat - self.compare_grid_res / 2,
            south=self.min_lat + self.compare_grid_res / 2,
            west=self.min_lon + self.compare_grid_res / 2,
            east=self.max_lon - self.compare_grid_res / 2,
            resolution_lat=self.compare_grid_res,
            resolution_lon=self.compare_grid_res,
        )
        target_dataset = grid.create_regridding_dataset(lat_name="lat", lon_name="lon")
        # Convert sparse arrays to dense arrays
        data_regrid = data.regrid.conservative(target_dataset, nan_threshold=0)
        # data_regrid = data_regrid.compute()

        return data_regrid
        # target_dataset = grid.create_regridding_dataset(lat_name="lat", lon_name="lon")
        # return data.regrid.conservative(target_dataset,nan_threshold=0)

    @performance_monitor
    def remap_xesmf(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        import xesmf as xe
        regridder = xe.Regridder(data, new_grid, 'conservative')
        return regridder(data)

    @performance_monitor
    def remap_cdo(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.nc') as temp_input, \
                tempfile.NamedTemporaryFile(suffix='.nc') as temp_output, \
                tempfile.NamedTemporaryFile(suffix='.txt') as temp_grid:
            data.to_netcdf(temp_input.name)
            self.create_target_grid_file(temp_grid.name, new_grid)

            cmd = f"cdo -s remapcon,{temp_grid.name} {temp_input.name} {temp_output.name}"
            subprocess.run(cmd, shell=True, check=True)

            return Convert_Type.convert_nc(xr.open_dataset(temp_output.name))

    def create_target_grid_file(self, filename: str, new_grid: xr.Dataset) -> None:
        with open(filename, 'w') as f:
            f.write(f"gridtype = lonlat\n")
            f.write(f"xsize = {len(new_grid.lon)}\n")
            f.write(f"ysize = {len(new_grid.lat)}\n")
            f.write(f"xfirst = {self.min_lon + self.compare_grid_res / 2}\n")
            f.write(f"xinc = {self.compare_grid_res}\n")
            f.write(f"yfirst = {self.min_lat + self.compare_grid_res / 2}\n")
            f.write(f"yinc = {self.compare_grid_res}\n")

    def save_remapped_data(self, data: xr.Dataset, data_source: str, year: int) -> None:
        try:
            # Check if data is None (all regrid methods failed)
            if data is None:
                logging.warning(f"No data to save for {data_source} year {year} - all regrid methods failed")
                return
            
            # Convert sparse arrays to dense arrays
            data = data.resample(time=self.compare_tim_res).mean()
            data = data.sel(time=slice(f'{year}-01-01T00:00:00', f'{year}-12-31T23:59:59'))

            varname = self.ref_varname[0] if data_source == 'ref' else self.sim_varname[0]

            out_file = os.path.join(self.casedir, 'tmp', f'{data_source}_{varname}_remap_{year}.nc')
            data.to_netcdf(out_file)
            logging.info(f"Saved remapped {data_source} data for year {year} to {out_file}")
        finally:
            if hasattr(data, 'close'):
                data.close()
            gc.collect()


class DatasetProcessing(StationDatasetProcessing, GridDatasetProcessing):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Cache system will be initialized on-demand to avoid serialization issues
        self._cache_initialized = False
    
    def _get_cache(self):
        """Get cache manager with lazy initialization."""
        if not self._cache_initialized:
            try:
                self.cache_manager = get_cache_manager()
                self.data_cache = DataCache(self.cache_manager)
                self._cache_initialized = True
                logging.debug("Cache system lazily initialized")
            except Exception as e:
                logging.error(f"Failed to initialize required cache system: {e}")
                raise RuntimeError(f"CacheSystem initialization failed: {e}")
        return self.cache_manager

    def process(self, datasource: str) -> None:
        super().process(datasource)
        # Add any additional processing specific to this class if needed
