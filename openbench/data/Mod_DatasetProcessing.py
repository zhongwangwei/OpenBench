import glob
import importlib
import logging
import os
import re
import shutil
import gc
import time
import functools
from typing import List, Dict, Any, Tuple, Callable, Union

# Import cached glob for performance
try:
    from openbench.util.Mod_DatasetLoader import cached_glob
except ImportError:
    # Fallback to standard glob
    cached_glob = lambda pattern, **kwargs: sorted(glob.glob(pattern))

# Try to import psutil for performance monitoring, use fallback if not available
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logging.warning("psutil not available. Performance monitoring will be limited.")

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
except (AttributeError, ValueError, IndexError):
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

def performance_monitor(func: Callable = None, *, silent_on_error: bool = False) -> Callable:
    """Enhanced decorator to monitor function performance with error handling.

    Args:
        func: The function to decorate
        silent_on_error: If True, don't log errors (only re-raise them)
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get initial memory usage
            start_time = time.time()
            if _HAS_PSUTIL:
                try:
                    process = psutil.Process(os.getpid())
                    start_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
                    start_cpu = process.cpu_percent()
                except Exception as e:
                    logging.warning(f"Failed to get system resources: {e}")
                    start_mem = 0
                    start_cpu = 0
            else:
                start_mem = 0
                start_cpu = 0

            try:
                # Execute the function
                result = f(*args, **kwargs)

                # Calculate execution time and memory usage
                end_time = time.time()
                execution_time = end_time - start_time

                if _HAS_PSUTIL:
                    try:
                        end_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
                        end_cpu = process.cpu_percent()
                        memory_used = end_mem - start_mem
                        cpu_used = end_cpu - start_cpu
                    except Exception as e:
                        logging.warning(f"Failed to get end system resources: {e}")
                        memory_used = 0
                        cpu_used = 0
                else:
                    memory_used = 0
                    cpu_used = 0

                # Log performance
                logging.debug(f"Performance for {f.__name__}:")
                logging.debug(f"  Execution time: {execution_time:.2f} seconds")
                if _HAS_PSUTIL:
                    logging.debug(f"  Memory usage: {memory_used:.3f} GB")
                    logging.debug(f"  CPU usage: {cpu_used:.1f}%")

                # Use enhanced performance warning if available
                # Disabled to reduce log noise
                # if _HAS_EXCEPTIONS:
                #     log_performance_warning(f.__name__, execution_time)

                # Log warning if memory usage is high
                if _HAS_PSUTIL:
                    try:
                        total_memory = psutil.virtual_memory().total / (1024 ** 3)
                        if memory_used > 0.8 * total_memory:  # 80% of total memory
                            logging.warning(f"High memory usage detected in {f.__name__}: {memory_used:.3f} GB")
                    except Exception as e:
                        logging.debug(f"Failed to check total memory: {e}")

                return result

            except Exception as e:
                # Log error with performance context (unless silent)
                end_time = time.time()
                execution_time = end_time - start_time

                if _HAS_PSUTIL:
                    try:
                        end_mem = process.memory_info().rss / 1024 / 1024 / 1024
                        memory_used = end_mem - start_mem
                    except Exception:
                        memory_used = 0
                else:
                    memory_used = 0

                if not silent_on_error:
                    logging.error(f"Error in {f.__name__} after {execution_time:.2f}s and using {memory_used:.3f} GB:")
                    logging.error(str(e))
                raise

        return wrapper

    # Support both @performance_monitor and @performance_monitor(silent_on_error=True)
    if func is None:
        return decorator
    else:
        return decorator(func)


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

    @staticmethod
    def validate_year(year, default=None, min_year=1900, max_year=2100):
        """Validate and convert year value to integer within valid range.

        Args:
            year: Year value to validate (can be int, float, str, or None)
            default: Default value if year is invalid
            min_year: Minimum valid year (default 1900)
            max_year: Maximum valid year (default 2100)

        Returns:
            Valid integer year value
        """
        # Handle None, empty string, or whitespace
        if year is None or (isinstance(year, str) and year.strip() == ''):
            return default if default is not None else min_year

        # Try to convert to integer
        try:
            year_int = int(float(year))
        except (ValueError, TypeError):
            logging.warning(f"Invalid year value: {year}. Using default: {default if default else min_year}")
            return default if default is not None else min_year

        # Validate range
        if year_int < min_year:
            logging.warning(f"Year {year_int} is before {min_year}. Adjusting to {min_year}.")
            return min_year
        if year_int > max_year:
            logging.warning(f"Year {year_int} is after {max_year}. Adjusting to {max_year}.")
            return max_year

        return year_int

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
            'M': 'ME',  # Month end
            'Y': 'YE',  # Year end
            'Q': 'QE',  # Quarter end
            'H': 'h',  # Hour
            'T': 'min',  # Minute
            'S': 's',  # Second
            'L': 'ms',  # Millisecond
            'U': 'us',  # Microsecond
            'N': 'ns',  # Nanosecond
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

    def _is_climatology_mode(self) -> bool:
        """
        Check if compare_tim_res indicates climatology mode.
        
        Returns:
            bool: True if in climatology mode (climatology-year or climatology-month)
        """
        if not hasattr(self, 'compare_tim_res') or not self.compare_tim_res:
            return False
        compare_tim_res_str = str(self.compare_tim_res).strip().lower()
        return compare_tim_res_str in ['climatology-year', 'climatology-month']

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
                'month': 'ME',  # Month End (new alias)
                'mon': 'ME',
                'monthly': 'ME',
                'day': 'D',
                'daily': 'D',
                'hour': 'h',  # Hour (lowercase in new pandas)
                'Hour': 'h',
                'hr': 'h',
                'Hr': 'h',
                'h': 'h',
                'hourly': 'h',
                'year': 'YE',  # Year End (new alias)
                'yr': 'YE',
                'yearly': 'YE',
                'week': 'W',
                'wk': 'W',
                'weekly': 'W',
            }
        else:
            freq_map = {
                'month': 'M',  # Month (old alias)
                'mon': 'M',
                'monthly': 'M',
                'day': 'D',
                'daily': 'D',
                'hour': 'H',  # Hour (uppercase in old pandas)
                'Hour': 'H',
                'hr': 'H',
                'Hr': 'H',
                'h': 'H',
                'hourly': 'H',
                'year': 'Y',  # Year (old alias)
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
        # Set default values for optional config keys before updating
        self.debug_mode = False  # Default debug_mode to False
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
            # Use ref_fulllist if available, otherwise use dataset-specific filename
            if hasattr(self, 'ref_fulllist') and self.ref_fulllist and os.path.exists(self.ref_fulllist):
                stnlist_path = self.ref_fulllist
            else:
                stnlist_path = os.path.join(self.casedir, f"stn_{self.ref_source}_{self.sim_source}_list.txt")
            self.station_list = Convert_Type.convert_Frame(pd.read_csv(stnlist_path, header=0))
            output_dir = os.path.join(self.casedir, 'data', f'stn_{self.ref_source}_{self.sim_source}')
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
            logging.debug(f"Processing {datasource} data")
            self._preprocess(datasource)
            logging.debug(f"{datasource.capitalize()} data prepared!")
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
            logging.debug(f"Processing {data_params['data_type']} data")
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
            lon_vals = ds['lon'].values
            # Check if longitude needs conversion (0-360 to -180-180)
            if lon_vals.max() > 180:
                # Assign new lon values
                ds = ds.assign_coords(lon=(ds['lon'] + 180) % 360 - 180)
                # Sort by lon to properly align data with new coordinates
                ds = ds.sortby('lon')
                # Update valid_min/valid_max attributes to match new coordinate range
                if 'valid_min' in ds['lon'].attrs:
                    ds['lon'].attrs['valid_min'] = -180.0
                if 'valid_max' in ds['lon'].attrs:
                    ds['lon'].attrs['valid_max'] = 180.0
        return ds

    def check_time(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str) -> xr.Dataset:
        # Validate year values
        syear = self.validate_year(syear, default=1990)
        eyear = self.validate_year(eyear, default=2020)

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
            except (ValueError, TypeError) as e:
                try:
                    ds1 = xr.Dataset({f'{ds.name}': (['time', 'lon', 'lat'], data)},
                                     coords={'time': time_index, 'lon': lon, 'lat': lat})
                except (ValueError, TypeError) as e2:
                    ds1 = xr.Dataset({f'{ds.name}': (['lat', 'lon', 'time'], data)},
                                     coords={'lat': lat, 'lon': lon, 'time': time_index})
            ds1 = ds1.transpose('time', 'lat', 'lon')
            return ds1[f'{ds.name}']

        if not hasattr(ds['time'], 'dt'):
            try:
                ds['time'] = pd.to_datetime(ds['time'])
            except (ValueError, TypeError, AttributeError) as e:
                lon = ds.lon.values
                lat = ds.lat.values
                data = ds.values
                time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
                try:
                    ds1 = xr.Dataset({f'{ds.name}': (['time', 'lat', 'lon'], data)},
                                     coords={'time': time_index, 'lat': lat, 'lon': lon})
                except (ValueError, TypeError) as e:
                    try:
                        ds1 = xr.Dataset({f'{ds.name}': (['time', 'lon', 'lat'], data)},
                                         coords={'time': time_index, 'lon': lon, 'lat': lat})
                    except (ValueError, TypeError) as e2:
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
        except (ValueError, KeyError) as e:
            try:
                return ds.transpose('time', 'lat', 'lon')
            except (ValueError, KeyError) as e2:
                try:
                    return ds.transpose('time', 'lon', 'lat')
                except (ValueError, KeyError) as e3:
                    return ds.squeeze()

    @performance_monitor
    # NOTE: @cached removed - cache key collisions caused race conditions
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
    # NOTE: @cached removed - cache key collisions caused race conditions
    def make_time_integrity(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str, datasource: str) -> xr.Dataset:
        # Validate year values
        syear = self.validate_year(syear, default=1990)
        eyear = self.validate_year(eyear, default=2020)

        match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
        if match:
            num_value, time_unit = match.groups()
            num_value = int(num_value) if num_value else 1
            time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
            if time_unit.lower() in ['m', 'month', 'mon','me']:
                # Normalize to monthly resolution without enforcing a specific day match.
                # Set times to 15th for plotting/consistency, but do NOT reindex/fill missing months.
                # Compare by month presence only.
                mid_month = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-15T00:00:00'))
                try:
                    ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-15T00:00:00'))
                except (ValueError, AttributeError, TypeError):
                    # If we cannot format, keep existing times but ensure datetime type
                    try:
                        ds['time'] = pd.to_datetime(ds['time'].values)
                        ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-15T00:00:00'))
                    except Exception:
                        ds['time'] = mid_month
                # Use mid-month index for monthly comparison/fill
                time_index = mid_month
                time_var = ds.time
                # Remove potential duplicate timestamps created by monthly normalization
                try:
                    _, index_unique = np.unique(ds['time'], return_index=True)
                    ds = ds.isel(time=np.sort(index_unique))
                    time_var = ds.time
                except Exception:
                    pass
                # Safely set calendar attribute using encoding
                try:
                    time_var.encoding['calendar'] = 'proleptic_gregorian'
                except Exception:
                    try:
                        new_time = xr.DataArray(
                            time_var.values,
                            dims=['time'],
                            attrs={'calendar': 'proleptic_gregorian'}
                        )
                        ds = ds.assign_coords(time=new_time)
                        time_var = ds.time
                    except Exception:
                        logging.debug("Could not set calendar attribute for monthly data; proceeding.")
                        pass
                # For monthly data: only check by Year-Month presence and avoid reindexing/filling.
                expected_months = pd.period_range(start=f'{syear}-01', end=f'{eyear}-12', freq='M')
                # Build a PeriodIndex from the existing timestamps for robust monthly comparison
                try:
                    present_months = pd.PeriodIndex(pd.to_datetime(ds['time'].values), freq='M')
                except Exception:
                    # Fallback: coerce via Series then to_period
                    present_months = pd.to_datetime(pd.Series(ds['time'].values))
                    present_months = present_months.dt.to_period('M')
                missing_months = expected_months.difference(pd.PeriodIndex(present_months, freq='M'))
                # If months are missing, reindex to monthly midpoints and fill with NaN for missing months
                if len(missing_months) > 0:
                    logging.info(f"Monthly data has {len(missing_months)} missing month(s) between {syear} and {eyear}; filling missing months with NaN.")
                    ds = ds.reindex(time=time_index)
                return ds
            elif time_unit.lower() in ['d', 'day', '1d', '1day']:
                # Normalize to daily resolution (set to 12:00), and fill missing days by reindexing.
                day_noon = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-%m-%dT12:00:00'))
                try:
                    ds['time'] = pd.to_datetime(ds['time'].dt.floor('D').dt.strftime('%Y-%m-%dT12:00:00'))
                except (ValueError, AttributeError, TypeError):
                    try:
                        ds['time'] = pd.to_datetime(ds['time'].values)
                        ds['time'] = pd.to_datetime(pd.Series(ds['time']).dt.floor('D').dt.strftime('%Y-%m-%dT12:00:00'))
                    except Exception:
                        ds['time'] = day_noon
                # Use daily noon index for comparison/fill
                time_index = day_noon
                time_var = ds.time
                # Remove duplicates potentially created by normalization
                try:
                    _, index_unique = np.unique(ds['time'], return_index=True)
                    ds = ds.isel(time=np.sort(index_unique))
                    time_var = ds.time
                except Exception:
                    pass
                # Safely set calendar attribute using encoding
                try:
                    time_var.encoding['calendar'] = 'proleptic_gregorian'
                except Exception:
                    try:
                        new_time = xr.DataArray(
                            time_var.values,
                            dims=['time'],
                            attrs={'calendar': 'proleptic_gregorian'}
                        )
                        ds = ds.assign_coords(time=new_time)
                        time_var = ds.time
                    except Exception:
                        logging.debug("Could not set calendar attribute for daily data; proceeding.")
                        pass
                # Check by date presence and fill missing by reindexing
                expected_days = pd.period_range(start=f'{syear}-01-01', end=f'{eyear}-12-31', freq='D')
                try:
                    present_days = pd.PeriodIndex(pd.to_datetime(ds['time'].values), freq='D')
                except Exception:
                    # Fallback: coerce to datetime first, then derive daily periods safely.
                    fallback_days = pd.to_datetime(pd.Series(ds['time'].values))
                    fallback_days = fallback_days.dt.to_period('D')
                    present_days = pd.PeriodIndex(fallback_days, freq='D')
                missing_days = expected_days.difference(present_days)
                if len(missing_days) > 0:
                    logging.info(f"Daily data has {len(missing_days)} missing day(s) between {syear} and {eyear}; filling missing days with NaN.")
                    ds = ds.reindex(time=time_index)
                return ds
            elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
                # Normalize to hourly resolution (set to HH:30), and fill missing hours by reindexing.
                hour_mid = pd.to_datetime(pd.Series(time_index).dt.floor('H').dt.strftime('%Y-%m-%dT%H:30:00'))
                try:
                    ds['time'] = pd.to_datetime(pd.Series(ds['time']).dt.floor('H').dt.strftime('%Y-%m-%dT%H:30:00'))
                except (ValueError, AttributeError, TypeError):
                    try:
                        ds['time'] = pd.to_datetime(ds['time'].values)
                        ds['time'] = pd.to_datetime(pd.Series(ds['time']).dt.floor('H').dt.strftime('%Y-%m-%dT%H:30:00'))
                    except Exception:
                        ds['time'] = hour_mid
                # Use mid-hour index for comparison/fill
                time_index = hour_mid
                time_var = ds.time
                # Remove duplicates potentially created by normalization
                try:
                    _, index_unique = np.unique(ds['time'], return_index=True)
                    ds = ds.isel(time=np.sort(index_unique))
                    time_var = ds.time
                except Exception:
                    pass
                # Safely set calendar attribute using encoding
                try:
                    time_var.encoding['calendar'] = 'proleptic_gregorian'
                except Exception:
                    try:
                        new_time = xr.DataArray(
                            time_var.values,
                            dims=['time'],
                            attrs={'calendar': 'proleptic_gregorian'}
                        )
                        ds = ds.assign_coords(time=new_time)
                        time_var = ds.time
                    except Exception:
                        logging.debug("Could not set calendar attribute for hourly data; proceeding.")
                        pass
                # Check by hour presence and fill missing by reindexing
                expected_hours = pd.period_range(start=f'{syear}-01-01', end=f'{eyear}-12-31 23:00:00', freq='H')
                try:
                    present_hours = pd.PeriodIndex(pd.to_datetime(ds['time'].values), freq='H')
                except Exception:
                    present_hours = pd.to_datetime(pd.Series(ds['time'].values))
                    present_hours = present_hours.dt.to_period('H')
                missing_hours = expected_hours.difference(pd.PeriodIndex(present_hours, freq='H'))
                if len(missing_hours) > 0:
                    logging.info(f"Hourly data has {len(missing_hours)} missing hour(s) between {syear} and {eyear}; filling missing hours with NaN.")
                    ds = ds.reindex(time=time_index)
                return ds
            elif time_unit.lower() in ['y', 'year', '1y', '1year']:
                time_index = pd.to_datetime(pd.Series(time_index).dt.strftime('%Y-01-01T00:00:00'))
                try:
                    ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-01-01T00:00:00'))
                except (ValueError, AttributeError, TypeError):
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
            missing_times = time_index[~np.isin(time_index, time_values)]
            if len(missing_times) > 0 and len(missing_times) < len(time_var):
                logging.warning("Time series is not complete. Missing time values found.")
                logging.info('Filling missing time values with np.nan')
                # Fill missing time values with np.nan
                ds = ds.reindex(time=time_index)
                ds = ds.where(ds.time.isin(time_values), np.nan)
        return ds

    @performance_monitor
    def apply_model_specific_time_adjustment(self, ds: xr.Dataset, datasource: str, syear: int, eyear: int,
                                             tim_res: str) -> xr.Dataset:
        # Get model name from _model attribute (e.g., TE-routing_model = "TE")
        source = self.sim_source if datasource == 'sim' else self.ref_source
        try:
            model = getattr(self, f"{source}_model")
        except AttributeError:
            model = source
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
        ds = None
        try:
            try:
                ds = xr.open_dataset(VarFile)  # .squeeze()
            except (ValueError, OSError) as e:
                ds = xr.open_dataset(VarFile, decode_times=False)  # .squeeze()
        except Exception as e:
            logging.error(f"Failed to open dataset: {VarFile}")
            logging.error(f"Error: {str(e)}")
            if ds is not None:
                ds.close()
            raise

        try:
            ds = self.apply_custom_filter(datasource, ds, varname)
            ds = Convert_Type.convert_nc(ds)
        except Exception as e:
            # Check if varname list is empty
            if not varname or len(varname) == 0:
                logging.error("Variable name list is empty")
                raise ValueError("Variable name list cannot be empty")

            # Check if variable exists in dataset
            if varname[0] not in ds:
                available_vars = list(ds.data_vars) + list(ds.coords)
                logging.error(f"Variable '{varname[0]}' not found in dataset")
                logging.error(f"Available variables: {available_vars}")
                raise KeyError(f"Variable '{varname[0]}' not in dataset")

            ds = Convert_Type.convert_nc(ds[varname[0]])
        return ds

    def apply_custom_filter(self, datasource: str, ds: xr.Dataset, varname: List) -> xr.Dataset:
        if datasource == 'stat':
            # Validate varname list is not empty
            if not varname or len(varname) == 0:
                raise ValueError("Variable name list cannot be empty for station data")

            # Validate variable exists in dataset
            if varname[0] not in ds:
                available_vars = list(ds.data_vars) + list(ds.coords)
                raise KeyError(f"Variable '{varname[0]}' not found in station dataset. Available: {available_vars}")

            return ds[varname[0]]
        else:
            # Get model name from _model attribute (e.g., TE-routing_model = "TE")
            source = self.sim_source if datasource == 'sim' else self.ref_source
            try:
                model = getattr(self, f"{source}_model")
            except AttributeError:
                model = source
            try:
                logging.info(f"Loading custom variable filter for {model}")
                custom_module = importlib.import_module(f"openbench.data.custom.{model}_filter")
                custom_filter = getattr(custom_module, f"filter_{model}")
                self, ds_or_da = custom_filter(self, ds)

                # If filter returned a Dataset, extract the variable; if DataArray, use directly
                if isinstance(ds_or_da, xr.Dataset):
                    current_varname = getattr(self, f"{datasource}_varname")
                    var_to_extract = current_varname[0] if isinstance(current_varname, list) else current_varname
                    if var_to_extract in ds_or_da:
                        return ds_or_da[var_to_extract]
                    else:
                        raise KeyError(f"Variable {var_to_extract} not found in filtered dataset")
                else:
                    return ds_or_da
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
            logging.error(f"Error: Invalid time range (syear={syear}, eyear={eyear})")
            raise ValueError(f"Invalid time range: eyear ({eyear}) must be >= syear ({syear})")
        else:
            return ds.sel(time=slice(f'{syear}-01-01T00:00:00', f'{eyear}-12-31T23:59:59'))

    @performance_monitor
    # NOTE: @cached removed - cache key collisions caused race conditions
    def resample_data(self, dfx1: xr.Dataset, tim_res: str, startx: int, endx: int) -> xr.Dataset:
        # Check if climatology mode - skip resampling
        tim_res_lower = str(tim_res).strip().lower()
        if tim_res_lower in ['climatology-year', 'climatology-month']:
            logging.debug(f"resample_data: Climatology mode detected ({tim_res}), returning data unchanged")
            return dfx1
            
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
                logging.debug(f"Saved {output_file}")
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
            logging.debug(f"Using {optimal_cores} cores for splitting years")

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
        # Try primary path first (use cached glob for performance)
        var_files = cached_glob(os.path.join(dirx, f'{prefix}{year}*{suffix}.nc'))

        # Try alternative path if no files found
        if not var_files:
            var_files = cached_glob(os.path.join(dirx, str(year), f'{prefix}{year}*{suffix}.nc'))

        # Filter files: only keep files where the part between prefix+year and suffix contains no letters
        # This prevents matching files like "prefix_cama_year" when we want "prefix_year"
        # E.g., "FUXI-test_hist_2006-01.nc" ✓, "FUXI-test_hist_cama_2006-01.nc" ✗
        if var_files:
            filtered_files = []
            # Escape special regex characters in prefix and suffix
            prefix_escaped = re.escape(prefix)
            suffix_escaped = re.escape(suffix) if suffix else ''
            # Pattern: prefix + year + (only digits and symbols, no letters) + suffix + .nc
            pattern = re.compile(rf'^{prefix_escaped}{year}[^a-zA-Z]*{suffix_escaped}\.nc$')
            for f in var_files:
                filename = os.path.basename(f)
                if pattern.match(filename):
                    filtered_files.append(f)
                else:
                    logging.debug(f"Filtered out file (contains letters after year): {filename}")
            var_files = filtered_files

        # Verify files were found
        if not var_files:
            logging.error(f"No files found for year {year} with prefix '{prefix}' and suffix '{suffix}'")
            logging.error(f"Searched in: {dirx}")
            logging.error(f"  - Pattern 1: {os.path.join(dirx, f'{prefix}{year}*{suffix}.nc')}")
            logging.error(f"  - Pattern 2: {os.path.join(dirx, str(year), f'{prefix}{year}*{suffix}.nc')}")
            raise FileNotFoundError(f"No data files found for year {year}")

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
        logging.debug('The dataset groupby is Single --> split it to Year')
        varfile = self.check_file_exist(os.path.join(dirx, f'{prefix}{suffix}.nc'))
        ds = self.select_var(syear, eyear, tim_res, varfile, varname, datasource)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        ds = self.select_timerange(ds, self.minyear, self.maxyear)
        # Use updated varunit from filter if available (filter may have modified it)
        current_varunit = getattr(self, f"{datasource}_varunit", varunit)
        ds, varunit = self.process_units(ds, current_varunit)
        self.split_year(ds, casedir, suffix, prefix, self.minyear, self.maxyear, datasource)

    @performance_monitor
    def preprocess_non_yearly_files(self, dirx: str, syear: int, eyear: int, tim_res: str, varunit: str, varname: List[str],
                                    casedir: str, suffix: str, prefix: str, datasource: str) -> None:
        logging.debug('The dataset groupby is not Year --> combine it to Year')
        ds = self.combine_year(syear, casedir, dirx, suffix, prefix, varname, datasource, tim_res)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        # Use updated varunit from filter if available (filter may have modified it)
        current_varunit = getattr(self, f"{datasource}_varunit", varunit)
        ds, varunit = self.process_units(ds, current_varunit)
        ds = self.select_timerange(ds, syear, eyear)
        ds.to_netcdf(os.path.join(casedir, 'scratch', f'{datasource}_{prefix}{syear}{suffix}.nc'))

    @performance_monitor
    def preprocess_yearly_files(self, dirx: str, syear: int, eyear: int, tim_res: str, varunit: str, varname: List[str],
                                casedir: str, suffix: str, prefix: str, datasource: str) -> None:
        varfiles = self.check_file_exist(os.path.join(dirx, f'{prefix}{syear}{suffix}.nc'))
        ds = self.select_var(syear, eyear, tim_res, varfiles, varname, datasource)
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, tim_res, datasource)
        # Use updated varunit from filter if available (filter may have modified it)
        current_varunit = getattr(self, f"{datasource}_varunit", varunit)
        ds, varunit = self.process_units(ds, current_varunit)
        ds = self.select_timerange(ds, syear, eyear)
        ds.to_netcdf(os.path.join(casedir, 'scratch', f'{datasource}_{prefix}{syear}{suffix}.nc'))

    @performance_monitor
    # NOTE: @cached removed - cache key collisions caused race conditions
    # Different datasets with same structure produced identical keys
    def process_units(self, ds: xr.Dataset, varunit: str) -> Tuple[xr.Dataset, str]:
        try:
            # 确保我们获取实际的数据数组而不是方法
            if isinstance(ds, xr.Dataset):
                # 如果是数据集，获取第一个变量的数据
                data_vars_list = list(ds.data_vars)
                if not data_vars_list:
                    logging.error("Dataset has no data variables")
                    raise ValueError("Dataset must contain at least one data variable")
                var_name = data_vars_list[0]
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
            logging.debug(f"Converted unit from {varunit} to {new_unit}")

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
            logging.debug("Processing station data")
            if not hasattr(self, 'station_list') or self.station_list is None or self.station_list.empty:
                logging.error("Station list is empty; cannot process station data.")
                return

            indices = range(len(self.station_list['ID']))
            try:
                Parallel(n_jobs=-1)(
                    delayed(self._make_stn_parallel)(
                        self.station_list, data_params['datasource'], i
                    ) for i in indices
                )
            except (PermissionError, OSError) as exc:
                logging.warning(
                    "Parallel station processing unavailable (%s). Falling back to sequential execution.",
                    exc
                )
                for i in indices:
                    self._make_stn_parallel(self.station_list, data_params['datasource'], i)
        finally:
            gc.collect()

    def process_single_station_data(self, stn_data: xr.Dataset, start_year: int, end_year: int, datasource: str) -> xr.Dataset:
        var_attr = self.ref_varname if datasource == 'ref' else self.sim_varname
        var_attr_is_list = isinstance(var_attr, list)

        # Work on a copy of the current variable list so that temporary
        # fallbacks do not mutate the original configuration.
        current_var_list = list(var_attr) if var_attr_is_list else [var_attr]
        original_var_list = list(var_attr) if var_attr_is_list else [var_attr]
        original_varname = original_var_list[0] if original_var_list else None

        try:
            # Validate varname list is not empty
            if not current_var_list:
                logging.error("Variable name list is empty")
                raise ValueError("Variable name list cannot be empty for station data")

            # Check if the variable exists in the dataset
            if current_var_list[0] not in stn_data:
                # Try to apply custom filter for variable fallback
                # Get model name from _model attribute (e.g., TE-routing_model = "TE")
                source = self.sim_source if datasource == 'sim' else self.ref_source
                try:
                    model = getattr(self, f"{source}_model")
                except AttributeError:
                    model = source
                try:
                    import importlib
                    custom_module = importlib.import_module(f"openbench.data.custom.{model}_filter")
                    custom_filter = getattr(custom_module, f"filter_{model}")
                    
                    # Call custom filter with dataset
                    logging.info(f"Variable '{current_var_list[0]}' not found, trying custom filter for {model}")
                    updated_self, filtered_data = custom_filter(self, stn_data)

                    # If custom filter handled it, use the updated info and data
                    if updated_self is not None and filtered_data is not None:
                        # Update varname based on what the filter set (ensure copy)
                        if datasource == 'ref':
                            new_var_attr = self.ref_varname
                        else:
                            new_var_attr = self.sim_varname
                        current_var_list = list(new_var_attr) if isinstance(new_var_attr, list) else [new_var_attr]
                        logging.info(f"Custom filter updated variable to: {current_var_list[0]}")
                        ds = filtered_data
                    else:
                        # Custom filter didn't handle this case, raise error
                        available_vars = list(stn_data.data_vars) + list(stn_data.coords)
                        logging.error(f"Variable '{current_var_list[0]}' not found in the station data.")
                        logging.error(f"Available variables: {available_vars}")
                        raise ValueError(f"Variable '{current_var_list[0]}' not found in the station data.")
                except (ImportError, AttributeError):
                    # No custom filter available, raise original error
                    available_vars = list(stn_data.data_vars) + list(stn_data.coords)
                    logging.error(f"Variable '{current_var_list[0]}' not found in the station data.")
                    logging.error(f"Available variables: {available_vars}")
                    logging.error(f"No custom filter available for {model}")
                    raise ValueError(f"Variable '{current_var_list[0]}' not found in the station data.")
            else:
                ds = stn_data[current_var_list[0]]

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
            # Skip resampling for climatology mode - handled by Mod_Climatology
            if len(ds.time) > 0 and not self._is_climatology_mode():
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

            # Store original variable name as attribute for later renaming if needed
            if original_varname:
                ds.attrs['_original_varname'] = original_varname

            return ds  # .where((ds > -1e20) & (ds < 1e20), np.nan)
        finally:
            # Restore the canonical variable definition so that fallback
            # adjustments do not leak into subsequent stations
            if datasource == 'ref':
                self.ref_varname = list(original_var_list) if var_attr_is_list else original_varname
            else:
                self.sim_varname = list(original_var_list) if var_attr_is_list else original_varname

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
                if processed_data is None:
                    logging.info(
                        f"Skipping station {station['ID']} ({datasource}) - no valid data after processing"
                    )
                    return
                self.save_station_data(processed_data, station, datasource)
        finally:
            gc.collect()

    def save_station_data(self, data: xr.Dataset, station: pd.Series, datasource: str) -> None:
        try:
            station = Convert_Type.convert_Frame(station)
            output_file = os.path.join(self.casedir, 'data',
                                       f'stn_{self.ref_source}_{self.sim_source}',
                                       f'{self.item}_{datasource}_{station["ID"]}_{station["use_syear"]}_{station["use_eyear"]}.nc')

            # Rename variable back to original name if needed (for variable fallback scenarios)
            if '_original_varname' in data.attrs:
                original_varname = data.attrs['_original_varname']
                current_varname = data.name if hasattr(data, 'name') and data.name else None

                # Always rename if original_varname is different from current name
                if current_varname != original_varname:
                    logging.info(f"Renaming variable '{current_varname}' back to '{original_varname}' before saving (station {station['ID']})")
                    # Convert DataArray to Dataset with the original variable name
                    if current_varname:
                        data_to_save = data.to_dataset(name=original_varname)
                    else:
                        # If no current name, convert to dataset and rename the variable
                        data_to_save = data.to_dataset()
                        # Get the first (and should be only) data variable name
                        current_var_list = list(data_to_save.data_vars)
                        if current_var_list:
                            data_to_save = data_to_save.rename({current_var_list[0]: original_varname})
                    # Remove the temporary attribute
                    if '_original_varname' in data_to_save.attrs:
                        del data_to_save.attrs['_original_varname']
                    data_to_save.to_netcdf(output_file)
                else:
                    # Remove the temporary attribute
                    if '_original_varname' in data.attrs:
                        del data.attrs['_original_varname']
                    data.to_netcdf(output_file)
            else:
                data.to_netcdf(output_file)

            logging.debug(f"Saved station data to {output_file}")
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

            # Only save if processed_data is not None (i.e., had valid time range)
            if processed_data is not None:
                self.save_extracted_data(processed_data, station, datasource)
            else:
                logging.info(f"Skipping station {station['ID']} - no data in time range {start_year}-{end_year}")
        finally:
            gc.collect()

    @performance_monitor
    def extract_single_station_data(self, dataset: xr.Dataset, station: pd.Series, datasource: str) -> xr.Dataset:
        if dataset is None:
            logging.error(f"Dataset is None for station {station['ID']} ({datasource})")
            raise ValueError("Dataset cannot be None when extracting station data")

        if datasource == 'ref':
            lat_key, lon_key = 'sim_lat', 'sim_lon'
        elif datasource == 'sim':
            lat_key, lon_key = 'ref_lat', 'ref_lon'
        else:
            logging.error(f"Invalid datasource: {datasource}")
            raise ValueError(f"Invalid datasource: {datasource}")

        # Fallback to reference coordinates if simulation coordinates are unavailable
        if lat_key not in station or pd.isna(station.get(lat_key)):
            lat_key = 'ref_lat'
        if lon_key not in station or pd.isna(station.get(lon_key)):
            lon_key = 'ref_lon'

        target_lat = float(station[lat_key])
        target_lon = float(station[lon_key])

        lon_coord = 'lon' if 'lon' in dataset.coords else 'x'
        lat_coord = 'lat' if 'lat' in dataset.coords else 'y'

        try:
            return dataset.sel({
                lat_coord: [target_lat],
                lon_coord: [target_lon]
            }, method="nearest")
        except (KeyError, ValueError, pd.errors.InvalidIndexError) as exc:
            logging.debug(
                "Coordinate selection failed for station %s (%s): %s. Falling back to manual indexing.",
                station['ID'], datasource, exc
            )

            lat_values = dataset[lat_coord].values
            lon_values = dataset[lon_coord].values

            lat_idx = int(np.argmin(np.abs(lat_values - target_lat)))
            lon_idx = int(np.argmin(np.abs(lon_values - target_lon)))

            data = dataset.isel({lat_coord: lat_idx, lon_coord: lon_idx})
            data = data.expand_dims({
                lat_coord: [lat_values[lat_idx]],
                lon_coord: [lon_values[lon_idx]]
            })
            return data

    @performance_monitor
    def process_extracted_data(self, data: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
        data = data.sel(time=slice(f'{start_year}-01-01T00:00:00', f'{end_year}-12-31T23:59:59'))

        # Check if time dimension is empty after slicing
        if len(data.time) == 0:
            logging.warning(f"No data available in time range {start_year}-{end_year}. Skipping this station.")
            return None

        data = data  # .where((data > -1e20) & (data < 1e20), np.nan)
        # Skip resampling for climatology mode - handled by Mod_Climatology
        if self._is_climatology_mode():
            return data
        return data.resample(time=self.compare_tim_res).mean()

    def save_extracted_data(self, data: xr.Dataset, station: pd.Series, datasource: str) -> None:
        try:
            output_file = os.path.join(self.casedir, 'data',
                                       f'stn_{self.ref_source}_{self.sim_source}',
                                       f'{self.item}_{datasource}_{station["ID"]}_{station["use_syear"]}_{station["use_eyear"]}.nc')

            data.to_netcdf(output_file)
            logging.debug(f"Saved extracted station data to {output_file}")
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
        logging.debug(f"Combining data to yearly files...")
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
            # Force refresh since files were just created by parallel processing
            var_files = cached_glob(os.path.join(self.casedir, 'scratch', f'{data_source}_{data_params["varname"][0]}_remap_*.nc'), force_refresh=True)
        else:
            var_files = cached_glob(os.path.join(data_dir, f'{data_source}_{data_params["prefix"]}*{data_params["suffix"]}.nc'))

        self.combine_and_save_data(var_files, data_params)

    @performance_monitor
    def combine_and_save_data(self, var_files: List[str], data_params: Dict[str, Any]) -> None:
        with xr.open_mfdataset(var_files, combine='by_coords') as ds:
            ds = ds.sortby('time')
            output_file = self.get_output_filename(data_params)
            # Try to use ProgressBar, but fall back to silent mode if it fails (e.g., non-interactive environment)
            try:
                with ProgressBar():
                    ds.to_netcdf(output_file)
            except (OSError, IOError, BrokenPipeError):
                # ProgressBar failed (likely non-interactive environment), save without progress bar
                ds.to_netcdf(output_file)
            gc.collect()  # Add garbage collection after saving combined data

        # Only cleanup temp files if we created them (i.e., when processing grid data)
        if self.ref_data_type != 'stn' and self.sim_data_type != 'stn':
            self.cleanup_temp_files(data_params)

    def get_output_filename(self, data_params: Dict[str, Any]) -> str:
        if data_params['datasource'] == 'ref':
            return os.path.join(self.casedir, 'data', f'{self.item}_{data_params["datasource"]}_{self.ref_source}_{data_params["varname"][0]}.nc')
        else:
            return os.path.join(self.casedir, 'data', f'{self.item}_{data_params["datasource"]}_{self.sim_source}_{data_params["varname"][0]}.nc')

    def cleanup_temp_files(self, data_params: Dict[str, Any]) -> None:
        """Clean up temporary files, silently skipping non-existent files."""
        failed_removals = []
        for year in range(self.minyear, self.maxyear + 1):
            temp_file = os.path.join(self.casedir, 'scratch', f'{data_params["datasource"]}_{data_params["varname"][0]}_remap_{year}.nc')
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
            logging.debug(f"Extracting station data for {data_params['datasource']} data")
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
                logging.debug(f"Processing {var_file} for year {year}")
                logging.debug(f"Processing {data_source} data for year {year}")

            with xr.open_dataset(var_file) as data:
                data = Convert_Type.convert_nc(data)
                data = self.preprocess_grid_data(data)
                remapped_data = self.remap_data(data)
                self.save_remapped_data(remapped_data, data_source, year)
        finally:
            gc.collect()

    @performance_monitor
    # NOTE: @cached removed - cache key collisions caused race conditions
    def preprocess_grid_data(self, data: xr.Dataset) -> xr.Dataset:
        # Check if lon and lat are 2D
        data = self.check_coordinate(data)
        if data['lon'].ndim == 2 and data['lat'].ndim == 2:
            try:
                from regrid.regrid_wgs84 import convert_to_wgs84_xesmf
                data = convert_to_wgs84_xesmf(data, self.compare_grid_res)
            except (ImportError, ValueError, RuntimeError) as e:
                logging.debug(f"xesmf regridding failed, falling back to scipy: {e}")
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
            self.remap_interpolate,  # Conservative regrid - last resort
            self.remap_cdo,          # CDO - most stable, climate science standard
            self.remap_xesmf,        # xESMF - fallback option
        ]

        # Collect errors but don't report until all methods fail
        errors = []

        for method in remapping_methods:
            try:
                return method(data, new_grid)
            except Exception as e:
                errors.append(f"{method.__name__}: {e}")
                continue

        # If all remapping methods fail, try basic interpolation as fallback
        try:
            return self.remap_basic_interpolation(data, new_grid)
        except Exception as e:
            errors.append(f"remap_basic_interpolation: {e}")

        # All methods failed - now report all errors
        logging.error("All remapping methods failed:")
        for error in errors:
            logging.error(f"  - {error}")
        logging.warning("Returning original data without remapping")
        return data

    @performance_monitor
    @cached(key_prefix="create_target_grid", ttl=7200)  # Cache for 2 hours (grid rarely changes)
    def create_target_grid(self) -> xr.Dataset:
        lon_new = np.arange(self.min_lon + self.compare_grid_res / 2, self.max_lon, self.compare_grid_res)
        lat_new = np.arange(self.min_lat + self.compare_grid_res / 2, self.max_lat, self.compare_grid_res)
        return xr.Dataset({'lon': lon_new, 'lat': lat_new})

    @performance_monitor(silent_on_error=True)
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

    @performance_monitor(silent_on_error=True)
    def remap_xesmf(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        import xesmf as xe
        regridder = xe.Regridder(data, new_grid, 'conservative')
        return regridder(data)

    @performance_monitor(silent_on_error=True)
    def remap_cdo(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        import subprocess
        import tempfile
        import os

        # Prepare data - ensure proper coordinate attributes
        data_prepared = data.copy()

        # Add CF-compliant coordinate attributes if missing
        if 'lon' in data_prepared.coords:
            if 'standard_name' not in data_prepared['lon'].attrs:
                data_prepared['lon'].attrs['standard_name'] = 'longitude'
            if 'units' not in data_prepared['lon'].attrs:
                data_prepared['lon'].attrs['units'] = 'degrees_east'

        if 'lat' in data_prepared.coords:
            if 'standard_name' not in data_prepared['lat'].attrs:
                data_prepared['lat'].attrs['standard_name'] = 'latitude'
            if 'units' not in data_prepared['lat'].attrs:
                data_prepared['lat'].attrs['units'] = 'degrees_north'

        temp_input_name = None
        temp_output_name = None
        temp_grid_name = None

        try:
            # Create temporary files
            temp_input = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
            temp_input_name = temp_input.name
            temp_input.close()

            temp_output = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
            temp_output_name = temp_output.name
            temp_output.close()

            temp_grid = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
            temp_grid_name = temp_grid.name
            temp_grid.close()

            # Save data to NetCDF with NETCDF4_CLASSIC format for CDO compatibility
            data_prepared.to_netcdf(temp_input_name, format='NETCDF4_CLASSIC')

            # Create target grid file
            self.create_target_grid_file(temp_grid_name, new_grid)

            # Use remapcon (conservative remapping) - CDO's standard conservative method
            cmd = f"cdo -s remapcon,{temp_grid_name} {temp_input_name} {temp_output_name}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

            # Read result and load into memory before cleanup
            with xr.open_dataset(temp_output_name) as result_ds:
                result_data = result_ds.load()
            return Convert_Type.convert_nc(result_data)

        finally:
            # Clean up temporary files
            for f in [temp_input_name, temp_output_name, temp_grid_name]:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except (OSError, PermissionError) as e:
                        logging.debug(f"Could not delete temporary file {f}: {e}")

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
            # Skip resampling for climatology mode - handled by Mod_Climatology
            if not self._is_climatology_mode():
                data = data.resample(time=self.compare_tim_res).mean()
            data = data.sel(time=slice(f'{year}-01-01T00:00:00', f'{year}-12-31T23:59:59'))

            varname = self.ref_varname[0] if data_source == 'ref' else self.sim_varname[0]

            out_file = os.path.join(self.casedir, 'scratch', f'{data_source}_{varname}_remap_{year}.nc')
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
