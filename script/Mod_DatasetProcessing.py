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
from typing import List, Dict, Any, Tuple, Callable

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed

from Lib_Unit import UnitProcessing
from Mod_Converttype import Convert_Type


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger("xarray").setLevel(logging.WARNING)

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance including execution time and memory usage."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
        start_time = time.time()
        
        # Get initial CPU usage
        start_cpu = process.cpu_percent()
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Calculate execution time and memory usage
            end_time = time.time()
            end_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
            end_cpu = process.cpu_percent()
            
            execution_time = end_time - start_time
            memory_used = end_mem - start_mem
            cpu_used = end_cpu - start_cpu
            
            # Log performance 
            logging.info(f"Performance  for {func.__name__}:")
            logging.info(f"  Execution time: {execution_time:.2f} seconds")
            logging.info(f"  Memory usage: {memory_used:.3f} GB")
            logging.info(f"  CPU usage: {cpu_used:.1f}%")
            
            # Log warning if memory usage is high
            if memory_used > 0.8 * psutil.virtual_memory().total / (1024**3):  # 80% of total memory
                logging.warning(f"High memory usage detected in {func.__name__}: {memory_used:.3f} GB")
            
            return result
            
        except Exception as e:
            # Log error with performance context
            end_time = time.time()
            end_mem = process.memory_info().rss / 1024 / 1024 / 1024
            execution_time = end_time - start_time
            memory_used = end_mem - start_mem
            
            logging.error(f"Error in {func.__name__} after {execution_time:.2f}s and using {memory_used:.3f} GB:")
            logging.error(str(e))
            raise
            
    return wrapper

def get_system_resources():
    """
    Get system resources information.
    
    Returns:
        dict: Dictionary containing system resource information
    """
    try:
        # Get total memory in GB
        total_memory = psutil.virtual_memory().total / (1024**3)
        # Get available memory in GB
        available_memory = psutil.virtual_memory().available / (1024**3)
        # Get number of CPU cores
        cpu_count = psutil.cpu_count(logical=False)
        # Get CPU frequency
        cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 0
        
        return {
            'total_memory_gb': total_memory,
            'available_memory_gb': available_memory,
            'cpu_count': cpu_count,
            'cpu_freq_mhz': cpu_freq
        }
    except Exception as e:
        logging.warning(f"Failed to get system resources: {e}")
        return {
            'total_memory_gb': 8,  # Default values
            'available_memory_gb': 4,
            'cpu_count': 4,
            'cpu_freq_mhz': 0
        }

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

class BaseDatasetProcessing:
    def __init__(self, config: Dict[str, Any]):
        self.initialize_attributes(config)
        self.setup_output_directories()
        self.initialize_resource_parameters()

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
    def process(self, datasource: str) -> None:
        logging.info(f"Processing {datasource} data")
        self._preprocess(datasource)
        logging.info(f"{datasource.capitalize()} data prepared!")

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
    def check_dataset_time_integrity(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str, datasource: str) -> xr.Dataset:
        """Checks and fills missing time values in an xarray Dataset with specified comparison scales."""
        # Ensure the dataset has a proper time index
        ds = self.check_time(ds, syear, eyear, tim_res)
        # Apply model-specific time adjustments
        if datasource == 'stat':
            pass
        else:
            if self.sim_data_type != 'stn':
                ds = self.apply_model_specific_time_adjustment(ds, datasource, syear, eyear, tim_res)
        ds = self.make_time_integrity(ds, syear, eyear, tim_res, datasource)
        return ds

    @performance_monitor
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
            time_values = time_var
            # Create a complete time series based on the specified time frequency and range 
            # Compare the actual time with the complete time series to find the missing time
            # print('Checking time series completeness...')
            missing_times = time_index[~np.isin(time_index, time_values)]
            if len(missing_times) > 0:
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
        model = self.sim_model if datasource == 'sim' else self.ref_source
        try:
            custom_module = importlib.import_module(f"custom.{model}_filter")
            custom_time_adjustment = getattr(custom_module, f"adjust_time_{model}")
            ds = custom_time_adjustment(self, ds, syear, eyear, tim_res)
        except (ImportError, AttributeError):
            logging.warning(f"No custom time adjustment found for {model}. Using original time values.")
            pass
        return ds

    @performance_monitor
    def select_var(self, syear: int, eyear: int, tim_res: str, VarFile: str, varname: List[str], datasource: str) -> xr.Dataset:
        ds = None
        try:
            # Get file size in GB
            file_size_gb = os.path.getsize(VarFile) / (1024**3)
            
            # Get optimal chunk size (auto)
            chunks = self.get_optimal_chunks(file_size_gb)
            
            # Update number of cores based on file size
            self.num_cores = self.get_optimal_cores(file_size_gb)
            
            logging.info(f"Using auto chunking and {self.num_cores} cores for processing {VarFile}")
            
            try:
                ds = xr.open_dataset(VarFile, chunks=chunks)
            except:
                ds = xr.open_dataset(VarFile, decode_times=False, chunks=chunks)
            
            # Apply filters and conversions
            ds = self.apply_custom_filter(datasource, ds)
            ds = Convert_Type.convert_nc(ds)
            
            # Clear memory before returning
            gc.collect()
            return ds
        
        except Exception as e:
            logging.error(f"Failed to open dataset: {VarFile}")
            logging.error(f"Error: {str(e)}")
            raise
        finally:
            # Ensure dataset is properly closed
            if ds is not None and hasattr(ds, 'close'):
                ds.close()
            gc.collect()

    @performance_monitor
    def apply_custom_filter(self, datasource: str, ds: xr.Dataset) -> xr.Dataset:
        if datasource == 'stat':
            return ds
        else:
            model = self.sim_model if datasource == 'sim' else self.ref_source
            try:
                logging.info(f"Attempting to load custom filter for {model}")
                custom_module = importlib.import_module(f"custom.{model}_filter")
                custom_filter = getattr(custom_module, f"filter_{model}")
                self, ds = custom_filter(self, ds)
            except ImportError:
                logging.info(f"No custom filter found for {model}, using original dataset")
                return ds
            except AttributeError:
                logging.info(f"No filter_{model} function found in {model}_filter module, using original dataset")
                return ds
            except Exception as e:
                logging.warning(f"Error applying custom filter for {model}: {str(e)}")
                return ds
            return ds

    @performance_monitor
    def select_timerange(self, ds: xr.Dataset, syear: int, eyear: int) -> xr.Dataset:
        if (eyear < syear) or (ds.sel(time=slice(f'{syear}-01-01T00:00:00', f'{eyear}-12-31T23:59:59')) is None):
            logging.error(f"Error: Attempting checking the data time range.")
            exit()
        else:
            return ds.sel(time=slice(f'{syear}-01-01T00:00:00', f'{eyear}-12-31T23:59:59'))

    @performance_monitor
    def resample_data(self, dfx1: xr.Dataset, tim_res: str, startx: int, endx: int) -> xr.Dataset:
        match = re.match(r'(\d+)\s*([a-zA-Z]+)', tim_res)
        if not match:
            logging.error("Invalid time resolution format. Use '3month', '6hr', etc.")
            raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")

        value, unit = match.groups()
        value = int(value)
        freq = self.freq_map.get(unit.lower())
        if not freq:
            logging.error(f"Unsupported time unit: {unit}")
            raise ValueError(f"Unsupported time unit: {unit}")

        time_index = pd.date_range(start=f'{startx}-01-01T00:00:00', end=f'{endx}-12-31T59:59:59', freq=f'{value}{freq}')
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
            dataset_size_gb = ds.nbytes / (1024**3)
            
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
        for file in var_files:
            ds = self.select_var(year, year, tim_res, file, varname, datasource)
            datasets.append(ds)
        data0 = xr.concat(datasets, dim="time").sortby('time')
        return data0

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
        logging.info("Processing station data")
        Parallel(n_jobs=-1)(
            delayed(self._make_stn_parallel)(
                self.station_list, data_params['datasource'], i
            ) for i in range(len(self.station_list['ID']))
        )

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

        # if not UnitProcessing.check_units(self.ref_varunit, self.sim_varunit):
        #    ds, _ = UnitProcessing.process_unit(self, ds, getattr(self, f"{datasource}_varunit"))

        ds = self.select_timerange(ds, start_year, end_year)
        return ds  # .where((ds > -1e20) & (ds < 1e20), np.nan)

    @performance_monitor
    def _make_stn_parallel(self, station_list: pd.DataFrame, datasource: str, index: int) -> None:
        station = station_list.iloc[index]
        start_year = int(station['use_syear'])
        end_year = int(station['use_eyear'])
        file_path = station["sim_dir"] if datasource == 'sim' else station["ref_dir"]
        with xr.open_dataset(file_path) as stn_data:
            stn_data = Convert_Type.convert_nc(stn_data)
            processed_data = self.process_single_station_data(stn_data, start_year, end_year, datasource)
            self.save_station_data(processed_data, station, datasource)

    def save_station_data(self, data: xr.Dataset, station: pd.Series, datasource: str) -> None:
        station = Convert_Type.convert_Frame(station)
        output_file = os.path.join(self.casedir, 'output', 'data', f'stn_{self.ref_source}_{self.sim_source}', f'{self.item}_{datasource}_{station["ID"]}_{station["use_syear"]}_{station["use_eyear"]}.nc')
        data.to_netcdf(output_file)
        logging.info(f"Saved station data to {output_file}")

    @performance_monitor
    def _extract_stn_parallel(self, datasource: str, dataset: xr.Dataset, station_list: pd.DataFrame, index: int) -> None:
        station = station_list.iloc[index]
        start_year = int(station['use_syear'])
        end_year = int(station['use_eyear'])

        station_data = self.extract_single_station_data(dataset, station, datasource)
        processed_data = self.process_extracted_data(station_data, start_year, end_year)
        self.save_extracted_data(processed_data, station, datasource)
        gc.collect()  # Add garbage collection after processing each station

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
        output_file = os.path.join(self.casedir, 'output', 'data', f'stn_{self.ref_source}_{self.sim_source}', f'{self.item}_{datasource}_{station["ID"]}_{station["use_syear"]}_{station["use_eyear"]}.nc')
        data.to_netcdf(output_file)
        logging.info(f"Saved extracted station data to {output_file}")


class GridDatasetProcessing(BaseDatasetProcessing):
    @performance_monitor
    def process_grid_data(self, data_params: Dict[str, Any]) -> None:
        self.prepare_grid_data(data_params)
        self.remap_and_combine_data(data_params)
        self.extract_station_data_if_needed(data_params)

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

        self.cleanup_temp_files(data_params)


    def get_output_filename(self, data_params: Dict[str, Any]) -> str:
        if data_params['datasource'] == 'ref':
            return os.path.join(self.casedir, 'output', 'data', f'{self.item}_{data_params["datasource"]}_{self.ref_source}_{data_params["varname"][0]}.nc')
        else:
            return os.path.join(self.casedir, 'output', 'data', f'{self.item}_{data_params["datasource"]}_{self.sim_source}_{data_params["varname"][0]}.nc')

    def cleanup_temp_files(self, data_params: Dict[str, Any]) -> None:
        try:
            for year in range(self.minyear, self.maxyear + 1):
                os.remove(os.path.join(self.casedir, 'tmp', f'{data_params["datasource"]}_{data_params["varname"][0]}_remap_{year}.nc'))
        except OSError:
            logging.warning("Failed to remove some temporary files.")

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
            gc.collect()  # Add garbage collection after processing each grid

    @performance_monitor
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

        # raise RuntimeError("All remapping methods failed")

    @performance_monitor
    def create_target_grid(self) -> xr.Dataset:
        lon_new = np.arange(self.min_lon + self.compare_grid_res / 2, self.max_lon, self.compare_grid_res)
        lat_new = np.arange(self.min_lat + self.compare_grid_res / 2, self.max_lat, self.compare_grid_res)
        return xr.Dataset({'lon': lon_new, 'lat': lat_new})

    @performance_monitor
    def remap_interpolate(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.Dataset:
        from regrid import Grid
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
        # Convert sparse arrays to dense arrays
        data = data.resample(time=self.compare_tim_res).mean()
        data = data.sel(time=slice(f'{year}-01-01T00:00:00', f'{year}-12-31T23:59:59'))

        varname = self.ref_varname[0] if data_source == 'ref' else self.sim_varname[0]

        out_file = os.path.join(self.casedir, 'tmp', f'{data_source}_{varname}_remap_{year}.nc')
        data.to_netcdf(out_file)
        logging.info(f"Saved remapped {data_source} data for year {year} to {out_file}")


class DatasetProcessing(StationDatasetProcessing, GridDatasetProcessing):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def process(self, datasource: str) -> None:
        super().process(datasource)
        # Add any additional processing specific to this class if needed
