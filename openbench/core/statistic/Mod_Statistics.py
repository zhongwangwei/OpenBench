# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from scipy import stats
from joblib import Parallel, delayed
import dask.array as da
import os
import glob
import importlib
import logging
from typing import List, Dict, Any, Tuple

import pandas as pd
import re
import gc
from joblib import Parallel, delayed
import warnings
from dask.diagnostics import ProgressBar
import shutil
from . import *
from ...data.Mod_DatasetProcessing import BaseDatasetProcessing
from openbench.util.Mod_Converttype import Convert_Type
import time

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class BasicProcessing(statistics_calculate, BaseDatasetProcessing):
    def __init__(self, info):
        """
        Initialize the Statistics class.

        Args:
            info (dict): A dictionary containing additional information to be added as attributes.
        """
        self.name = 'statistics'
        self.version = '0.2'
        self.release = '0.2'
        self.date = 'Mar 2024'
        self.author = "Zhongwang Wei"
        self.__dict__.update(info)
        statistics_calculate.__init__(self, info)

    def check_time_freq(self, time_freq: str) -> xr.Dataset:
        if not [time_freq][0].isdigit():
            time_freq = f'1{time_freq}'
        match = re.match(r'(\d+)\s*([a-zA-Z]+)', time_freq)
        if match:
            num_value, unit = match.groups()
            num_value = int(num_value) if num_value else 1
            unit = self.freq_map.get(unit.lower())
            time_freq = f'{num_value}{unit}'
        else:
            raise ValueError(f"Invalid time resolution format: {time_freq}. Use '3month', '6hr', etc.")
        return time_freq

    def process_data_source(self, source: str, config: Dict[str, Any]) -> xr.Dataset:
        source_config = {k: v for k, v in config.items() if k.startswith(source)}
        dirx = source_config[f'{source}_dir']
        syear = int(source_config[f'{source}_syear'])
        eyear = int(source_config[f'{source}_eyear'])
        time_freq = source_config[f'{source}_tim_res']
        time_freq = self.check_time_freq(time_freq)
        varname = source_config[f'{source}_varname']
        varunit = source_config[f'{source}_varunit']
        groupby = source_config[f'{source}_data_groupby'].lower()
        suffix = source_config[f'{source}_suffix']
        prefix = source_config[f'{source}_prefix']
        logging.info(f"Processing data source '{source}' from '{dirx}'...")

        if groupby == 'single':
            ds = self.process_single_groupby(dirx, suffix, prefix, varname, varunit, syear, eyear, time_freq)
        elif groupby == 'year':
            years = range(syear, eyear + 1)
            ds_list = Parallel(n_jobs=self.num_cores)(
                delayed(self.process_yearly_groupby)(dirx, suffix, prefix, varname, varunit, year, year, time_freq)
                for year in years
            )
            ds = xr.concat(ds_list, dim='time')
        else:
            logging.info(f"Combining data to one file...")
            years = range(syear, eyear + 1)
            ds_list = Parallel(n_jobs=self.num_cores)(
                delayed(self.process_other_groupby)(dirx, suffix, prefix, varname, varunit, year, year, time_freq)
                for year in years
            )
            ds = xr.concat(ds_list, dim='time')
        ds = Convert_Type.convert_nc(ds)
        return ds

    def process_single_groupby(self, dirx: str, suffix: str, prefix: str, varname: List[str], varunit: List[str], syear: int, eyear: int,
                               time_freq: str) -> xr.Dataset:
        VarFile = self.check_file_exist(os.path.join(dirx, f'{prefix}{suffix}.nc'))
        if isinstance(varname, str): varname = [varname]
        ds = self.select_var(syear, eyear, time_freq, VarFile, varname, 'stat')
        ds = self.load_and_process_dataset(ds, syear, eyear, time_freq, varunit)
        return ds

    def process_yearly_groupby(self, dirx: str, suffix: str, prefix, varname: List[str], varunit: List[str], syear: int, eyear: int,
                               time_freq: str) -> xr.Dataset:
        VarFile = self.check_file_exist(os.path.join(dirx, f'{prefix}{syear}{suffix}.nc'))
        if isinstance(varname, str): varname = [varname]
        ds = self.select_var(syear, eyear, time_freq, VarFile, varname, 'stat')
        ds = self.load_and_process_dataset(ds, syear, eyear, time_freq, varunit)
        return ds

    def process_other_groupby(self, dirx: str, suffix: str, prefix: str, varname: List[str], varunit: List[str], syear: int, eyear: int,
                              time_freq: str) -> xr.Dataset:
        if isinstance(varname, str): varname = [varname]
        ds = self.combine_year(syear, dirx, dirx, suffix, prefix, varname, 'stat', time_freq)
        ds = self.load_and_process_dataset(ds, syear, eyear, time_freq, varunit)
        return ds

    def load_and_process_dataset(self, ds: xr.Dataset, syear: str, eyear: str, time_freq, varunit) -> xr.Dataset:
        ds = self.check_coordinate(ds)
        ds = self.check_dataset_time_integrity(ds, syear, eyear, time_freq, 'stat')
        ds = self.select_timerange(ds, syear, eyear)
        ds, varunit = self.process_units(ds, varunit)
        return ds

    def remap_data(self, data_list):
        """
        Remap all datasets to the resolution specified in main.nml.
        Tries CDO first, then xESMF, and finally xarray's interp method.
        """

        remapping_methods = [
            self.remap_xesmf,
            self.remap_cdo,
            self.remap_interpolate
        ]

        remapped_data = []
        for i, data in enumerate(data_list):
            data = self.preprocess_grid_data(data)
            data.to_netcdf('test.nc')
            new_grid = self.create_target_grid()
            for method in remapping_methods:
                try:
                    remapped = method(data, new_grid)
                except Exception as e:
                    logging.warning(f"{method.__name__} failed: {e}")

            # Skip resampling for climatology mode
            compare_tim_res_lower = str(self.compare_tim_res).strip().lower()
            if compare_tim_res_lower not in ['climatology-year', 'climatology-month']:
                remapped = remapped.resample(time=self.compare_tim_res).mean()
            remapped_data.append(remapped)
        return remapped_data

    def preprocess_grid_data(self, data: xr.Dataset) -> xr.Dataset:
        # Check if lon and lat are 2D
        data = self.check_coordinate(data)
        if data['lon'].ndim == 2 and data['lat'].ndim == 2:
            try:
                from openbench.data.regrid.regrid_wgs84 import convert_to_wgs84_xesmf
                data = convert_to_wgs84_xesmf(data, self.compare_grid_res)
            except:
                from openbench.data.regrid.regrid_wgs84 import convert_to_wgs84_scipy
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

    def create_target_grid(self) -> xr.Dataset:
        min_lon = self.main_nml['general']['min_lon']
        min_lat = self.main_nml['general']['min_lat']
        max_lon = self.main_nml['general']['max_lon']
        max_lat = self.main_nml['general']['max_lat']
        lon_new = np.arange(min_lon + self.compare_grid_res / 2, max_lon, self.compare_grid_res)
        lat_new = np.arange(min_lat + self.compare_grid_res / 2, max_lat, self.compare_grid_res)
        return xr.Dataset({'lon': lon_new, 'lat': lat_new})

    def remap_interpolate(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.DataArray:
        from openbench.data.regrid import Grid, Regridder
        min_lon = self.main_nml['general']['min_lon']
        min_lat = self.main_nml['general']['min_lat']
        max_lon = self.main_nml['general']['max_lon']
        max_lat = self.main_nml['general']['max_lat']

        grid = Grid(
            north=max_lat - self.compare_grid_res / 2,
            south=min_lat + self.compare_grid_res / 2,
            west=min_lon + self.compare_grid_res / 2,
            east=max_lon - self.compare_grid_res / 2,
            resolution_lat=self.compare_grid_res,
            resolution_lon=self.compare_grid_res,
        )
        target_dataset = grid.create_regridding_dataset(lat_name="lat", lon_name="lon")
        # Convert sparse arrays to dense arrays
        data_regrid = data.regrid.conservative(target_dataset, nan_threshold=0)
        return Convert_Type.convert_nc(data_regrid)

    def remap_xesmf(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.DataArray:
        import xesmf as xe
        regridder = xe.Regridder(data, new_grid, 'conservative')
        ds = regridder(data)
        return list(Convert_Type.convert_nc(ds.data_vars).values())[0]

    def remap_cdo(self, data: xr.Dataset, new_grid: xr.Dataset) -> xr.DataArray:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.nc') as temp_input, \
                tempfile.NamedTemporaryFile(suffix='.nc') as temp_output, \
                tempfile.NamedTemporaryFile(suffix='.txt') as temp_grid:
            data.to_netcdf(temp_input.name)
            self.create_target_grid_file(temp_grid.name, new_grid)

            cmd = f"cdo -s remapcon,{temp_grid.name} {temp_input.name} {temp_output.name}"
            subprocess.run(cmd, shell=True, check=True)
            ds = xr.open_dataset(temp_output.name)
            return list(Convert_Type.convert_nc(ds.data_vars).values())[0]

    def create_target_grid_file(self, filename: str, new_grid: xr.Dataset) -> None:
        min_lon = self.main_nml['general']['min_lon']
        min_lat = self.main_nml['general']['min_lat']
        with open(filename, 'w') as f:
            f.write(f"gridtype = lonlat\n")
            f.write(f"xsize = {len(new_grid.lon)}\n")
            f.write(f"ysize = {len(new_grid.lat)}\n")
            f.write(f"xfirst = {min_lon + self.compare_grid_res / 2}\n")
            f.write(f"xinc = {self.compare_grid_res}\n")
            f.write(f"yfirst = {min_lat + self.compare_grid_res / 2}\n")
            f.write(f"yinc = {self.compare_grid_res}\n")

    def save_result(self, method_name: str, result, data_sources: List[str]) -> xr.Dataset:
        # Remove the existing output directory
        filename_parts = [method_name] + data_sources
        filename = "_".join(filename_parts) + "_output.nc"
        output_file = os.path.join(self.output_dir, f"{method_name}", filename)
        logging.info(f"Saving {method_name} output to {output_file}")
        if isinstance(result, xr.DataArray) or isinstance(result, xr.Dataset):
            if isinstance(result, xr.DataArray):
                result = result.to_dataset(name=f"{method_name}")
            else:
                if method_name in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
                    if method_name not in result.data_vars:
                        varname = next(iter(result.data_vars))
                        result = result.rename({varname: method_name})
            result['lat'].attrs['_FillValue'] = float('nan')
            result['lat'].attrs['standard_name'] = 'latitude'
            result['lat'].attrs['long_name'] = 'latitude'
            result['lat'].attrs['units'] = 'degrees_north'
            result['lat'].attrs['axis'] = 'Y'
            result['lon'].attrs['_FillValue'] = float('nan')
            result['lon'].attrs['standard_name'] = 'longitude'
            result['lon'].attrs['long_name'] = 'longitude'
            result['lon'].attrs['units'] = 'degrees_east'
            result['lon'].attrs['axis'] = 'X'
            result.to_netcdf(output_file)
        else:
            # If the result is not xarray object, we might need to handle it differently
            # For now, let's just print it
            logging.info(f"Result of {method_name}: {result}")
        return output_file

    coordinate_map = {
        'longitude': 'lon', 'long': 'lon', 'lon_cama': 'lon', 'lon0': 'lon', 'x': 'lon',
        'latitude': 'lat', 'lat_cama': 'lat', 'lat0': 'lat', 'y': 'lat',
        'Time': 'time', 'TIME': 'time', 't': 'time', 'T': 'time',
        'elevation': 'elev', 'height': 'elev', 'z': 'elev', 'Z': 'elev',
        'h': 'elev', 'H': 'elev', 'ELEV': 'elev', 'HEIGHT': 'elev',
    }
    freq_map = {
        'month': 'ME',
        'mon': 'ME',
        'monthly': 'ME',
        'day': 'D',
        'daily': 'D',
        'hour': 'H',
        'Hour': 'H',
        'hr': 'H',
        'Hr': 'H',
        'h': 'H',
        'hourly': 'H',
        'year': 'Y',
        'yr': 'Y',
        'yearly': 'Y',
        'week': 'W',
        'wk': 'W',
        'weekly': 'W',
    }


class StatisticsProcessing(BasicProcessing):
    def __init__(self, main_nml, stats_nml, output_dir, num_cores=-1):
        super().__init__(main_nml)
        super().__init__(stats_nml)
        self.name = 'StatisticsDataHandler'
        self.version = '0.3'
        self.release = '0.3'
        self.date = 'June 2024'
        self.author = "Zhongwang Wei"

        self.stats_nml = stats_nml
        self.main_nml = main_nml
        self.general_config = self.stats_nml['general']
        self.output_dir = output_dir
        self.num_cores = num_cores

        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml['general']['compare_grid_res']
        self.compare_tim_res = self.main_nml['general'].get('compare_tim_res', '1').lower()

        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ['climatology-year', 'climatology-month']:
            logging.info(f"StatisticsProcessing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion")
        else:
            # this should be done in read_namelist
            # adjust the time frequency
            match = re.match(r'(\d*)\s*([a-zA-Z]+)', self.compare_tim_res)
            if not match:
                logging.error("Invalid time resolution format. Use '3month', '6hr', etc.")
                raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")
            value, unit = match.groups()
            if not value:
                value = 1
            else:
                value = int(value)  # Convert the numerical value to an integer
            # Get the corresponding pandas frequency
            freq = self.freq_map.get(unit.lower())
            if not freq:
                logging.error(f"Unsupported time unit: {unit}")
                raise ValueError(f"Unsupported time unit: {unit}")
            self.compare_tim_res = f'{value}{freq}'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def setup_output_directories(self, statistic_method):
        if os.path.exists(os.path.join(self.output_dir, f"{statistic_method}")):
            shutil.rmtree(os.path.join(self.output_dir, f"{statistic_method}"))
        # Create a new output directory
        if not os.path.exists(os.path.join(self.output_dir, f"{statistic_method}")):
            os.makedirs(os.path.join(self.output_dir, f"{statistic_method}"))

    # Basic statistical methods
    def scenarios_Basic_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        logging.info(f"Processing {statistic_method}")
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [source.strip()]
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Basic_Plot import make_Basic
            make_Basic(output_file, statistic_method, [source], self.main_nml['general'], option)

    def scenarios_Mann_Kendall_Trend_Test_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            option['significance_level'] = statistic_nml['significance_level']
            sources = [source.strip()]
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Mann_Kendall_Trend_Test import make_Mann_Kendall_Trend_Test
            make_Mann_Kendall_Trend_Test(output_file, statistic_method, [source], self.main_nml['general'], option)

    def scenarios_Correlation_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)

        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [f'{source}1', f'{source}2']
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Correlation import make_Correlation
            make_Correlation(output_file, statistic_method, self.main_nml['general'], option)

    def scenarios_Standard_Deviation_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [source.strip()]
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Standard_Deviation import make_Standard_Deviation
            make_Standard_Deviation(output_file, statistic_method, [source], self.main_nml['general'], option)

    def scenarios_Hellinger_Distance_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)

        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [f'{source}1', f'{source}2']
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Hellinger_Distance import make_Hellinger_Distance
            make_Hellinger_Distance(output_file, statistic_method, [source], self.main_nml['general'], option)

    def scenarios_Z_Score_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [source.strip()]
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            # make_Z_Score(output_file, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def scenarios_Three_Cornered_Hat_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            nX = int(statistic_nml[f'{source}_nX'])
            if nX < 3:
                logging.error('Error: Three Cornered Hat method must be at least 3 dataset.')
                exit(1)
            sources = [f'{source}{i}' for i in range(1, nX + 1)]
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Three_Cornered_Hat import make_Three_Cornered_Hat
            make_Three_Cornered_Hat(output_file, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def scenarios_Partial_Least_Squares_Regression_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)

        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            nX = int(statistic_nml[f'{source}_nX'])
            sources = [f'{source}_Y'] + [f'{source}_X{i + 1}' for i in range(nX)]
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Partial_Least_Squares_Regression import make_Partial_Least_Squares_Regression
            make_Partial_Least_Squares_Regression(output_file, statistic_method, [source], self.main_nml['general'],
                                                  statistic_nml, option)

    # Advanced statistical methods
    def scenarios_Functional_Response_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [f'{source}1', f'{source}2']
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_Functional_Response import make_Functional_Response
            make_Functional_Response(output_file, statistic_method, [source], self.main_nml['general'], option)

    def scenarios_False_Discovery_Rate_analysis(self, statistic_method, statistic_nml, option):
        return

    def scenarios_ANOVA_analysis(self, statistic_method, statistic_nml, option):
        self.setup_output_directories(statistic_method)
        # Load data sources for this method
        data_sources_key = f'{statistic_method}_data_source'
        if data_sources_key not in self.general_config:
            logging.warning(f"Warning: No data sources found for '{statistic_method}' in stats.nml [general] section.")
            return

            # Assuming 'statistic_method' is defined and corresponds to one of the keys in the configuration
        data_source_config = self.general_config.get(f'{statistic_method}_data_source', '')

        # Check if the data_source_config is a string; if not, handle it appropriately
        if isinstance(data_source_config, str):
            data_sources = data_source_config.split(',')
        else:
            # Assuming data_source_config is a list or another iterable; adjust as necessary
            data_sources = data_source_config  # If it's already a list, no need to split
        for source in data_sources:
            sources = [f'{source}_Y', f'{source}_X']
            output_file = self.run_analysis(source.strip(), sources, statistic_method)
            from openbench.visualization.Fig_ANOVA import make_ANOVA
            make_ANOVA(output_file, statistic_method, [source], self.main_nml['general'], statistic_nml, option)

    def run_analysis(self, source: str, sources: List[str], statistic_method):
        method_function = getattr(self, f"stat_{statistic_method.lower()}", None)
        if method_function:
            data_list = [self.process_data_source(isource.strip(), self.stats_nml[statistic_method])
                         for isource in sources]

            if statistic_method == 'Partial_Least_Squares_Regression':
                try:
                    Y_vars = self.process_data_source(sources[0].strip(), self.stats_nml[statistic_method])
                except:
                    logging.error("No dependent variable (Y) found. Ensure at least one variable has '_Y_' in its name.")
                    raise ValueError("No dependent variable (Y) found. Ensure at least one variable has '_Y_' in its name.")

            if len(data_list) == 0:
                logging.error(f"No data sources found for '{statistic_method}'.")
                raise ValueError(f"No data sources found for '{statistic_method}'.")
            # Remap data
            data_list = self.remap_data(data_list)

            # Call the method with the loaded data
            result = method_function(*data_list)
            output_file = self.save_result(statistic_method, result, [source])
            return output_file
        else:
            logging.warning(f"Warning: Analysis method '{statistic_method}' not implemented.")

def wait_for_file(file_path, max_wait_time=30, check_interval=1):
    """
    Wait for a file to exist and be readable.
    
    Args:
        file_path: Path to the file
        max_wait_time: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
    
    Returns:
        bool: True if file exists and is readable, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if os.path.exists(file_path):
            try:
                # Try to get file size to ensure it's complete
                size = os.path.getsize(file_path)
                if size > 0:
                    logging.info(f"File found and ready: {file_path} ({size} bytes)")
                    return True
            except (OSError, IOError):
                pass
        
        logging.debug(f"Waiting for file: {file_path}")
        time.sleep(check_interval)
    
    logging.error(f"Timeout waiting for file: {file_path}")
    return False
