# -*- coding: utf-8 -*-
import os, glob
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import sys
from dask.diagnostics import ProgressBar
import warnings
import pandas as pd
import re


class timelib:
    def __init__(self):
        self.name = 'DatasetPreprocessing'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.freq_map = {
            'month': 'ME',
            'mon': 'ME',
            'monthly': 'ME',
            'day': 'D',
            'daily': 'D',
            'hour': 'H',
            'hr': 'H',
            'hourly': 'H',
            'year': 'Y',
            'yr': 'Y',
            'yearly': 'Y',
            'week': 'W',
            'wk': 'W',
            'weekly': 'W',
            } 

    def check_time(self, ds: xr.Dataset, syear: int, eyear: int, tim_res: str) -> xr.Dataset:
        print("Checking time coordinate...")
        if 'time' not in ds.coords:
            print("The dataset does not contain a 'time' coordinate.")
            # Based on the syear and eyear, create a time index
            time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
            ds = ds.expand_dims('time')  # Ensure 'time' dimension exists
            ds = ds.assign_coords(time=time_index)  # Assign the created time index to the dataset
        elif not np.issubdtype(ds.time.dtype, np.datetime64):
            try:
                ds['time'] = pd.to_datetime(ds.time.values)
            except:
                # Delete the time coordinate
                ds = ds.drop('time')
                # Delete the time dimension
                ds = ds.squeeze('time')
                time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
                ds = ds.expand_dims('time')  # Ensure 'time' dimension exists
                ds = ds.assign_coords(time=time_index)  # Assign the created time index to the dataset
        else:
            # Check for duplicate time values and remove them
            if ds['time'].to_index().duplicated().any():
                print("Duplicate time values found. Removing duplicates.")
                _, unique_indices = np.unique(ds['time'], return_index=True)
                ds = ds.isel(time=unique_indices)

            # Replace the existing time index with a new one based on syear, eyear, and tim_res
            time_index = pd.date_range(start=f'{syear}-01-01T00:00:00', end=f'{eyear}-12-31T23:59:59', freq=tim_res)
            ds = ds.reindex(time=time_index)  # Reindex the dataset with the created time index

        return ds
    