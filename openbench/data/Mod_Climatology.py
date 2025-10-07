# -*- coding: utf-8 -*-
"""
Climatology processing module for OpenBench
Handles climatological mean evaluations with different time dimensions
"""

import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Tuple, Optional, List


class ClimatologyProcessor:
    """
    Process data for climatological evaluations.

    Supports:
    - Annual climatology (single time point or no time dimension)
    - Monthly climatology (12 time points)
    """

    def __init__(self):
        """Initialize the climatology processor."""
        self.name = 'ClimatologyProcessor'
        self.version = '1.0'
        self.author = 'Zhongwang Wei / zhongwang007@gmail.com'

        # Climatology types
        self.ANNUAL_CLIMATOLOGY = 'annual'
        self.MONTHLY_CLIMATOLOGY = 'monthly'

        # Standard climatology time stamps
        self.ANNUAL_TIME = pd.Timestamp('2000-06-15')
        self.MONTHLY_TIMES = pd.date_range('2000-01-15', periods=12, freq='MS') + pd.Timedelta(days=14)

        # Metrics that don't support climatology evaluation
        self.UNSUPPORTED_METRICS = [
            'nPhaseScore',  # Phase score requires temporal cycle
            'nIavScore',    # Interannual variability score requires multiple years
        ]

    def detect_climatology_type(self, ds: xr.Dataset) -> Optional[str]:
        """
        Detect the climatology type of a dataset based on its time dimension.

        Args:
            ds: Input dataset

        Returns:
            str: Climatology type ('annual', 'monthly', or None)
        """
        if 'time' not in ds.dims and 'time' not in ds.coords:
            logging.info("No time dimension found - treating as annual climatology")
            return self.ANNUAL_CLIMATOLOGY

        time_size = len(ds.time) if 'time' in ds.dims else len(ds.time.values)

        if time_size == 1:
            logging.info("Single time point found - treating as annual climatology")
            return self.ANNUAL_CLIMATOLOGY
        elif time_size == 12:
            logging.info("12 time points found - treating as monthly climatology")
            return self.MONTHLY_CLIMATOLOGY
        else:
            logging.debug(f"Dataset has {time_size} time points - not a standard climatology")
            return None

    def prepare_reference_climatology(self, ds: xr.Dataset) -> Tuple[xr.Dataset, str]:
        """
        Prepare reference data for climatology evaluation.

        Args:
            ds: Reference dataset

        Returns:
            Tuple of (processed dataset, climatology type)
        """
        clim_type = self.detect_climatology_type(ds)

        if clim_type is None:
            return ds, None

        if clim_type == self.ANNUAL_CLIMATOLOGY:
            # Set time to standard annual climatology time
            if 'time' in ds.dims:
                ds = ds.isel(time=0).expand_dims('time')
            else:
                ds = ds.expand_dims('time')
            ds = ds.assign_coords(time=[self.ANNUAL_TIME])
            logging.info(f"Reference set to annual climatology with time: {self.ANNUAL_TIME}")

        elif clim_type == self.MONTHLY_CLIMATOLOGY:
            # Set times to standard monthly climatology times
            if len(ds.time) != 12:
                logging.error(f"Monthly climatology should have 12 time points, got {len(ds.time)}")
                return ds, None
            ds = ds.assign_coords(time=self.MONTHLY_TIMES)
            logging.info(f"Reference set to monthly climatology with times: 2000-01-15 to 2000-12-15")

        return ds, clim_type

    def prepare_simulation_climatology(self, ds: xr.Dataset, clim_type: str) -> xr.Dataset:
        """
        Prepare simulation data to match reference climatology.

        Args:
            ds: Simulation dataset with full time series
            clim_type: Climatology type from reference data

        Returns:
            Processed dataset with climatological mean
        """
        if clim_type is None:
            return ds

        if 'time' not in ds.dims:
            logging.warning("Simulation data has no time dimension for climatology calculation")
            return ds

        if clim_type == self.ANNUAL_CLIMATOLOGY:
            # Calculate multi-year mean
            ds_mean = ds.mean(dim='time', skipna=True)
            ds_mean = ds_mean.expand_dims('time')
            ds_mean = ds_mean.assign_coords(time=[self.ANNUAL_TIME])
            logging.info("Calculated annual climatology from simulation data")

        elif clim_type == self.MONTHLY_CLIMATOLOGY:
            # Calculate multi-year monthly mean
            try:
                # Group by month and calculate mean
                ds_monthly = ds.groupby('time.month').mean(dim='time', skipna=True)

                # Reorder to ensure months are in order (1-12)
                ds_monthly = ds_monthly.sortby('month')

                # Drop the month coordinate and create new time dimension
                ds_mean = ds_monthly.rename({'month': 'time'})
                ds_mean = ds_mean.assign_coords(time=self.MONTHLY_TIMES)

                logging.info("Calculated monthly climatology from simulation data")
            except Exception as e:
                logging.error(f"Error calculating monthly climatology: {e}")
                return ds
        else:
            logging.warning(f"Unknown climatology type: {clim_type}")
            return ds

        return ds_mean

    def is_metric_supported(self, metric_name: str) -> bool:
        """
        Check if a metric is supported for climatology evaluation.

        Args:
            metric_name: Name of the metric

        Returns:
            bool: True if supported, False otherwise
        """
        is_supported = metric_name not in self.UNSUPPORTED_METRICS

        if not is_supported:
            logging.info(f"Metric '{metric_name}' is not supported for climatology evaluation - skipping")

        return is_supported

    def validate_climatology_compatibility(self, ref_ds: xr.Dataset, sim_ds: xr.Dataset) -> bool:
        """
        Validate that reference and simulation datasets are compatible for climatology evaluation.

        Args:
            ref_ds: Reference dataset
            sim_ds: Simulation dataset

        Returns:
            bool: True if compatible, False otherwise
        """
        # Check if both have time dimension
        if 'time' not in ref_ds.dims or 'time' not in sim_ds.dims:
            logging.warning("Both datasets must have time dimension for climatology evaluation")
            return False

        # Check if time dimensions match
        ref_time_size = len(ref_ds.time)
        sim_time_size = len(sim_ds.time)

        if ref_time_size != sim_time_size:
            logging.error(f"Time dimension mismatch: reference has {ref_time_size}, simulation has {sim_time_size}")
            return False

        # Check if time coordinates match (compare as timestamps)
        try:
            ref_times = pd.to_datetime(ref_ds.time.values)
            sim_times = pd.to_datetime(sim_ds.time.values)

            # Compare dates only (ignore time of day)
            ref_dates = ref_times.normalize()
            sim_dates = sim_times.normalize()

            if not all(ref_dates == sim_dates):
                logging.warning("Time coordinates do not match exactly")
                logging.debug(f"Reference times: {ref_dates}")
                logging.debug(f"Simulation times: {sim_dates}")
                # This is just a warning, not an error
        except Exception as e:
            logging.debug(f"Could not compare time coordinates: {e}")

        return True


def process_climatology_evaluation(ref_ds: xr.Dataset, sim_ds: xr.Dataset,
                                   metrics: List[str]) -> Tuple[Optional[xr.Dataset],
                                                                  Optional[xr.Dataset],
                                                                  List[str]]:
    """
    Process datasets for climatology evaluation.

    Args:
        ref_ds: Reference dataset
        sim_ds: Simulation dataset
        metrics: List of metrics to evaluate

    Returns:
        Tuple of (processed reference, processed simulation, supported metrics)
    """
    processor = ClimatologyProcessor()

    # Prepare reference climatology
    ref_processed, clim_type = processor.prepare_reference_climatology(ref_ds)

    # If not a climatology, return original data
    if clim_type is None:
        logging.debug("Not a climatology evaluation - using original time series")
        return ref_ds, sim_ds, metrics

    # Prepare simulation climatology
    sim_processed = processor.prepare_simulation_climatology(sim_ds, clim_type)

    # Validate compatibility
    if not processor.validate_climatology_compatibility(ref_processed, sim_processed):
        logging.error("Climatology compatibility validation failed")
        return None, None, []

    # Filter supported metrics
    supported_metrics = [m for m in metrics if processor.is_metric_supported(m)]

    if len(supported_metrics) < len(metrics):
        unsupported = set(metrics) - set(supported_metrics)
        logging.info(f"Skipping unsupported metrics for climatology: {unsupported}")

    return ref_processed, sim_processed, supported_metrics
