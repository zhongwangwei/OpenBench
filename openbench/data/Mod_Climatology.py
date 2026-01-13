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

        # Legacy constants (kept for backward compatibility, but no longer used internally)
        # Modern code should use syear-based time coordinate generation
        self.ANNUAL_TIME = pd.Timestamp('2000-06-15')
        self.MONTHLY_TIMES = pd.date_range('2000-01-15', periods=12, freq='MS') + pd.Timedelta(days=14)

        # Metrics that never support climatology evaluation (regardless of time points)
        self.NEVER_SUPPORTED_METRICS = [
            'nPhaseScore',  # Phase score requires temporal cycle
            'nIavScore',    # Interannual variability score requires multiple years
            'APFB',         # Annual Peak Flow Bias requires multi-year data with hydrological year grouping
        ]

        # Metrics that require multiple time points (not supported for annual climatology with 1 time point)
        self.MULTI_TIME_METRICS = [
            'correlation',      # Correlation requires at least 2 time points
            'correlation_R2',   # R-squared correlation
            'ubcorrelation',    # Unbiased correlation
            'ubcorrelation_R2', # Unbiased correlation R-squared
            'NSE',             # Nash-Sutcliffe Efficiency
            'KGE',             # Kling-Gupta Efficiency
            'KGESS',           # KGE with multiple components
            'mKGE',            # Modified KGE
            'nKGE',            # Normalized KGE
            'pc_ampli',        # Phase and amplitude require temporal variation
            'pc_max',          # Temporal phase
            'pc_min',          # Temporal phase
            'CRMSD',           # Centered RMSD uses std(dim='time') and correlation
            'rv',              # Relative variability requires std(dim='time')
            'cp',              # Coefficient of Persistence uses .diff(dim='time')
            'br2',             # R-squared × slope requires correlation
        ]

    def is_climatology_mode(self, compare_tim_res: str) -> bool:
        """
        Check if the configuration indicates climatology mode.

        Args:
            compare_tim_res: Comparison time resolution from configuration

        Returns:
            bool: True if compare_tim_res indicates climatology mode
        """
        if not compare_tim_res:
            return False
        compare_tim_res_str = str(compare_tim_res).strip().lower()
        return compare_tim_res_str in ['climatology-year', 'climatology-month']

    def get_climatology_type_from_config(self, compare_tim_res: str) -> Optional[str]:
        """
        Get the climatology type from compare_tim_res configuration.

        Args:
            compare_tim_res: Comparison time resolution from configuration

        Returns:
            str: Climatology type ('annual', 'monthly', or None)
        """
        if not compare_tim_res:
            return None

        compare_tim_res_str = str(compare_tim_res).strip().lower()

        if compare_tim_res_str == 'climatology-year':
            logging.info("Climatology-year mode - using annual climatology")
            return self.ANNUAL_CLIMATOLOGY
        elif compare_tim_res_str == 'climatology-month':
            logging.info("Climatology-month mode - using monthly climatology")
            return self.MONTHLY_CLIMATOLOGY
        else:
            return None

    def prepare_reference_climatology(self, ds: xr.Dataset, clim_type: str, syear: int) -> Optional[xr.Dataset]:
        """
        Prepare reference data for climatology evaluation.

        Args:
            ds: Reference dataset
            clim_type: Climatology type ('annual' or 'monthly')
            syear: Start year for time coordinate assignment

        Returns:
            Processed dataset, or None if processing fails
        """
        if clim_type is None:
            raise ValueError("Climatology type cannot be None")

        # Check time dimension status
        has_time_dim = 'time' in ds.dims or 'time' in ds.coords
        time_size = len(ds.time) if has_time_dim else 0

        if clim_type == self.ANNUAL_CLIMATOLOGY:
            # Annual climatology processing
            if not has_time_dim or time_size == 0:
                # No time dimension - add time dimension with syear-01-01
                ds = ds.expand_dims('time')
                annual_time = pd.Timestamp(f'{syear}-01-01')
                ds = ds.assign_coords(time=[annual_time])
                logging.info(f"Reference: Added time dimension with {annual_time}")
            elif time_size == 1:
                # Single time point - set to syear-01-01
                ds = ds.isel(time=0).expand_dims('time')
                annual_time = pd.Timestamp(f'{syear}-01-01')
                ds = ds.assign_coords(time=[annual_time])
                logging.info(f"Reference: Set single time point to {annual_time}")
            elif time_size == 12:
                # 12 time points - set to syear's 12 months, then average to annual
                monthly_times = pd.date_range(f'{syear}-01-01', periods=12, freq='MS') + pd.Timedelta(days=14)
                ds = ds.assign_coords(time=monthly_times)
                # Average to annual climatology
                ds = ds.mean(dim='time', skipna=True).expand_dims('time')
                annual_time = pd.Timestamp(f'{syear}-01-01')
                ds = ds.assign_coords(time=[annual_time])
                logging.info(f"Reference: Averaged 12 months to annual climatology at {annual_time}")
            else:
                # Multiple time points (e.g., daily data) - average to annual climatology
                logging.info(f"Reference: Processing {time_size} time points to annual climatology")
                ds = ds.mean(dim='time', skipna=True).expand_dims('time')
                annual_time = pd.Timestamp(f'{syear}-01-01')
                ds = ds.assign_coords(time=[annual_time])
                logging.info(f"Reference: Averaged {time_size} time points to annual climatology at {annual_time}")

        elif clim_type == self.MONTHLY_CLIMATOLOGY:
            # Monthly climatology processing
            if not has_time_dim or time_size == 0:
                raise ValueError("Monthly climatology requires time dimension with data")
            elif time_size == 12:
                # 12 time points - set to syear's 12 months as monthly climatology
                monthly_times = pd.date_range(f'{syear}-01-01', periods=12, freq='MS') + pd.Timedelta(days=14)
                ds = ds.assign_coords(time=monthly_times)
                logging.info(f"Reference: Set 12 time points to monthly climatology for year {syear}")
            else:
                # Multiple time points - calculate monthly climatology via groupby
                try:
                    logging.info(f"Reference: Processing {time_size} time points to monthly climatology")
                    ds_monthly = ds.groupby('time.month').mean(dim='time', skipna=True)

                    # Reorder to ensure months are in order (1-12)
                    ds_monthly = ds_monthly.sortby('month')

                    # Check if we got 12 months
                    if len(ds_monthly.month) != 12:
                        missing_months = set(range(1, 13)) - set(ds_monthly.month.values)
                        raise ValueError(
                            f"Expected 12 months after groupby, got {len(ds_monthly.month)}. "
                            f"Missing months: {sorted(missing_months)}"
                        )

                    # Rename month dimension to time and assign monthly times
                    ds = ds_monthly.rename({'month': 'time'})
                    monthly_times = pd.date_range(f'{syear}-01-01', periods=12, freq='MS') + pd.Timedelta(days=14)
                    ds = ds.assign_coords(time=monthly_times)
                    logging.info(f"Reference: Calculated monthly climatology for year {syear} from {time_size} time points")
                except Exception as e:
                    logging.error(f"Error calculating monthly climatology from reference data: {e}")
                    raise

        return ds

    def prepare_simulation_climatology(self, ds: xr.Dataset, clim_type: str, syear: int) -> xr.Dataset:
        """
        Prepare simulation data to match reference climatology.

        Args:
            ds: Simulation dataset with full time series
            clim_type: Climatology type from reference data
            syear: Start year for time coordinate assignment

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
            annual_time = pd.Timestamp(f'{syear}-01-01')
            ds_mean = ds_mean.assign_coords(time=[annual_time])
            logging.info(f"Calculated annual climatology from simulation data at {annual_time}")

        elif clim_type == self.MONTHLY_CLIMATOLOGY:
            # Calculate multi-year monthly mean
            try:
                # Group by month and calculate mean
                ds_monthly = ds.groupby('time.month').mean(dim='time', skipna=True)

                # Reorder to ensure months are in order (1-12)
                ds_monthly = ds_monthly.sortby('month')

                # Drop the month coordinate and create new time dimension
                ds_mean = ds_monthly.rename({'month': 'time'})
                monthly_times = pd.date_range(f'{syear}-01-01', periods=12, freq='MS') + pd.Timedelta(days=14)
                ds_mean = ds_mean.assign_coords(time=monthly_times)

                logging.info(f"Calculated monthly climatology from simulation data for year {syear}")
            except Exception as e:
                logging.error(f"Error calculating monthly climatology: {e}")
                return ds
        else:
            logging.warning(f"Unknown climatology type: {clim_type}")
            return ds

        return ds_mean

    def is_metric_supported(self, metric_name: str, clim_type: str = None, time_points: int = None) -> bool:
        """
        Check if a metric is supported for climatology evaluation.

        Args:
            metric_name: Name of the metric
            clim_type: Climatology type ('annual' or 'monthly')
            time_points: Number of time points in the climatology data

        Returns:
            bool: True if supported, False otherwise
        """
        # Check if metric is never supported for climatology
        if metric_name in self.NEVER_SUPPORTED_METRICS:
            logging.info(f"Metric '{metric_name}' is not supported for climatology evaluation - skipping")
            return False

        # Check if metric requires multiple time points
        if metric_name in self.MULTI_TIME_METRICS:
            # For annual climatology with 1 time point, skip multi-time metrics
            if clim_type == self.ANNUAL_CLIMATOLOGY and time_points == 1:
                logging.info(f"Metric '{metric_name}' requires multiple time points, skipping for annual climatology (1 time point)")
                return False
            # For monthly climatology with less than 2 time points, skip
            if time_points is not None and time_points < 2:
                logging.info(f"Metric '{metric_name}' requires at least 2 time points, got {time_points} - skipping")
                return False

        return True

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
                logging.error("Time coordinates do not match between reference and simulation")
                logging.error(f"Reference times: {ref_dates.values}")
                logging.error(f"Simulation times: {sim_dates.values}")
                # For climatology evaluation, time coordinate match is REQUIRED
                return False
        except Exception as e:
            logging.warning(f"Could not compare time coordinates: {e}")
            # If we can't compare, assume they match and let downstream validation catch issues

        return True


def process_climatology_evaluation(ref_ds: xr.Dataset, sim_ds: xr.Dataset,
                                   metrics: List[str],
                                   compare_tim_res: str = None,
                                   syear: int = None) -> Tuple[Optional[xr.Dataset],
                                                                           Optional[xr.Dataset],
                                                                           List[str]]:
    """
    Process datasets for climatology evaluation.

    Args:
        ref_ds: Reference dataset
        sim_ds: Simulation dataset
        metrics: List of metrics to evaluate
        compare_tim_res: Comparison time resolution (e.g., 'Climatology-year', 'Climatology-month')
        syear: Start year for time coordinate assignment (default: 2000)

    Returns:
        Tuple of (processed reference, processed simulation, supported metrics)
    """
    processor = ClimatologyProcessor()

    # Set default syear if not provided
    if syear is None:
        syear = 2000

    # Check if climatology mode is enabled via compare_tim_res
    is_climatology = processor.is_climatology_mode(compare_tim_res)

    if not is_climatology:
        # No climatology configuration - return None to indicate regular time series mode
        logging.debug("Non-climatology evaluation - using original time series")
        return None, None, metrics

    # Get climatology type from compare_tim_res
    clim_type = processor.get_climatology_type_from_config(compare_tim_res)

    if clim_type is None:
        logging.error(f"Invalid compare_tim_res value: {compare_tim_res}")
        return None, None, []

    # Validate syear
    if syear < 1000 or syear > 9999:
        logging.error(f"Invalid syear value: {syear}. Must be between 1000 and 9999.")
        return None, None, []

    # Climatology mode explicitly enabled
    logging.info(f"Climatology evaluation mode activated (compare_tim_res='{compare_tim_res}', syear={syear})")

    # Prepare reference and simulation climatology
    try:
        ref_processed = processor.prepare_reference_climatology(ref_ds, clim_type, syear)
        sim_processed = processor.prepare_simulation_climatology(sim_ds, clim_type, syear)
    except Exception as e:
        logging.error(f"Failed to prepare climatology datasets: {e}")
        return None, None, []

    # Validate compatibility
    if not processor.validate_climatology_compatibility(ref_processed, sim_processed):
        logging.error("Climatology compatibility validation failed")
        return None, None, []

    # Get number of time points from processed reference data
    if 'time' in ref_processed.dims:
        time_points = len(ref_processed.time)
        if time_points == 0:
            logging.error("Time dimension exists but contains no time points")
            return None, None, []
    else:
        time_points = 1
        logging.debug("No time dimension found, treating as single time point")

    # Filter supported metrics based on climatology type and time points
    supported_metrics = [
        m for m in metrics
        if processor.is_metric_supported(m, clim_type=clim_type, time_points=time_points)
    ]

    # Provide detailed information about skipped metrics
    if len(supported_metrics) < len(metrics):
        unsupported = set(metrics) - set(supported_metrics)

        # Categorize why metrics were skipped
        never_supported = unsupported & set(processor.NEVER_SUPPORTED_METRICS)
        need_multi_time = unsupported & set(processor.MULTI_TIME_METRICS)
        other_unsupported = unsupported - never_supported - need_multi_time

        if never_supported:
            logging.info(f"Skipped (never supported for climatology): {never_supported}")
        if need_multi_time:
            logging.info(f"Skipped (require ≥2 time points, have {time_points}): {need_multi_time}")
        if other_unsupported:
            logging.info(f"Skipped (other reasons): {other_unsupported}")

    logging.info(
        f"Climatology evaluation ready: {clim_type} climatology with {time_points} time point(s), "
        f"{len(supported_metrics)}/{len(metrics)} metrics will be evaluated"
    )
    return ref_processed, sim_processed, supported_metrics
