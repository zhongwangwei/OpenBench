# -*- coding: utf-8 -*-
"""
Modular Evaluation Engine for OpenBench

This module provides a modular evaluation engine with pluggable metrics,
unified interfaces, and enhanced error handling.

Author: Zhongwang Wei  
Version: 1.0
Date: July 2025
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
import pandas as pd

# Import dependencies
try:
    from openbench.util.Mod_Interfaces import IEvaluationEngine, IMetricsCalculator, BaseEvaluator
    from openbench.util.Mod_Exceptions import EvaluationError, error_handler, ValidationError
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False
    IEvaluationEngine = object
    IMetricsCalculator = object
    BaseEvaluator = object
    EvaluationError = Exception
    ValidationError = Exception
    def error_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        print("Error handler imported")
        exit()

# Import caching
try:
    from data.Mod_CacheSystem import cached, get_cache_manager
    _HAS_CACHE = True
except ImportError:
    _HAS_CACHE = False
    def cached(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class MetricCalculator(IMetricsCalculator if _HAS_DEPENDENCIES else object):
    """Base class for metric calculators."""
    
    def __init__(self, name: str, description: str = "", unit: str = ""):
        """
        Initialize metric calculator.
        
        Args:
            name: Metric name
            description: Metric description
            unit: Metric unit
        """
        self.name = name
        self.description = description
        self.unit = unit
    
    def get_name(self) -> str:
        """Get metric name."""
        return self.name
    
    def get_description(self) -> str:
        """Get metric description."""
        return self.description
    
    def get_unit(self) -> str:
        """Get metric unit."""
        return self.unit
    
    @abstractmethod
    def calculate(self, simulation: xr.Dataset, reference: xr.Dataset) -> float:
        """Calculate metric value."""
        pass
    
    def validate_inputs(self, simulation: xr.Dataset, reference: xr.Dataset) -> bool:
        """Validate input datasets."""
        if not isinstance(simulation, xr.Dataset) or not isinstance(reference, xr.Dataset):
            return False
        
        # Check if datasets have compatible shapes
        sim_vars = list(simulation.data_vars.keys())
        ref_vars = list(reference.data_vars.keys())
        
        if not sim_vars or not ref_vars:
            return False
        
        return True


class BiasCalculator(MetricCalculator):
    """Calculator for bias metric."""
    
    def __init__(self):
        super().__init__("bias", "Mean bias", "units of variable")
    
    @error_handler(reraise=True)
    def calculate(self, simulation: xr.Dataset, reference: xr.Dataset) -> float:
        """Calculate bias."""
        if not self.validate_inputs(simulation, reference):
            raise ValidationError("Invalid inputs for bias calculation")
        
        # Get first data variable from each dataset
        sim_var = list(simulation.data_vars.keys())[0]
        ref_var = list(reference.data_vars.keys())[0]
        
        sim_data = simulation[sim_var]
        ref_data = reference[ref_var]
        
        # Calculate bias
        bias = (sim_data - ref_data).mean().item()
        return bias


class RMSECalculator(MetricCalculator):
    """Calculator for RMSE metric."""
    
    def __init__(self):
        super().__init__("RMSE", "Root Mean Square Error", "units of variable")
    
    @error_handler(reraise=True)
    def calculate(self, simulation: xr.Dataset, reference: xr.Dataset) -> float:
        """Calculate RMSE."""
        if not self.validate_inputs(simulation, reference):
            raise ValidationError("Invalid inputs for RMSE calculation")
        
        sim_var = list(simulation.data_vars.keys())[0]
        ref_var = list(reference.data_vars.keys())[0]
        
        sim_data = simulation[sim_var]
        ref_data = reference[ref_var]
        
        # Calculate RMSE
        mse = ((sim_data - ref_data) ** 2).mean()
        rmse = np.sqrt(mse).item()
        return rmse


class CorrelationCalculator(MetricCalculator):
    """Calculator for correlation coefficient."""
    
    def __init__(self):
        super().__init__("correlation", "Pearson correlation coefficient", "dimensionless")
    
    @error_handler(reraise=True)
    def calculate(self, simulation: xr.Dataset, reference: xr.Dataset) -> float:
        """Calculate correlation."""
        if not self.validate_inputs(simulation, reference):
            raise ValidationError("Invalid inputs for correlation calculation")
        
        sim_var = list(simulation.data_vars.keys())[0]
        ref_var = list(reference.data_vars.keys())[0]
        
        sim_data = simulation[sim_var].values.flatten()
        ref_data = reference[ref_var].values.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(sim_data) | np.isnan(ref_data))
        sim_clean = sim_data[mask]
        ref_clean = ref_data[mask]
        
        if len(sim_clean) < 2:
            return np.nan
        
        # Calculate correlation
        corr = np.corrcoef(sim_clean, ref_clean)[0, 1]
        return corr


class NSECalculator(MetricCalculator):
    """Calculator for Nash-Sutcliffe Efficiency."""
    
    def __init__(self):
        super().__init__("NSE", "Nash-Sutcliffe Efficiency", "dimensionless")
    
    @error_handler(reraise=True)
    def calculate(self, simulation: xr.Dataset, reference: xr.Dataset) -> float:
        """Calculate NSE."""
        if not self.validate_inputs(simulation, reference):
            raise ValidationError("Invalid inputs for NSE calculation")
        
        sim_var = list(simulation.data_vars.keys())[0]
        ref_var = list(reference.data_vars.keys())[0]
        
        sim_data = simulation[sim_var].values.flatten()
        ref_data = reference[ref_var].values.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(sim_data) | np.isnan(ref_data))
        sim_clean = sim_data[mask]
        ref_clean = ref_data[mask]
        
        if len(sim_clean) < 2:
            return np.nan
        
        # Calculate NSE
        numerator = np.sum((ref_clean - sim_clean) ** 2)
        denominator = np.sum((ref_clean - np.mean(ref_clean)) ** 2)
        
        if denominator == 0:
            return np.nan
        
        nse = 1 - (numerator / denominator)
        return nse


class ModularEvaluationEngine(BaseEvaluator if _HAS_DEPENDENCIES else object):
    """Modular evaluation engine with pluggable metrics."""
    
    def __init__(self, name: str = "ModularEvaluationEngine"):
        """Initialize modular evaluation engine."""
        if _HAS_DEPENDENCIES:
            super().__init__(name)
        else:
            self.name = name
            self._metrics_registry = {}
        
        # Register default metrics
        self._register_default_metrics()
        
        # Configuration
        self.output_format = 'netcdf'
        self.save_intermediates = False
        self.parallel_processing = False
    
    def _register_default_metrics(self):
        """Register default metric calculators."""
        default_metrics = [
            BiasCalculator(),
            RMSECalculator(),
            CorrelationCalculator(),
            NSECalculator()
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def register_metric(self, metric: IMetricsCalculator) -> None:
        """Register a metric calculator."""
        if _HAS_DEPENDENCIES:
            super().register_metric(metric)
        else:
            self._metrics_registry[metric.get_name()] = metric
    
    def unregister_metric(self, metric_name: str) -> bool:
        """Unregister a metric calculator."""
        if metric_name in self._metrics_registry:
            del self._metrics_registry[metric_name]
            return True
        return False
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        if _HAS_DEPENDENCIES:
            return super().get_supported_metrics()
        else:
            return list(self._metrics_registry.keys())
    
    def get_metric_info(self, metric_name: str) -> Dict[str, str]:
        """Get information about a specific metric."""
        if metric_name not in self._metrics_registry:
            raise ValueError(f"Metric '{metric_name}' not found")
        
        metric = self._metrics_registry[metric_name]
        return {
            'name': metric.get_name(),
            'description': metric.get_description(),
            'unit': getattr(metric, 'get_unit', lambda: 'unknown')()
        }
    
    def validate_datasets(self, simulation: xr.Dataset, reference: xr.Dataset) -> bool:
        """Validate datasets for evaluation."""
        if _HAS_DEPENDENCIES:
            return super().validate_datasets(simulation, reference)
        else:
            return isinstance(simulation, xr.Dataset) and isinstance(reference, xr.Dataset)
    
    @error_handler(reraise=True)
    def evaluate(
        self, 
        simulation: xr.Dataset, 
        reference: xr.Dataset, 
        metrics: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate simulation against reference data.
        
        Args:
            simulation: Simulation dataset
            reference: Reference dataset
            metrics: List of metrics to calculate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Validate inputs
        if not self.validate_datasets(simulation, reference):
            raise ValidationError("Dataset validation failed")
        
        # Check requested metrics
        available_metrics = set(self.get_supported_metrics())
        requested_metrics = set(metrics)
        unavailable = requested_metrics - available_metrics
        
        if unavailable:
            logging.warning(f"Unavailable metrics requested: {unavailable}")
            metrics = list(requested_metrics & available_metrics)
        
        # Calculate metrics
        results = {
            'metrics': {},
            'metadata': {
                'simulation_info': self._get_dataset_info(simulation),
                'reference_info': self._get_dataset_info(reference),
                'evaluation_time': pd.Timestamp.now().isoformat()
            }
        }
        
        for metric_name in metrics:
            try:
                calculator = self._metrics_registry[metric_name]
                value = calculator.calculate(simulation, reference)
                results['metrics'][metric_name] = {
                    'value': value,
                    'info': self.get_metric_info(metric_name)
                }
                logging.debug(f"Calculated {metric_name}: {value}")
            except Exception as e:
                logging.error(f"Error calculating {metric_name}: {e}")
                results['metrics'][metric_name] = {
                    'value': np.nan,
                    'error': str(e),
                    'info': self.get_metric_info(metric_name)
                }
        
        return results
    
    def _get_dataset_info(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Get information about a dataset."""
        return {
            'variables': list(dataset.data_vars.keys()),
            'dimensions': dict(dataset.dims),
            'coordinates': list(dataset.coords.keys()),
            'shape': {var: dataset[var].shape for var in dataset.data_vars},
            'attributes': dict(dataset.attrs)
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to file."""
        try:
            if self.output_format == 'json':
                import json
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif self.output_format == 'csv':
                # Convert to DataFrame for CSV
                metrics_data = []
                for metric_name, metric_data in results['metrics'].items():
                    metrics_data.append({
                        'metric': metric_name,
                        'value': metric_data['value'],
                        'description': metric_data['info']['description'],
                        'unit': metric_data['info']['unit']
                    })
                df = pd.DataFrame(metrics_data)
                df.to_csv(output_path, index=False)
            else:
                logging.warning(f"Unsupported output format: {self.output_format}")
        
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            raise EvaluationError(f"Failed to save results: {e}")


class GridEvaluationEngine(ModularEvaluationEngine):
    """Evaluation engine specialized for gridded data."""
    
    def __init__(self):
        super().__init__("GridEvaluationEngine")
        self.spatial_aggregation = 'mean'  # mean, median, sum
        self.temporal_aggregation = 'mean'
    
    @error_handler(reraise=True)
    def evaluate(
        self, 
        simulation: xr.Dataset, 
        reference: xr.Dataset, 
        metrics: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate gridded data with spatial/temporal aggregation options."""
        # Apply spatial aggregation if requested
        if kwargs.get('spatial_aggregation'):
            self.spatial_aggregation = kwargs['spatial_aggregation']
        
        if kwargs.get('temporal_aggregation'):
            self.temporal_aggregation = kwargs['temporal_aggregation']
        
        # Preprocess data for gridded evaluation
        sim_processed = self._preprocess_gridded_data(simulation)
        ref_processed = self._preprocess_gridded_data(reference)
        
        # Call parent evaluation
        results = super().evaluate(sim_processed, ref_processed, metrics, **kwargs)
        
        # Add grid-specific metadata
        results['metadata']['evaluation_type'] = 'gridded'
        results['metadata']['spatial_aggregation'] = self.spatial_aggregation
        results['metadata']['temporal_aggregation'] = self.temporal_aggregation
        
        return results
    
    def _preprocess_gridded_data(self, data: xr.Dataset) -> xr.Dataset:
        """Preprocess gridded data for evaluation."""
        # Apply temporal aggregation if time dimension exists
        if 'time' in data.dims and self.temporal_aggregation != 'none':
            if self.temporal_aggregation == 'mean':
                data = data.mean(dim='time')
            elif self.temporal_aggregation == 'median':
                data = data.median(dim='time')
            elif self.temporal_aggregation == 'sum':
                data = data.sum(dim='time')
        
        return data


class StationEvaluationEngine(ModularEvaluationEngine):
    """Evaluation engine specialized for station data."""
    
    def __init__(self):
        super().__init__("StationEvaluationEngine")
        self.station_aggregation = 'individual'  # individual, mean, weighted
    
    @error_handler(reraise=True)
    def evaluate(
        self, 
        simulation: xr.Dataset, 
        reference: xr.Dataset, 
        metrics: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate station data with station-specific processing."""
        # Call parent evaluation
        results = super().evaluate(simulation, reference, metrics, **kwargs)
        
        # Add station-specific metadata
        results['metadata']['evaluation_type'] = 'station'
        results['metadata']['station_aggregation'] = self.station_aggregation
        
        return results


def create_evaluation_engine(engine_type: str = 'modular', **config) -> ModularEvaluationEngine:
    """
    Create an evaluation engine instance.
    
    Args:
        engine_type: Type of engine ('modular', 'grid', 'station')
        **config: Configuration parameters
        
    Returns:
        Evaluation engine instance
    """
    if engine_type == 'grid':
        engine = GridEvaluationEngine()
    elif engine_type == 'station':
        engine = StationEvaluationEngine()
    else:
        engine = ModularEvaluationEngine()
    
    # Apply configuration
    for key, value in config.items():
        if hasattr(engine, key):
            setattr(engine, key, value)
    
    return engine


# Convenience function for quick evaluation
@error_handler(reraise=True)
def evaluate_datasets(
    simulation: xr.Dataset,
    reference: xr.Dataset,
    metrics: List[str],
    engine_type: str = 'modular',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for dataset evaluation.
    
    Args:
        simulation: Simulation dataset
        reference: Reference dataset
        metrics: List of metrics to calculate
        engine_type: Type of evaluation engine
        **kwargs: Additional parameters
        
    Returns:
        Evaluation results
    """
    engine = create_evaluation_engine(engine_type, **kwargs)
    return engine.evaluate(simulation, reference, metrics, **kwargs)