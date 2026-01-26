# -*- coding: utf-8 -*-
"""
Enhanced Data Processing Pipeline for OpenBench

This module provides a comprehensive data processing pipeline with validation,
transformation, and error handling capabilities.

Author: Zhongwang Wei
Version: 1.0
Date: July 2025
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

# Import interfaces and exceptions
try:
    from openbench.util.Mod_Interfaces import IDataProcessor, ProcessingPipeline, BaseProcessor
    from openbench.util.Mod_Exceptions import DataProcessingError, ValidationError, error_handler
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False
    IDataProcessor = object
    ProcessingPipeline = object
    BaseProcessor = object
    DataProcessingError = Exception
    ValidationError = Exception
    def error_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class DataValidationProcessor(BaseProcessor if _HAS_DEPENDENCIES else object):
    """Processor for data validation operations."""
    
    def __init__(self, name: str = "DataValidator"):
        if _HAS_DEPENDENCIES:
            super().__init__(name)
        else:
            self.name = name
        
        self.validation_rules = {
            'required_dims': ['lat', 'lon'],
            'required_coords': [],
            'valid_data_range': None,
            'check_missing': True,
            'check_infinite': True
        }
    
    def set_validation_rules(self, rules: Dict[str, Any]) -> None:
        """Set custom validation rules."""
        self.validation_rules.update(rules)
    
    def validate_input(self, data: xr.Dataset) -> bool:
        """Enhanced input validation with detailed checks."""
        if not isinstance(data, xr.Dataset):
            return False
        
        # Check required dimensions
        for dim in self.validation_rules.get('required_dims', []):
            if dim not in data.dims:
                logging.warning(f"Missing required dimension: {dim}")
                return False
        
        # Check required coordinates
        for coord in self.validation_rules.get('required_coords', []):
            if coord not in data.coords:
                logging.warning(f"Missing required coordinate: {coord}")
                return False
        
        # Check for data variables
        if len(data.data_vars) == 0:
            logging.warning("Dataset has no data variables")
            return False
        
        return True
    
    @error_handler(reraise=True)
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """Process data with comprehensive validation."""
        if not self.validate_input(data):
            raise ValidationError("Dataset validation failed")
        
        # Check for missing values
        if self.validation_rules.get('check_missing', True):
            for var_name, var_data in data.data_vars.items():
                missing_count = var_data.isnull().sum().item()
                if missing_count > 0:
                    total_count = var_data.size
                    missing_ratio = missing_count / total_count
                    logging.info(f"Variable {var_name}: {missing_count}/{total_count} "
                               f"({missing_ratio:.2%}) missing values")
        
        # Check for infinite values
        if self.validation_rules.get('check_infinite', True):
            for var_name, var_data in data.data_vars.items():
                if np.issubdtype(var_data.dtype, np.floating):
                    infinite_count = np.isinf(var_data).sum().item()
                    if infinite_count > 0:
                        logging.warning(f"Variable {var_name}: {infinite_count} infinite values found")
        
        # Check data range if specified
        valid_range = self.validation_rules.get('valid_data_range')
        if valid_range:
            for var_name, var_data in data.data_vars.items():
                if np.issubdtype(var_data.dtype, np.number):
                    min_val, max_val = valid_range
                    out_of_range = ((var_data < min_val) | (var_data > max_val)).sum().item()
                    if out_of_range > 0:
                        logging.warning(f"Variable {var_name}: {out_of_range} values out of range [{min_val}, {max_val}]")
        
        return data


class CoordinateProcessor(BaseProcessor if _HAS_DEPENDENCIES else object):
    """Processor for coordinate standardization and transformation."""
    
    def __init__(self, name: str = "CoordinateProcessor"):
        if _HAS_DEPENDENCIES:
            super().__init__(name)
        else:
            self.name = name
        
        self.coordinate_map = {
            'latitude': 'lat',
            'longitude': 'lon',
            'lat_ucat': 'lat',
            'lon_ucat': 'lon',
            'time': 'time'
        }
        self.normalize_longitude = True
    
    def validate_input(self, data: xr.Dataset) -> bool:
        """Validate coordinate structure."""
        return isinstance(data, xr.Dataset)
    
    @error_handler(reraise=True)
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """Process coordinates for standardization."""
        result = data.copy()
        
        # Rename coordinates according to mapping
        for old_name, new_name in self.coordinate_map.items():
            if old_name in result.coords and old_name != new_name:
                result = result.rename({old_name: new_name})
                logging.debug(f"Renamed coordinate {old_name} to {new_name}")
        
        # Normalize longitude to [-180, 180] if requested
        if self.normalize_longitude and 'lon' in result.coords:
            lon_values = result.coords['lon'].values
            if np.any(lon_values > 180):
                lon_normalized = ((lon_values + 180) % 360) - 180
                result = result.assign_coords(lon=lon_normalized)
                logging.debug("Normalized longitude to [-180, 180] range")
        
        # Sort coordinates for consistency
        if 'lat' in result.coords:
            result = result.sortby('lat')
        if 'lon' in result.coords:
            result = result.sortby('lon')
        if 'time' in result.coords:
            result = result.sortby('time')
        
        return result


class UnitConversionProcessor(BaseProcessor if _HAS_DEPENDENCIES else object):
    """Processor for unit conversion operations."""
    
    def __init__(self, name: str = "UnitConverter"):
        if _HAS_DEPENDENCIES:
            super().__init__(name)
        else:
            self.name = name
        
        self.conversion_rules = {}
        self.target_units = {}
    
    def set_conversion_rules(self, rules: Dict[str, Dict[str, float]]) -> None:
        """Set unit conversion rules."""
        self.conversion_rules = rules
    
    def set_target_units(self, units: Dict[str, str]) -> None:
        """Set target units for variables."""
        self.target_units = units
    
    def validate_input(self, data: xr.Dataset) -> bool:
        """Validate input for unit conversion."""
        return isinstance(data, xr.Dataset)
    
    @error_handler(reraise=True)
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """Process data with unit conversions."""
        result = data.copy()
        
        for var_name, var_data in result.data_vars.items():
            # Check if conversion is needed
            if var_name in self.target_units:
                target_unit = self.target_units[var_name]
                current_unit = var_data.attrs.get('units', '')
                
                if current_unit and current_unit != target_unit:
                    # Look for conversion rule
                    conversion_key = f"{current_unit}_to_{target_unit}"
                    if conversion_key in self.conversion_rules:
                        factor = self.conversion_rules[conversion_key]
                        result[var_name] = var_data * factor
                        result[var_name].attrs['units'] = target_unit
                        logging.info(f"Converted {var_name} from {current_unit} to {target_unit}")
                    else:
                        logging.warning(f"No conversion rule found for {conversion_key}")
        
        return result


class QualityControlProcessor(BaseProcessor if _HAS_DEPENDENCIES else object):
    """Processor for data quality control operations."""
    
    def __init__(self, name: str = "QualityControl"):
        if _HAS_DEPENDENCIES:
            super().__init__(name)
        else:
            self.name = name
        
        self.outlier_methods = ['iqr', 'zscore', 'custom']
        self.outlier_thresholds = {
            'iqr_factor': 1.5,
            'zscore_threshold': 3.0
        }
    
    def set_outlier_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set outlier detection thresholds."""
        self.outlier_thresholds.update(thresholds)
    
    def validate_input(self, data: xr.Dataset) -> bool:
        """Validate input for quality control."""
        return isinstance(data, xr.Dataset)
    
    def detect_outliers_iqr(self, data: xr.DataArray) -> xr.DataArray:
        """Detect outliers using Interquartile Range method."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        factor = self.outlier_thresholds.get('iqr_factor', 1.5)
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        return (data < lower_bound) | (data > upper_bound)
    
    def detect_outliers_zscore(self, data: xr.DataArray) -> xr.DataArray:
        """Detect outliers using Z-score method."""
        threshold = self.outlier_thresholds.get('zscore_threshold', 3.0)
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    @error_handler(reraise=True)
    def process(self, data: xr.Dataset, outlier_method: str = 'iqr', **kwargs) -> xr.Dataset:
        """Process data with quality control."""
        result = data.copy()
        
        for var_name, var_data in result.data_vars.items():
            if np.issubdtype(var_data.dtype, np.number):
                # Detect outliers
                if outlier_method == 'iqr':
                    outliers = self.detect_outliers_iqr(var_data)
                elif outlier_method == 'zscore':
                    outliers = self.detect_outliers_zscore(var_data)
                else:
                    continue
                
                outlier_count = outliers.sum().item()
                if outlier_count > 0:
                    total_count = var_data.size
                    outlier_ratio = outlier_count / total_count
                    logging.info(f"Variable {var_name}: {outlier_count}/{total_count} "
                               f"({outlier_ratio:.2%}) outliers detected using {outlier_method}")
                    
                    # Option to flag or remove outliers
                    if kwargs.get('remove_outliers', False):
                        result[var_name] = var_data.where(~outliers)
                    else:
                        # Add outlier flag as attribute
                        result[var_name].attrs['outliers_detected'] = outlier_count
        
        return result


class DataPipelineBuilder:
    """Builder class for constructing data processing pipelines."""
    
    def __init__(self, name: str = "DataPipeline"):
        """Initialize pipeline builder."""
        self.name = name
        if _HAS_DEPENDENCIES:
            self.pipeline = ProcessingPipeline(name)
        else:
            self.pipeline = None
            self.processors = []
    
    def add_validation(self, validation_rules: Optional[Dict[str, Any]] = None) -> 'DataPipelineBuilder':
        """Add data validation processor."""
        validator = DataValidationProcessor()
        if validation_rules:
            validator.set_validation_rules(validation_rules)
        
        if _HAS_DEPENDENCIES:
            self.pipeline.add_processor(validator)
        else:
            self.processors.append(validator)
        
        return self
    
    def add_coordinate_processing(self, coordinate_map: Optional[Dict[str, str]] = None) -> 'DataPipelineBuilder':
        """Add coordinate processing."""
        coord_processor = CoordinateProcessor()
        if coordinate_map:
            coord_processor.coordinate_map = coordinate_map
        
        if _HAS_DEPENDENCIES:
            self.pipeline.add_processor(coord_processor)
        else:
            self.processors.append(coord_processor)
        
        return self
    
    def add_unit_conversion(self, conversion_rules: Optional[Dict] = None, 
                          target_units: Optional[Dict] = None) -> 'DataPipelineBuilder':
        """Add unit conversion processor."""
        unit_processor = UnitConversionProcessor()
        if conversion_rules:
            unit_processor.set_conversion_rules(conversion_rules)
        if target_units:
            unit_processor.set_target_units(target_units)
        
        if _HAS_DEPENDENCIES:
            self.pipeline.add_processor(unit_processor)
        else:
            self.processors.append(unit_processor)
        
        return self
    
    def add_quality_control(self, outlier_thresholds: Optional[Dict] = None) -> 'DataPipelineBuilder':
        """Add quality control processor."""
        qc_processor = QualityControlProcessor()
        if outlier_thresholds:
            qc_processor.set_outlier_thresholds(outlier_thresholds)
        
        if _HAS_DEPENDENCIES:
            self.pipeline.add_processor(qc_processor)
        else:
            self.processors.append(qc_processor)
        
        return self
    
    def add_custom_processor(self, processor: IDataProcessor) -> 'DataPipelineBuilder':
        """Add a custom processor."""
        if _HAS_DEPENDENCIES:
            self.pipeline.add_processor(processor)
        else:
            self.processors.append(processor)
        
        return self
    
    def build(self) -> Union[ProcessingPipeline, 'SimplePipeline']:
        """Build the final pipeline."""
        if _HAS_DEPENDENCIES:
            return self.pipeline
        else:
            return SimplePipeline(self.name, self.processors)


class SimplePipeline:
    """Simple pipeline implementation for fallback mode."""
    
    def __init__(self, name: str, processors: List):
        """Initialize simple pipeline."""
        self.name = name
        self.processors = processors
    
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """Process data through all processors."""
        result = data
        
        for processor in self.processors:
            if hasattr(processor, 'validate_input'):
                if not processor.validate_input(result):
                    raise ValueError(f"Invalid input for processor: {processor.name}")
            
            result = processor.process(result, **kwargs)
        
        return result
    
    def get_processor_count(self) -> int:
        """Get number of processors."""
        return len(self.processors)


def create_standard_pipeline(config: Optional[Dict[str, Any]] = None) -> Union[ProcessingPipeline, SimplePipeline]:
    """
    Create a standard data processing pipeline with common processors.
    
    Args:
        config: Configuration dictionary for pipeline setup
        
    Returns:
        Configured data processing pipeline
    """
    config = config or {}
    
    builder = DataPipelineBuilder("StandardPipeline")
    
    # Add validation if requested
    if config.get('enable_validation', True):
        validation_rules = config.get('validation_rules', {})
        builder.add_validation(validation_rules)
    
    # Add coordinate processing if requested
    if config.get('enable_coordinate_processing', True):
        coordinate_map = config.get('coordinate_map', {})
        builder.add_coordinate_processing(coordinate_map)
    
    # Add unit conversion if requested
    if config.get('enable_unit_conversion', False):
        conversion_rules = config.get('conversion_rules', {})
        target_units = config.get('target_units', {})
        builder.add_unit_conversion(conversion_rules, target_units)
    
    # Add quality control if requested
    if config.get('enable_quality_control', False):
        outlier_thresholds = config.get('outlier_thresholds', {})
        builder.add_quality_control(outlier_thresholds)
    
    return builder.build()


# Convenience function for quick data processing
@error_handler(reraise=True)
def process_dataset(
    data: xr.Dataset,
    pipeline_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> xr.Dataset:
    """
    Process a dataset using a standard pipeline.
    
    Args:
        data: Input dataset
        pipeline_config: Pipeline configuration
        **kwargs: Additional processing parameters
        
    Returns:
        Processed dataset
    """
    pipeline = create_standard_pipeline(pipeline_config)
    return pipeline.process(data, **kwargs)