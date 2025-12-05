# -*- coding: utf-8 -*-
"""
Output Management System for OpenBench

This module provides a unified output management system with support for
multiple formats, structured organization, and metadata handling.

Author: Zhongwang Wei  
Version: 1.0
Date: July 2025
"""

import os
import json
import logging
import shutil
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

# Import dependencies
try:
    from openbench.util.Mod_Interfaces import IOutputManager, IOutputFormatter, BaseComponent
    from openbench.util.Mod_Exceptions import OutputError, FileSystemError, error_handler, ValidationError
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False
    IOutputManager = object
    IOutputFormatter = object
    BaseComponent = object
    OutputError = Exception
    FileSystemError = Exception
    ValidationError = Exception
    def error_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class OutputFormatter(IOutputFormatter if _HAS_DEPENDENCIES else object):
    """Base class for output formatters."""
    
    def __init__(self, name: str, extension: str, description: str = ""):
        """
        Initialize output formatter.
        
        Args:
            name: Formatter name
            extension: File extension (e.g., '.nc', '.csv')
            description: Formatter description
        """
        self.name = name
        self.extension = extension
        self.description = description
    
    def get_name(self) -> str:
        """Get formatter name."""
        return self.name
    
    def get_extension(self) -> str:
        """Get file extension."""
        return self.extension
    
    def get_description(self) -> str:
        """Get formatter description."""
        return self.description
    
    @abstractmethod
    def format_data(self, data: Any, output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Format and save data."""
        pass
    
    def validate_data(self, data: Any) -> bool:
        """Validate data for formatting."""
        return data is not None


class NetCDFFormatter(OutputFormatter):
    """NetCDF output formatter."""
    
    def __init__(self):
        super().__init__("netcdf", ".nc", "NetCDF format for scientific data")
    
    @error_handler(reraise=True)
    def format_data(self, data: Any, output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Format data as NetCDF."""
        if not self.validate_data(data):
            raise ValidationError("Invalid data for NetCDF formatting")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if isinstance(data, xr.DataArray):
            # Add metadata attributes
            if metadata:
                data.attrs.update(metadata)
            data.to_netcdf(output_path)
        elif isinstance(data, xr.Dataset):
            # Add metadata attributes
            if metadata:
                data.attrs.update(metadata)
            data.to_netcdf(output_path)
        else:
            raise ValidationError(f"NetCDF formatter requires xarray DataArray or Dataset, got {type(data)}")
        
        logging.info(f"Saved NetCDF file: {output_path}")
    
    def validate_data(self, data: Any) -> bool:
        """Validate data for NetCDF formatting."""
        return isinstance(data, (xr.DataArray, xr.Dataset))


class CSVFormatter(OutputFormatter):
    """CSV output formatter."""
    
    def __init__(self):
        super().__init__("csv", ".csv", "Comma-separated values format")
    
    @error_handler(reraise=True)
    def format_data(self, data: Any, output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Format data as CSV."""
        if not self.validate_data(data):
            raise ValidationError("Invalid data for CSV formatting")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        elif isinstance(data, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        else:
            raise ValidationError(f"CSV formatter requires DataFrame, dict, or list of dicts, got {type(data)}")
        
        logging.info(f"Saved CSV file: {output_path}")
    
    def validate_data(self, data: Any) -> bool:
        """Validate data for CSV formatting."""
        return isinstance(data, (pd.DataFrame, dict, list))


class JSONFormatter(OutputFormatter):
    """JSON output formatter."""
    
    def __init__(self):
        super().__init__("json", ".json", "JavaScript Object Notation format")
    
    @error_handler(reraise=True)
    def format_data(self, data: Any, output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Format data as JSON."""
        if not self.validate_data(data):
            raise ValidationError("Invalid data for JSON formatting")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Prepare data for JSON serialization
        if isinstance(data, dict):
            json_data = convert_numpy(data)
            if metadata:
                json_data['metadata'] = convert_numpy(metadata)
        else:
            json_data = {
                'data': convert_numpy(data),
                'metadata': convert_numpy(metadata) if metadata else {}
            }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logging.info(f"Saved JSON file: {output_path}")
    
    def validate_data(self, data: Any) -> bool:
        """Validate data for JSON formatting."""
        # JSON can handle most basic Python types
        return True


class OutputStructure:
    """Class to define output directory structure."""
    
    def __init__(self, base_dir: str):
        """
        Initialize output structure.
        
        Args:
            base_dir: Base output directory
        """
        # base_dir is now the case directory (e.g., output/Debug)
        # All output folders are directly under base_dir, no 'output' subdirectory
        self.base_dir = Path(base_dir)
        self.structure = {
            'metrics': self.base_dir / 'metrics',
            'scores': self.base_dir / 'scores',
            'data': self.base_dir / 'data',
            'comparisons': self.base_dir / 'comparisons',
            'reports': self.base_dir / 'reports',
        }
    
    def create_structure(self) -> None:
        """Create output directory structure."""
        for category, path in self.structure.items():
            path.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created directory: {path}")
    
    def get_path(self, category: str, *subdirs: str) -> Path:
        """
        Get path for a specific category.
        
        Args:
            category: Output category (metrics, scores, etc.)
            *subdirs: Additional subdirectories
            
        Returns:
            Full path to the category directory
        """
        if category not in self.structure:
            raise ValueError(f"Unknown output category: {category}")
        
        path = self.structure[category]
        for subdir in subdirs:
            path = path / subdir
        
        return path
    
    def cleanup_category(self, category: str) -> None:
        """Clean up files in a specific category."""
        if category in self.structure:
            path = self.structure[category]
            if path.exists():
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Cleaned up category: {category}")


class ModularOutputManager(BaseComponent if _HAS_DEPENDENCIES else object):
    """Modular output management system."""
    
    def __init__(self, base_dir: str, name: str = "ModularOutputManager"):
        """
        Initialize output manager.
        
        Args:
            base_dir: Base output directory
            name: Manager name
        """
        if _HAS_DEPENDENCIES:
            super().__init__(name)
        else:
            self.name = name
            self._formatters_registry = {}
        
        self.base_dir = base_dir
        self.structure = OutputStructure(base_dir)
        self.structure.create_structure()
        
        # Register default formatters
        self._register_default_formatters()
        
        # Configuration
        self.auto_create_dirs = True
        self.overwrite_existing = True
        self.backup_existing = False
    
    def _register_default_formatters(self):
        """Register default output formatters."""
        default_formatters = [
            NetCDFFormatter(),
            CSVFormatter(),
            JSONFormatter()
        ]
        
        for formatter in default_formatters:
            self.register_formatter(formatter)
    
    def register_formatter(self, formatter: IOutputFormatter) -> None:
        """Register an output formatter."""
        if _HAS_DEPENDENCIES:
            super().register_formatter(formatter)
        else:
            self._formatters_registry[formatter.get_name()] = formatter
        
        logging.debug(f"Registered formatter: {formatter.get_name()}")
    
    def unregister_formatter(self, formatter_name: str) -> bool:
        """Unregister an output formatter."""
        if formatter_name in self._formatters_registry:
            del self._formatters_registry[formatter_name]
            logging.debug(f"Unregistered formatter: {formatter_name}")
            return True
        return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        if _HAS_DEPENDENCIES:
            return super().get_supported_formats()
        else:
            return list(self._formatters_registry.keys())
    
    def get_formatter_info(self, formatter_name: str) -> Dict[str, str]:
        """Get information about a specific formatter."""
        if formatter_name not in self._formatters_registry:
            raise ValueError(f"Formatter '{formatter_name}' not found")
        
        formatter = self._formatters_registry[formatter_name]
        return {
            'name': formatter.get_name(),
            'extension': formatter.get_extension(),
            'description': formatter.get_description()
        }
    
    @error_handler(reraise=True)
    def save_data(
        self,
        data: Any,
        category: str,
        filename: str,
        format_type: str = 'auto',
        metadata: Optional[Dict[str, Any]] = None,
        subdirs: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Save data to output directory.
        
        Args:
            data: Data to save
            category: Output category (metrics, scores, etc.)
            filename: Output filename (without extension)
            format_type: Output format ('auto', 'netcdf', 'csv', 'json')
            metadata: Optional metadata to include
            subdirs: Optional subdirectories
            **kwargs: Additional formatting options
            
        Returns:
            Path to saved file
        """
        # Determine format
        if format_type == 'auto':
            format_type = self._auto_detect_format(data)
        
        if format_type not in self._formatters_registry:
            raise ValueError(f"Unsupported format: {format_type}")
        
        formatter = self._formatters_registry[format_type]
        
        # Build output path
        subdirs_list = subdirs or []
        output_dir = self.structure.get_path(category, *subdirs_list)
        
        if self.auto_create_dirs:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{filename}{formatter.get_extension()}"
        
        # Handle existing files
        if output_path.exists() and not self.overwrite_existing:
            if self.backup_existing:
                backup_path = output_path.with_suffix(f".backup{formatter.get_extension()}")
                shutil.copy2(output_path, backup_path)
                logging.info(f"Backed up existing file to: {backup_path}")
            else:
                raise FileSystemError(f"File already exists: {output_path}")
        
        # Save data
        formatter.format_data(data, str(output_path), metadata)
        
        return str(output_path)
    
    def _auto_detect_format(self, data: Any) -> str:
        """Auto-detect appropriate format for data."""
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return 'netcdf'
        elif isinstance(data, pd.DataFrame):
            return 'csv'
        elif isinstance(data, (dict, list)):
            return 'json'
        else:
            # Default to JSON for other types
            return 'json'
    
    @error_handler(reraise=True)
    def save_metrics(
        self,
        metrics_data: Dict[str, Any],
        item: str,
        ref_source: str,
        sim_source: str,
        format_type: str = 'auto',
        **kwargs
    ) -> str:
        """
        Save metrics data with standardized naming.
        
        Args:
            metrics_data: Metrics data to save
            item: Evaluation item name
            ref_source: Reference data source
            sim_source: Simulation data source
            format_type: Output format
            **kwargs: Additional options
            
        Returns:
            Path to saved file
        """
        filename = f"{item}_ref_{ref_source}_sim_{sim_source}_metrics"
        
        metadata = {
            'type': 'metrics',
            'item': item,
            'reference_source': ref_source,
            'simulation_source': sim_source,
            'creation_time': pd.Timestamp.now().isoformat()
        }
        
        return self.save_data(
            metrics_data,
            'metrics',
            filename,
            format_type,
            metadata,
            **kwargs
        )
    
    @error_handler(reraise=True)
    def save_scores(
        self,
        scores_data: Dict[str, Any],
        item: str,
        ref_source: str,
        sim_source: str,
        format_type: str = 'auto',
        **kwargs
    ) -> str:
        """
        Save scores data with standardized naming.
        
        Args:
            scores_data: Scores data to save
            item: Evaluation item name
            ref_source: Reference data source
            sim_source: Simulation data source
            format_type: Output format
            **kwargs: Additional options
            
        Returns:
            Path to saved file
        """
        filename = f"{item}_ref_{ref_source}_sim_{sim_source}_scores"
        
        metadata = {
            'type': 'scores',
            'item': item,
            'reference_source': ref_source,
            'simulation_source': sim_source,
            'creation_time': pd.Timestamp.now().isoformat()
        }
        
        return self.save_data(
            scores_data,
            'scores',
            filename,
            format_type,
            metadata,
            **kwargs
        )
    
    @error_handler(reraise=True)
    def save_comparison(
        self,
        comparison_data: Dict[str, Any],
        comparison_name: str,
        format_type: str = 'auto',
        **kwargs
    ) -> str:
        """
        Save comparison results.
        
        Args:
            comparison_data: Comparison data to save
            comparison_name: Name of the comparison
            format_type: Output format
            **kwargs: Additional options
            
        Returns:
            Path to saved file
        """
        metadata = {
            'type': 'comparison',
            'comparison_name': comparison_name,
            'creation_time': pd.Timestamp.now().isoformat()
        }
        
        return self.save_data(
            comparison_data,
            'comparisons',
            comparison_name,
            format_type,
            metadata,
            **kwargs
        )
    
    def list_outputs(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all output files.
        
        Args:
            category: Specific category to list (optional)
            
        Returns:
            Dictionary of category -> list of files
        """
        results = {}
        
        categories = [category] if category else self.structure.structure.keys()
        
        for cat in categories:
            if cat in self.structure.structure:
                cat_path = self.structure.structure[cat]
                if cat_path.exists():
                    files = [f.name for f in cat_path.rglob('*') if f.is_file()]
                    results[cat] = sorted(files)
                else:
                    results[cat] = []
        
        return results
    
    def cleanup_outputs(self, category: Optional[str] = None) -> None:
        """
        Clean up output files.
        
        Args:
            category: Specific category to clean (optional)
        """
        if category:
            self.structure.cleanup_category(category)
        else:
            for cat in self.structure.structure.keys():
                self.structure.cleanup_category(cat)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of output manager state."""
        summary = {
            'base_directory': str(self.base_dir),
            'supported_formats': self.get_supported_formats(),
            'output_categories': list(self.structure.structure.keys()),
            'configuration': {
                'auto_create_dirs': self.auto_create_dirs,
                'overwrite_existing': self.overwrite_existing,
                'backup_existing': self.backup_existing
            }
        }
        
        # Add file counts
        file_counts = {}
        for category, files in self.list_outputs().items():
            file_counts[category] = len(files)
        summary['file_counts'] = file_counts
        
        return summary


# Factory function for creating output managers
def create_output_manager(base_dir: str, **config) -> ModularOutputManager:
    """
    Create an output manager instance.
    
    Args:
        base_dir: Base output directory
        **config: Configuration parameters
        
    Returns:
        Output manager instance
    """
    manager = ModularOutputManager(base_dir)
    
    # Apply configuration
    for key, value in config.items():
        if hasattr(manager, key):
            setattr(manager, key, value)
    
    return manager


# Convenience functions
@error_handler(reraise=True)
def save_evaluation_results(
    manager: ModularOutputManager,
    results: Dict[str, Any],
    item: str,
    ref_source: str,
    sim_source: str,
    result_type: str = 'metrics'
) -> str:
    """
    Convenience function for saving evaluation results.
    
    Args:
        manager: Output manager instance
        results: Results data
        item: Evaluation item
        ref_source: Reference source
        sim_source: Simulation source
        result_type: Type of results ('metrics' or 'scores')
        
    Returns:
        Path to saved file
    """
    if result_type == 'metrics':
        return manager.save_metrics(results, item, ref_source, sim_source)
    elif result_type == 'scores':
        return manager.save_scores(results, item, ref_source, sim_source)
    else:
        raise ValueError(f"Unknown result type: {result_type}")